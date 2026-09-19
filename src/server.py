"""A local TCP front-end for `HHAgent`.

The wire protocol is newline-delimited JSON: one JSON object per line, in both
directions. Clients send commands, the server pushes events. Every event
carries `event`, a per-connection `seq` and a `ts`, plus `agent_id` where it
applies.

The registry holds agent blocks, which form chains: `create_agent` makes a root
block, `fork` appends a block holding a user prompt, and `run` executes that
block's single turn. Nothing is copied between blocks, so a chain of a thousand
messages still stores each message once. Rewinding is just forking from an
earlier block again; the blocks beyond it stay untouched.

Commands
    create_agent   {id?, endpoint?, key?, model?, tools?, timeout?,
                    max_tokens?, include_usage?,
                    verbose?}                          makes a root block
    fork           {id, prompt, images?, new_id?, model?, tools?, timeout?,
                    max_tokens?, include_usage?,
                    verbose?}                          appends a block
    run            {id}                                executes a block's turn
    cancel         {id}                                stop a running turn
    get_context    {id}                                flattened message list
    get_state      {id, tool?}                         rebuilt tool state
    set_state      {id, tool, key, value?/delete?}     seed or drop a state key
    resolve_tool   {id, call_id, result?, images?, error?,
                    call?}                             answer a local tool call
    list_agents    {}                                  every block, with links
    destroy_agent  {id}                                drop a block and its subtree
    ping           {echo?}

Events
    session_hello, command_received, command_finished, error, session_closing
    agent_created, agent_forked, agent_destroyed, agents_listed, context,
    state, state_seeded, cancel_result, pong, evicted, persist_warning,
    local_tool_answered
    and everything `HHAgent` emits while a block runs: turn_started,
    turn_finished, turn_failed, turn_cancelled, request_started,
    request_payload, response_received, request_finished, request_failed,
    content_delta, reasoning_delta, sse_chunk (verbose), sse_unparsed, usage,
    assistant_message, history_appended, tool_call_requested,
    tool_call_started, tool_call_finished, pipe_step_started,
    pipe_step_finished, state_loaded, state_delta,
    state_discarded, local_tool_called, local_tool_rollback,
    local_tool_resolved, local_tool_unresolved, rollback_started,
    rollback_finished, rollback_unavailable, failure_summary_started,
    failure_summary_finished, failure_summary_failed

When a turn does not commit — it failed, was cancelled, or was abandoned — every
tool it called is offered a rollback, newest call first, so effects outside the
state can be undone. A rollback that fails is reported and skipped; the rest
still run. Client-run tools are asked over the protocol like any other local
call.

A turn that ends without committing also leaves a note in its block, because the
transcript above it may still describe tool effects the rollback has undone: a
failed turn's note is summarised by the model (`failure_summary_*` events) and
always carries the raw error, while a cancelled or abandoned one uses fixed text.
The note is an ordinary message, so a block forked from that block carries it
along.

A block may also carry `local_tools`: tools the *client* runs. They are offered
to the model with everything else, and when one is called the server emits
`local_tool_called` and parks that turn until a `resolve_tool` command arrives
with the result. The wait holds no lock and touches no database, so the rest of
the server keeps working while a client decides.

A tool does not have to answer with text. A server hook that returns a
`ToolCall` — or a `ToolResult` carrying one — starts a **pipe**: the harness runs
the named tool next, and keeps following calls until one returns text. A local
tool can do the same by answering `resolve_tool` with `call` instead of `result`,
and either kind may hand off to the other. Only the tools a pipe called and the
last call's output are shown to the model: the tool message becomes
`[tool pipe] first -> second\n<the last result>`. Each step's arguments and
result are recorded in memory (`get_context` returns them as `pipe_traces`)
and reported by the `pipe_step_started` / `pipe_step_finished` events, but never
sent to the provider or written to the database — which is the point, since a
step may carry a file's bytes that the model must not have to quote, and that a
restart must not be asked to keep. A pipe runs without the model in the
loop, so it is bounded to `MAX_PIPE_DEPTH` calls before it is cut off with a
`pipe_depth` error; every step that reached a tool is rolled back with the turn
if the turn does not commit.

With `--db` the registry is mirrored to a SQLite file: a block is written when
it is forked and again when its turn ends, so a process that dies mid-turn
leaves the block forked-but-never-run. The API key is never written; restored
blocks inherit the key the server was started with, and tools are stored by name
and resolved against the catalogue on load. `--max-db-bytes` bounds the file:
once it is exceeded, the oldest subtrees are evicted, where a subtree is a root
tree or a block whose parent has other forks, aged by the newest block inside it.

Run it with `python src/server.py`, then talk to it, e.g.

    printf '%s\\n' '{"command":"ping","rid":"1"}' | nc 127.0.0.1 8765
"""

import argparse
import importlib.util
import json
import pathlib
import socketserver
import sqlite3
import threading
import time
import traceback
from typing import Any, cast

from agent import (
    DEFAULT_LOCAL_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    HHAgent,
    HHAgentCancelled,
    HHAgentError,
    StateDelta,
    image_content_parts,
)
from protocol import PROTOCOL_VERSION
from sandbox_tools import ADD_FILE_MAX_BYTES
from store import HHStore, HHStoreError
from tools import ToolCall, ToolEntry, ToolParam, builtin_tools

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# The longest command line the server reads. It is sized from `nix_add_file`:
# a client answers `resolve_tool` with a pipe call handing the tool a whole file
# as base64, about 4/3 of the file's bytes, so 200 MiB arrives as a ~267 MiB
# line plus the JSON around it. A server-side pipe never crosses this line, so
# it is bound only by the tool's own cap.
MAX_COMMAND_BYTES = ADD_FILE_MAX_BYTES * 4 // 3 + 2 * 1024 * 1024
DEFAULT_MAX_DB_BYTES = 64 * 1024 * 1024

COMMANDS: list[dict[str, Any]] = [
    {
        "command": "create_agent",
        "summary": "Create a root block; it holds no prompt until forked.",
        "fields": {
            "id": "optional block id, auto-generated when omitted",
            "endpoint": "optional, falls back to the server default",
            "key": "optional, falls back to the server default",
            "model": "optional, falls back to the server default",
            "tools": "optional list of names, defaults to every builtin",
            "local_tools": "optional list of tool definitions the client runs; "
            "each may set \"rollback\": true to be asked to undo a failed turn, "
            "and \"external_effects\": true to be named in the failure note when "
            "no undo is available",
            "timeout": "optional read timeout in seconds",
            "local_timeout": "optional seconds to wait for a local tool answer",
            "max_tokens": "optional output-token cap; 0 leaves it to the provider",
            "summary_model": "optional model for failure summaries; defaults to the block's own",
            "include_usage": "optional bool, ask for token accounting",
            "verbose": "optional bool, also emit raw sse_chunk events",
        },
    },
    {
        "command": "fork",
        "summary": "Append a block holding a user prompt; it starts dirty.",
        "fields": {
            "id": "required, the block to fork from",
            "prompt": "required, non-empty string",
            "images": (
                "optional list of http(s) URLs or data:image/... URIs; each a "
                "string or {url, detail}"
            ),
            "new_id": "optional id for the new block",
            "model": "optional override inherited from the parent",
            "tools": "optional override inherited from the parent",
            "local_tools": "optional override, replaces the inherited definitions",
            "timeout": "optional override inherited from the parent",
            "local_timeout": "optional override inherited from the parent",
            "max_tokens": "optional override inherited from the parent",
            "summary_model": "optional override inherited from the parent",
            "include_usage": "optional override inherited from the parent",
            "verbose": "optional override inherited from the parent",
        },
    },
    {
        "command": "run",
        "summary": "Execute a block's turn; events stream back and dirty clears.",
        "fields": {"id": "required, a forked block that has not run yet"},
    },
    {
        "command": "cancel",
        "summary": "Ask a running turn to stop.",
        "fields": {"id": "required"},
    },
    {
        "command": "get_context",
        "summary": "Return the flattened message list of a block's whole chain.",
        "fields": {"id": "required"},
    },
    {
        "command": "list_agents",
        "summary": "List every block with its parent, depth and state.",
        "fields": {},
    },
    {
        "command": "destroy_agent",
        "summary": "Drop a block and everything forked from it.",
        "fields": {"id": "required"},
    },
    {
        "command": "get_state",
        "summary": "Return a block's rebuilt tool state.",
        "fields": {
            "id": "required",
            "tool": "optional tool name; omit for every touched tool",
        },
    },
    {
        "command": "set_state",
        "summary": "Seed or drop one key of a block's own state deltas.",
        "fields": {
            "id": "required, a block that is neither running nor dirty",
            "tool": "required tool name",
            "key": "required top-level state key",
            "value": "required unless 'delete' is true",
            "delete": "optional bool, drop the key instead of setting it",
        },
    },
    {
        "command": "resolve_tool",
        "summary": "Answer a local tool call that a turn is waiting on.",
        "fields": {
            "id": "required, the block whose turn is waiting",
            "call_id": "required, from the local_tool_called event",
            "result": "the text handed back to the model; optional when 'images' or 'call' is given",
            "images": "optional list of http(s) URLs or data:image/... URIs the tool returned",
            "error": "optional; marks the call failed and is reported as such",
            "call": (
                "optional {name, arguments} for the tool to run next, continuing "
                "the pipe instead of ending it; cannot be combined with 'error' "
                "or 'images'"
            ),
        },
    },
    {"command": "ping", "summary": "Liveness probe.", "fields": {"echo": "optional"}},
]

# Settings a fork may override on the block it creates.
INHERITED = (
    "model", "timeout", "local_timeout", "max_tokens", "summary_model",
    "include_usage", "verbose",
)
BOOL_FIELDS = ("include_usage", "verbose")
FLOAT_FIELDS = ("timeout", "local_timeout")
INT_FIELDS = ("max_tokens",)


class HHTcpError(Exception):
    """An error to report back to the client as an `error` event."""

    def __init__(self, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


class Connection:
    """One client connection: JSONL framing with serialized event writes.

    Turns run on their own threads, so `send` is guarded by a lock and events
    from different blocks may interleave; `seq` stays monotonic per connection.
    """

    def __init__(self, sock: Any, peer: tuple[str, int]) -> None:
        self.sock = sock
        self.peer = peer
        self.closed = False
        self.commands = 0
        self._seq = 0
        self._lock = threading.Lock()

    def send(self, event: str, **fields: Any) -> bool:
        """Write one event line. Returns False once the peer is gone."""
        with self._lock:
            if self.closed:
                return False
            self._seq += 1
            payload = {
                **fields,
                "event": event,
                "seq": self._seq,
                "ts": time.time(),
            }
            line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
            try:
                self.sock.sendall(line.encode("utf-8"))
            except OSError:
                self.closed = True
                return False
            return True

    def forward(self, event: dict[str, Any]) -> None:
        """Relay an event produced by `HHAgent`."""
        name = event.get("event", "event")
        self.send(name, **{key: value for key, value in event.items() if key != "event"})

    def close(self) -> None:
        with self._lock:
            self.closed = True

    @property
    def events(self) -> int:
        """How many events have been written to this connection."""
        return self._seq


class HHHandler(socketserver.StreamRequestHandler):
    """Reads command lines for one client and hands them to the server."""

    disable_nagle_algorithm = True

    def handle(self) -> None:
        server = cast(HHServer, self.server)
        conn = Connection(self.request, self.client_address)
        conn.send(
            "session_hello",
            protocol=PROTOCOL_VERSION,
            server="headless-harness",
            peer=list(self.client_address),
            started_at=server.started_at,
            defaults=server.describe_defaults(),
            commands=COMMANDS,
            tools=tool_catalogue(),
            store=server.describe_store(),
        )
        try:
            while True:
                line = self.rfile.readline(MAX_COMMAND_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_COMMAND_BYTES:
                    conn.send(
                        "error",
                        code="command_too_large",
                        message=f"command exceeds {MAX_COMMAND_BYTES} bytes",
                        bytes=len(line),
                    )
                    break
                text = line.decode("utf-8", errors="replace").strip()
                # a line may be a ~268 MiB base64 payload, so each copy is let
                # go as soon as the next one exists
                del line
                if not text:
                    continue
                try:
                    command = json.loads(text)
                except json.JSONDecodeError as exc:
                    conn.send(
                        "error",
                        code="bad_json",
                        message=str(exc),
                        line=text[:200],
                    )
                    continue
                del text  # the parsed command owns its strings
                if not isinstance(command, dict):
                    conn.send(
                        "error",
                        code="bad_command",
                        message="every line must be a JSON object",
                        got=type(command).__name__,
                    )
                    continue
                server.dispatch(conn, command)
        except (ConnectionError, OSError):
            pass  # the peer went away mid-write; turn threads notice via conn.closed
        finally:
            conn.send("session_closing", commands=conn.commands, events_sent=conn.events)
            conn.close()


class HHServer(socketserver.ThreadingTCPServer):
    """Threaded TCP listener; one thread per connection, one per running turn."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 16

    def __init__(
        self,
        address: tuple[str, int],
        defaults: dict[str, str | None] | None = None,
        store: HHStore | None = None,
    ) -> None:
        super().__init__(address, HHHandler)
        self.defaults: dict[str, str | None] = dict(defaults or {})
        self.started_at = time.time()
        self._agents: dict[str, HHAgent] = {}
        self._registry_lock = threading.RLock()
        self.store = store
        self.restore_warnings: list[str] = []
        if store is not None:
            blocks, self.restore_warnings = store.load(
                str(self.defaults.get("key") or "")
            )
            self._agents = {block.id: block for block in blocks}
            if self._agents:
                print(
                    f"restored {len(self._agents)} block(s) from {store.path}",
                    flush=True,
                )
            for warning in self.restore_warnings:
                print(f"restore warning: {warning}", flush=True)

    def describe_store(self) -> dict[str, Any] | None:
        """What a client should know about persistence when it connects."""
        if self.store is None:
            return None
        info: dict[str, Any] = dict(self.store.stats())
        info["warnings"] = self.restore_warnings
        return info

    def describe_defaults(self) -> dict[str, Any]:
        """What created blocks inherit; the key is never echoed back."""
        return {
            "endpoint": self.defaults.get("endpoint"),
            "model": self.defaults.get("model"),
            "has_key": bool(self.defaults.get("key")),
        }

    def dispatch(self, conn: Connection, command: dict[str, Any]) -> None:
        """Route one command, always bracketing it with lifecycle events."""
        name = command.get("command")
        rid = command.get("rid")
        conn.commands += 1
        conn.send(
            "command_received",
            rid=rid,
            command=name,
            keys=sorted(str(key) for key in command),
            # the block this command targets, if any; for `fork` it is the parent
            target=command.get("id"),
        )
        started = time.monotonic()
        if not isinstance(name, str):
            conn.send(
                "error",
                code="bad_command",
                message="'command' must be a string",
                rid=rid,
                commands=[entry["command"] for entry in COMMANDS],
            )
            return
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            conn.send(
                "error",
                code="unknown_command",
                message=f"unknown command {name!r}",
                rid=rid,
                command=name,
                commands=[entry["command"] for entry in COMMANDS],
            )
            return
        try:
            handler(conn, command, rid)
        except HHTcpError as exc:
            conn.send(
                "error",
                code=exc.code,
                message=str(exc),
                rid=rid,
                command=name,
                detail=exc.detail,
            )
            return
        except Exception as exc:
            conn.send(
                "error",
                code="internal_error",
                message=f"{type(exc).__name__}: {exc}",
                rid=rid,
                command=name,
                traceback=traceback.format_exc()[-2000:],
            )
            return
        if name != "run":
            # a run's turn finishes on its own thread and reports from there
            conn.send(
                "command_finished",
                rid=rid,
                command=name,
                target=command.get("id"),
                status="ok",
                elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            )

    def cmd_create_agent(
        self, conn: Connection, command: dict[str, Any], rid: Any
    ) -> None:
        agent_id = command.get("id")
        if agent_id is not None and (not isinstance(agent_id, str) or not agent_id):
            raise HHTcpError("bad_id", "'id' must be a non-empty string")
        endpoint = command.get("endpoint") or self.defaults.get("endpoint")
        key = command.get("key") or self.defaults.get("key")
        if not endpoint or not key:
            raise HHTcpError(
                "missing_credentials",
                "create_agent needs 'endpoint' and 'key', or start the server "
                "with --endpoint/--key",
            )
        tools = build_tools(
            select_tools(command.get("tools")),
            parse_local_tools(command.get("local_tools")),
        )
        block = HHAgent.root(
            endpoint,
            key,
            model=command.get("model") or self.defaults.get("model") or DEFAULT_MODEL,
            tools=list(tools.values()),
            timeout=as_timeout(command.get("timeout") or DEFAULT_TIMEOUT, "timeout"),
            local_timeout=as_timeout(
                command.get("local_timeout") or DEFAULT_LOCAL_TIMEOUT, "local_timeout"
            ),
            max_tokens=as_max_tokens(
                DEFAULT_MAX_TOKENS
                if command.get("max_tokens") is None
                else command["max_tokens"],
                "max_tokens",
            ),
            summary_model=command.get("summary_model") or "",
            include_usage=bool(command.get("include_usage", True)),
            verbose=bool(command.get("verbose", False)),
            id=agent_id,
        )
        with self._registry_lock:
            if block.id in self._agents:
                raise HHTcpError(
                    "duplicate_agent",
                    f"agent {block.id!r} already exists",
                    detail={"agents": sorted(self._agents)},
                )
            self._agents[block.id] = block
        conn.send(
            "agent_created",
            rid=rid,
            agent_id=block.id,
            parent=None,
            depth=0,
            dirty=block.dirty,
            model=block.model,
            endpoint=block.endpoint,
            timeout=block.timeout,
            summary_model=block.summary_model,
            include_usage=block.include_usage,
            verbose=block.verbose,
            max_tokens=block.max_tokens,
            tools=sorted(block.tools),
            local_tools=sorted(t.name for t in block.tools.values() if t.is_local),
            tool_schemas=block.tool_schemas(),
        )
        self.persist(conn, block)

    def cmd_fork(self, conn: Connection, command: dict[str, Any], rid: Any) -> None:
        parent_id = require_id(command)
        parent = self.block(parent_id)
        prompt = command.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HHTcpError("bad_prompt", "'prompt' must be a non-empty string")
        new_id = command.get("new_id")
        if new_id is not None and (not isinstance(new_id, str) or not new_id):
            raise HHTcpError("bad_id", "'new_id' must be a non-empty string")
        images = parse_images(command.get("images"))
        try:
            child = parent.fork(prompt, id=new_id, images=images)
        except HHAgentError as exc:
            raise HHTcpError(
                "parent_dirty",
                str(exc),
                detail={"agent_id": parent_id, "dirty": parent.dirty},
            ) from exc
        apply_overrides(child, command)
        with self._registry_lock:
            if child.id in self._agents:
                raise HHTcpError(
                    "duplicate_agent",
                    f"agent {child.id!r} already exists",
                    detail={"agents": sorted(self._agents)},
                )
            self._agents[child.id] = child
        conn.send(
            "agent_forked",
            rid=rid,
            agent_id=child.id,
            parent=parent_id,
            depth=child.depth,
            dirty=child.dirty,
            prompt=child.prompt,
            prompt_chars=len(child.prompt),
            image_count=child.image_count,
            path=child.path(),
            context_len=len(child.context()),
            model=child.model,
            timeout=child.timeout,
            include_usage=child.include_usage,
            verbose=child.verbose,
            max_tokens=child.max_tokens,
            tools=sorted(child.tools),
            local_tools=sorted(t.name for t in child.tools.values() if t.is_local),
            state_namespaces=child.state_namespaces(),
        )
        self.persist(conn, child)

    def cmd_run(self, conn: Connection, command: dict[str, Any], rid: Any) -> None:
        agent_id = require_id(command)
        block = self.block(agent_id)
        if block.parent is None:
            raise HHTcpError(
                "root_agent",
                f"agent {agent_id!r} is a root; fork from it to get a prompt",
            )
        if block.running:
            raise HHTcpError(
                "agent_running", f"agent {agent_id!r} is already running"
            )
        if not block.dirty:
            raise HHTcpError(
                "agent_finished",
                f"agent {agent_id!r} has already finished its turn",
                detail={"error": block.error},
            )
        conn.send(
            "run_accepted",
            rid=rid,
            agent_id=agent_id,
            depth=block.depth,
            context_len=len(block.context()),
        )
        threading.Thread(
            target=self._run_turn,
            args=(conn, block, rid),
            name=f"turn-{agent_id}",
            daemon=True,
        ).start()

    def cmd_cancel(self, conn: Connection, command: dict[str, Any], rid: Any) -> None:
        agent_id = require_id(command)
        block = self.block(agent_id)
        conn.send(
            "cancel_result",
            rid=rid,
            agent_id=agent_id,
            cancelled=block.cancel(),
            running=block.running,
            dirty=block.dirty,
        )

    def cmd_get_context(
        self, conn: Connection, command: dict[str, Any], rid: Any
    ) -> None:
        agent_id = require_id(command)
        block = self.block(agent_id)
        context = block.context()
        conn.send(
            "context",
            rid=rid,
            agent_id=agent_id,
            depth=block.depth,
            path=block.path(),
            context=context,
            context_len=len(context),
            local_len=len(block.messages),
            messages=block.messages,
            # what the block's own tool pipes did, step by step, with their
            # intermediate values bounded: for inspection only, never for the
            # model and never for the store
            pipe_traces=block.pipe_traces,
        )

    def cmd_list_agents(
        self, conn: Connection, command: dict[str, Any], rid: Any
    ) -> None:
        with self._registry_lock:
            agents = [self.describe(block) for block in self._agents.values()]
            roots = [block.id for block in self._agents.values() if block.parent is None]
        conn.send(
            "agents_listed",
            rid=rid,
            count=len(agents),
            roots=sorted(roots),
            dirty=[entry["agent_id"] for entry in agents if entry["dirty"]],
            agents=agents,
        )

    def cmd_destroy_agent(
        self, conn: Connection, command: dict[str, Any], rid: Any
    ) -> None:
        agent_id = require_id(command)
        with self._registry_lock:
            target = self._agents.get(agent_id)
            if target is None:
                raise HHTcpError(
                    "unknown_agent",
                    f"no agent {agent_id!r}",
                    detail={"agents": sorted(self._agents)},
                )
            # a block is only reachable through its parent, so its whole
            # subtree goes with it
            doomed = [
                block
                for block in self._agents.values()
                if target in block.lineage()
            ]
            for block in doomed:
                del self._agents[block.id]
        if self.store is not None:
            self.store.delete(agent_id)
        cancelled = [block.id for block in doomed if block.cancel()]
        conn.send(
            "agent_destroyed",
            rid=rid,
            agent_id=agent_id,
            dropped=[block.id for block in doomed],
            count=len(doomed),
            cancelled=cancelled,
            age_ms=round((time.time() - target.created_at) * 1000, 3),
            remaining=len(self._agents),
        )

    def cmd_get_state(
        self, conn: Connection, command: dict[str, Any], rid: Any
    ) -> None:
        agent_id = require_id(command)
        block = self.block(agent_id)
        tool = command.get("tool")
        if tool is not None and (not isinstance(tool, str) or not tool):
            raise HHTcpError("bad_field", "'tool' must be a non-empty string")
        names = [namespace_for(tool)] if tool else block.state_namespaces()
        conn.send(
            "state",
            rid=rid,
            agent_id=agent_id,
            depth=block.depth,
            path=block.path(),
            tool=tool,
            keys=names,
            state={name: block.merged_state(name) for name in names},
        )

    def cmd_set_state(
        self, conn: Connection, command: dict[str, Any], rid: Any
    ) -> None:
        """Seed a block's own deltas. Only a finished block may be written to.

        A running block would race the turn, and a dirty one would need the seed
        folded into the turn's own diff at commit time. Seeding a finished block
        — a root, typically — is enough to give a whole subtree initial state.

        `tool` names a tool, which is resolved to the namespace it shares with
        other tools; `key` is the field inside that namespace.
        """
        agent_id = require_id(command)
        block = self.block(agent_id)
        if block.running:
            raise HHTcpError(
                "agent_running", f"agent {agent_id!r} is mid-turn; cancel it first"
            )
        if block.dirty:
            raise HHTcpError(
                "agent_dirty",
                f"agent {agent_id!r} has not run yet; seed the block it forked from",
            )
        tool = command.get("tool")
        if not isinstance(tool, str) or not tool:
            raise HHTcpError("bad_field", "'tool' must be a non-empty string")
        namespace = namespace_for(tool)
        key = command.get("key")
        if not isinstance(key, str) or not key:
            raise HHTcpError("bad_field", "'key' must be a non-empty string")
        drop = bool(command.get("delete", False))
        if not drop and "value" not in command:
            raise HHTcpError("bad_field", "set_state needs 'value', or 'delete': true")
        replaced = key in block.merged_state(namespace)
        delta = block.state_deltas.get(namespace)
        if delta is None:
            delta = StateDelta()
            block.state_deltas[namespace] = delta
        if drop:
            delta.changed.pop(key, None)
            if key not in delta.removed:
                delta.removed = (*delta.removed, key)
        else:
            delta.changed[key] = command["value"]
            delta.removed = tuple(name for name in delta.removed if name != key)
        if not delta.changed and not delta.removed:
            del block.state_deltas[namespace]
        conn.send(
            "state_seeded",
            rid=rid,
            agent_id=agent_id,
            tool=tool,
            state_namespace=namespace,
            key=key,
            value=command.get("value"),
            deleted=drop,
            replaced=replaced,
            keys=sorted(block.merged_state(namespace)),
        )
        self.persist(conn, block)

    def cmd_resolve_tool(
        self, conn: Connection, command: dict[str, Any], rid: Any
    ) -> None:
        """Hand a client-run tool's result to the turn that is waiting for it.

        Deliberately allowed on a running block: the whole point is that a turn
        is parked waiting for this. Nothing here blocks — resolving sets an
        event, and the turn thread picks the answer up.
        """
        agent_id = require_id(command)
        block = self.block(agent_id)
        call_id = command.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise HHTcpError("bad_field", "'call_id' must be a non-empty string")
        error = command.get("error")
        if error is not None and (not isinstance(error, str) or not error):
            raise HHTcpError("bad_field", "'error' must be a non-empty string")
        images = parse_images(command.get("images"))
        call = None
        if command.get("call") is not None:
            call = parse_tool_call(command["call"])
            if error is not None:
                raise HHTcpError(
                    "bad_call", "'error' and 'call' cannot both be given"
                )
            if images:
                raise HHTcpError(
                    "bad_call",
                    "'images' cannot ride a pipe: only the last call of a pipe "
                    "may return images",
                )
        if "result" not in command and error is None and not images and call is None:
            raise HHTcpError(
                "bad_field", "resolve_tool needs 'result', 'images', 'error' or 'call'"
            )
        result = command.get("result")
        if result is None:
            result = ""
        elif not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        if not block.resolve_local_call(call_id, result, error, images, call):
            raise HHTcpError(
                "unknown_call",
                f"agent {agent_id!r} is not waiting on {call_id!r}",
                detail={"pending": block.pending_calls()},
            )
        conn.send(
            "local_tool_answered",
            rid=rid,
            agent_id=agent_id,
            call_id=call_id,
            ok=error is None,
            result_chars=len(result),
            image_count=len(images),
            # the tool that runs next, when the answer kept a pipe going
            next=call.name if call is not None else None,
        )

    def cmd_ping(self, conn: Connection, command: dict[str, Any], rid: Any) -> None:
        with self._registry_lock:
            total = len(self._agents)
            dirty = sum(1 for block in self._agents.values() if block.dirty)
            running = sum(1 for block in self._agents.values() if block.running)
        conn.send(
            "pong",
            rid=rid,
            echo=command.get("echo"),
            server_time=time.time(),
            uptime_ms=round((time.time() - self.started_at) * 1000, 3),
            agents=total,
            dirty=dirty,
            running=running,
            threads=threading.active_count(),
            store=self.store.stats() if self.store else None,
        )

    def _run_turn(self, conn: Connection, block: HHAgent, rid: Any) -> None:
        """One block's turn, on its own thread, so the connection stays live."""
        started = time.monotonic()
        status, error, error_type = "ok", None, None
        try:
            for _chunk in block.stream(on_event=conn.forward):
                if conn.closed:
                    # nobody is listening any more; stop burning tokens
                    block.cancel()
        except HHAgentCancelled as exc:
            status, error, error_type = "cancelled", str(exc), type(exc).__name__
        except HHAgentError as exc:
            status, error, error_type = "error", str(exc), type(exc).__name__
        except Exception as exc:
            status, error, error_type = "error", str(exc), type(exc).__name__
        # durability before the completion event: a client that sees this turn
        # finish, then loses the process, must not lose the turn's result
        self.persist(conn, block)
        conn.send(
            "command_finished",
            rid=rid,
            command="run",
            agent_id=block.id,
            status=status,
            error=error,
            error_type=error_type,
            elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            dirty=block.dirty,
            messages=len(block.messages),
            context_len=len(block.context()),
            state_committed=sorted(block.state_deltas),
        )

    def persist(self, conn: Connection, block: HHAgent) -> None:
        """Mirror a block to disk, then evict if the file outgrew its budget.

        A block destroyed or evicted while its turn was running is not written
        back: re-inserting it would resurrect a subtree the caller dropped.
        """
        store = self.store
        if store is None:
            return
        # the existence check and the write share one lock hold: if they were
        # split, a concurrent eviction could delete this block's parent between
        # them and the insert would fail its foreign key
        with self._registry_lock:
            if self._agents.get(block.id) is not block:
                return
            dropped = store.save(block)
        if dropped:
            conn.send(
                "persist_warning",
                agent_id=block.id,
                dropped_tools=dropped,
                message="state for these tools could not be stored",
            )
        self.enforce_limit(conn)

    def enforce_limit(self, conn: Connection) -> None:
        """Drop whole subtrees, oldest first, until the file fits its budget.

        Candidates come from the store; protection comes from here, because only
        the registry knows which blocks have a turn in flight. A subtree with a
        live turn is skipped, which makes the budget a soft bound: the file can
        overshoot until that turn ends, and the next write tries again.
        """
        store = self.store
        if store is None or store.max_bytes <= 0:
            return
        while store.size_bytes() > store.max_bytes:
            with self._registry_lock:
                # picking a victim, deleting its rows and dropping it from the
                # registry share one lock hold: they are all writes to the same
                # two structures, and a turn writing the same subtree must not
                # slip between them (its insert would fail the foreign key)
                running = {
                    agent_id
                    for agent_id, candidate in self._agents.items()
                    if candidate.running
                }
                victim: sqlite3.Row | None = None
                victim_ids: set[str] = set()
                for row in store.candidates():
                    ids = set(store.subtree_ids(row["root"]))
                    if ids & running:
                        continue
                    victim, victim_ids = row, ids
                    break
                if victim is None:
                    break  # all protected; retry on the next write
                store.delete(victim["root"])
                for agent_id in victim_ids:
                    self._agents.pop(agent_id, None)
            conn.send(
                "evicted",
                agent_id=victim["root"],
                dropped=sorted(victim_ids),
                nodes=victim["nodes"],
                newest=victim["newest"],
                bytes=store.size_bytes(),
                max_bytes=store.max_bytes,
            )

    def block(self, agent_id: str) -> HHAgent:
        with self._registry_lock:
            block = self._agents.get(agent_id)
        if block is None:
            raise HHTcpError(
                "unknown_agent",
                f"no agent {agent_id!r}",
                detail={"agents": sorted(self._agents)},
            )
        return block

    def describe(self, block: HHAgent) -> dict[str, Any]:
        return {
            "agent_id": block.id,
            "parent": block.parent.id if block.parent else None,
            "depth": block.depth,
            "dirty": block.dirty,
            "running": block.running,
            "outcome": block.outcome,
            "error": block.error,
            "prompt_chars": len(block.prompt),
            "prompt_preview": block.prompt[:200],
            "image_count": block.image_count,
            "text_chars": len(block.text),
            "local_len": len(block.messages),
            "context_len": len(block.context()),
            "model": block.model,
            "summary_model": block.summary_model,
            "tools": sorted(block.tools),
            "state_namespaces": block.state_namespaces(),
            "local_tools": sorted(
                t.name for t in block.tools.values() if t.is_local
            ),
            "rollback_tools": sorted(
                t.name for t in block.tools.values() if t.has_rollback
            ),
            "waiting_on": block.pending_calls(),
            "pipe_traces": len(block.pipe_traces),
            "include_usage": block.include_usage,
            "verbose": block.verbose,
            "max_tokens": block.max_tokens,
            "created_at": block.created_at,
            "age_ms": round((time.time() - block.created_at) * 1000, 3),
        }


def require_id(command: dict[str, Any]) -> str:
    agent_id = command.get("id")
    if not isinstance(agent_id, str) or not agent_id:
        raise HHTcpError("bad_id", "'id' must be a non-empty string")
    return agent_id


def parse_images(value: Any) -> list[dict[str, Any]]:
    """Check the wire `images` field and render it as content parts."""
    try:
        return image_content_parts(value)
    except HHAgentError as exc:
        raise HHTcpError("bad_image", str(exc)) from exc


def parse_tool_call(value: Any) -> ToolCall:
    """Check a wire `call` field and render it as a `ToolCall`."""
    if not isinstance(value, dict):
        raise HHTcpError(
            "bad_call", "'call' must be an object with 'name' and 'arguments'"
        )
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise HHTcpError("bad_call", "'call' needs a non-empty 'name'")
    arguments = value.get("arguments", {})
    if not isinstance(arguments, dict):
        raise HHTcpError("bad_call", "'call' arguments must be an object")
    return ToolCall(name=name, arguments=arguments)


def apply_overrides(block: HHAgent, command: dict[str, Any]) -> None:
    """Apply the settings a fork is allowed to change on its new block."""
    for name in INHERITED:
        value = command.get(name)
        if value is None:
            continue
        if name in BOOL_FIELDS:
            if not isinstance(value, bool):
                raise HHTcpError("bad_field", f"{name!r} must be a boolean")
            setattr(block, name, value)
        elif name in FLOAT_FIELDS:
            setattr(block, name, as_timeout(value, name))
        elif name in INT_FIELDS:
            setattr(block, name, as_max_tokens(value, name))
        else:
            if not isinstance(value, str) or not value:
                raise HHTcpError(
                    "bad_field", f"{name!r} must be a non-empty string"
                )
            setattr(block, name, value)
    # `tools` and `local_tools` are separate axes: whichever is supplied
    # replaces that half, and the other half carries over untouched
    catalogue = None
    if command.get("tools") is not None:
        catalogue = select_tools(command["tools"])
    local = None
    if command.get("local_tools") is not None:
        local = parse_local_tools(command["local_tools"])
    if catalogue is not None or local is not None:
        if catalogue is None:
            catalogue = [t for t in block.tools.values() if not t.is_local]
        if local is None:
            local = [t for t in block.tools.values() if t.is_local]
        block.tools = build_tools(catalogue, local)


def namespace_for(name: str) -> str:
    """The state namespace a name refers to.

    Clients address state by tool name, and a tool's state lives in the
    namespace it shares with others of the same `state_namespace`. A name that is not
    a known tool is taken to be a namespace already, so a tool that has since
    been removed stays reachable.
    """
    for tool in builtin_tools:
        if tool.name == name:
            return tool.namespace
    return name


def as_timeout(value: Any, field: str) -> float:
    """Coerce a timeout field, rejecting nonsense before it reaches a block."""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise HHTcpError("bad_field", f"{field!r} must be a number: {exc}") from exc
    if timeout <= 0:
        raise HHTcpError("bad_field", f"{field!r} must be positive")
    return timeout


def as_max_tokens(value: Any, field: str) -> int:
    """Coerce an output-token cap; `0` means no cap and is allowed."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise HHTcpError("bad_field", f"{field!r} must be an integer")
    try:
        cap = int(value)
    except ValueError as exc:
        raise HHTcpError("bad_field", f"{field!r} must be an integer: {exc}") from exc
    if cap < 0:
        raise HHTcpError("bad_field", f"{field!r} must not be negative")
    return cap


def build_tools(
    catalogue: list[ToolEntry], local: list[ToolEntry]
) -> dict[str, ToolEntry]:
    """Merge server-run and client-run tools, refusing name collisions."""
    tools: dict[str, ToolEntry] = {}
    for tool in (*catalogue, *local):
        if tool.name in tools:
            raise HHTcpError("bad_tools", f"tool {tool.name!r} is declared twice")
        tools[tool.name] = tool
    return tools


def parse_local_tools(value: Any) -> list[ToolEntry]:
    """Build client-run tool definitions from a `local_tools` command field.

    A local tool is only a schema — name, description, parameters — because the
    hook lives on the client. The server offers it to the model like any other
    and asks for the result when it is called.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise HHTcpError("bad_tools", "'local_tools' must be a list of definitions")
    parsed: list[ToolEntry] = []
    for index, entry in enumerate(value):
        where = f"local_tools[{index}]"
        if not isinstance(entry, dict):
            raise HHTcpError("bad_tools", f"{where} must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise HHTcpError("bad_tools", f"{where} needs a non-empty 'name'")
        description = entry.get("description") or ""
        if not isinstance(description, str):
            raise HHTcpError("bad_tools", f"{where} 'description' must be a string")
        params = entry.get("params") or []
        if not isinstance(params, list):
            raise HHTcpError("bad_tools", f"{where} 'params' must be a list")
        rollback = entry.get("rollback", False)
        if not isinstance(rollback, bool):
            raise HHTcpError("bad_tools", f"{where} 'rollback' must be a boolean")
        effects = entry.get("external_effects", False)
        if not isinstance(effects, bool):
            raise HHTcpError(
                "bad_tools", f"{where} 'external_effects' must be a boolean"
            )
        declared: list[ToolParam] = []
        for param in params:
            if not isinstance(param, dict):
                raise HHTcpError(
                    "bad_tools", f"{where} has a parameter that is not an object"
                )
            param_name = param.get("name")
            if not isinstance(param_name, str) or not param_name:
                raise HHTcpError("bad_tools", f"{where} has a parameter with no 'name'")
            declared.append(
                ToolParam(
                    name=param_name,
                    type=param.get("type") or "string",
                    description=param.get("description") or "",
                )
            )
        parsed.append(
            ToolEntry(
                name=name,
                description=description,
                params=declared,
                hook=None,
                remote_rollback=rollback,
                external_effects=effects,
            )
        )
    return parsed


def select_tools(names: Any) -> list[Any]:
    """Resolve requested tool names against the builtin catalogue."""
    catalogue = {tool.name: tool for tool in builtin_tools}
    if names is None:
        return list(catalogue.values())
    if not isinstance(names, list):
        raise HHTcpError("bad_tools", "'tools' must be a list of tool names")
    selected = []
    for name in names:
        tool = catalogue.get(name)
        if tool is None:
            raise HHTcpError(
                "unknown_tool",
                f"unknown tool {name!r}",
                detail={"available": sorted(catalogue)},
            )
        selected.append(tool)
    return selected


def tool_catalogue() -> list[dict[str, Any]]:
    """Describe the builtin tools for the `session_hello` event."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "state_namespace": tool.namespace,
            "external_effects": tool.external_effects,
            "rollback": tool.has_rollback,
            "params": [
                {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                }
                for param in tool.params
            ],
        }
        for tool in builtin_tools
    ]


def local_credentials() -> dict[str, str]:
    """Best-effort fallback to the sibling `test.py`, which holds the key."""
    path = pathlib.Path(__file__).resolve().parent.parent / "test.py"
    if not path.is_file():
        return {}
    spec = importlib.util.spec_from_file_location("_hh_local_credentials", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return {}
    found = {}
    for field in ("endpoint", "key"):
        value = getattr(module, field, None)
        if isinstance(value, str):
            found[field] = value
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="server.py",
        description="Expose HHAgent over a local TCP port, one JSON object per line.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--endpoint", help="default provider URL for new blocks")
    parser.add_argument("--key", help="default provider key for new blocks")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--db",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "harness.db"),
        help="SQLite file to persist blocks in",
    )
    parser.add_argument(
        "--max-db-bytes",
        type=int,
        default=DEFAULT_MAX_DB_BYTES,
        help="evict whole subtrees, oldest first, once the file exceeds this "
        "(0 disables the limit)",
    )
    args = parser.parse_args(argv)

    endpoint, key, source = args.endpoint, args.key, "flags"
    if not endpoint or not key:
        fallback = local_credentials()
        endpoint = endpoint or fallback.get("endpoint")
        key = key or fallback.get("key")
        source = "test.py"
    if not endpoint or not key:
        print(
            "warning: no credentials, blocks must be created with endpoint/key",
            flush=True,
        )
    else:
        print(f"credentials: {source}", flush=True)

    try:
        store = HHStore(args.db, max_bytes=args.max_db_bytes)
    except HHStoreError as exc:
        print(f"error: {exc}", flush=True)
        raise SystemExit(2) from exc
    server = HHServer(
        (args.host, args.port),
        defaults={"endpoint": endpoint, "key": key, "model": args.model},
        store=store,
    )
    print(
        f"listening on {args.host}:{args.port} "
        f"(model default {args.model}, protocol {PROTOCOL_VERSION})",
        flush=True,
    )
    print(
        f"store: {args.db} ({store.size_bytes()} bytes, "
        f"limit {args.max_db_bytes or 'none'})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
