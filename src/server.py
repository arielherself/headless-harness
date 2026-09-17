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
                    include_usage?, verbose?}          makes a root block
    fork           {id, prompt, new_id?, model?, tools?, timeout?,
                    include_usage?, verbose?}          appends a block
    run            {id}                                executes a block's turn
    cancel         {id}                                stop a running turn
    get_context    {id}                                flattened message list
    get_state      {id, tool?}                         rebuilt tool state
    set_state      {id, tool, key, value?/delete?}     seed or drop a state key
    list_agents    {}                                  every block, with links
    destroy_agent  {id}                                drop a block and its subtree
    ping           {echo?}

Events
    session_hello, command_received, command_finished, error, session_closing
    agent_created, agent_forked, agent_destroyed, agents_listed, context,
    state, state_seeded, cancel_result, pong, evicted, persist_warning
    and everything `HHAgent` emits while a block runs: turn_started,
    turn_finished, turn_failed, turn_cancelled, request_started,
    request_payload, response_received, request_finished, request_failed,
    content_delta, reasoning_delta, sse_chunk (verbose), sse_unparsed, usage,
    assistant_message, history_appended, tool_call_requested,
    tool_call_started, tool_call_finished, state_loaded, state_delta,
    state_discarded

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
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    HHAgent,
    HHAgentCancelled,
    HHAgentError,
    StateDelta,
)
from store import HHStore, HHStoreError
from tools import builtin_tools

PROTOCOL_VERSION = 2
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_COMMAND_BYTES = 8 * 1024 * 1024
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
            "timeout": "optional read timeout in seconds",
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
            "new_id": "optional id for the new block",
            "model": "optional override inherited from the parent",
            "tools": "optional override inherited from the parent",
            "timeout": "optional override inherited from the parent",
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
    {"command": "ping", "summary": "Liveness probe.", "fields": {"echo": "optional"}},
]

# Settings a fork may override on the block it creates.
INHERITED = ("model", "timeout", "include_usage", "verbose")
BOOL_FIELDS = ("include_usage", "verbose")


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
        block = HHAgent.root(
            endpoint,
            key,
            model=command.get("model") or self.defaults.get("model") or DEFAULT_MODEL,
            tools=select_tools(command.get("tools")),
            timeout=float(command.get("timeout") or DEFAULT_TIMEOUT),
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
            include_usage=block.include_usage,
            verbose=block.verbose,
            tools=sorted(block.tools),
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
        try:
            child = parent.fork(prompt, id=new_id)
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
            path=child.path(),
            context_len=len(child.context()),
            model=child.model,
            timeout=child.timeout,
            include_usage=child.include_usage,
            verbose=child.verbose,
            tools=sorted(child.tools),
            state_tools=child.state_tools(),
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
        names = [tool] if tool else block.state_tools()
        conn.send(
            "state",
            rid=rid,
            agent_id=agent_id,
            depth=block.depth,
            path=block.path(),
            tools=names,
            state={name: block.merged_state(name) for name in names},
        )

    def cmd_set_state(
        self, conn: Connection, command: dict[str, Any], rid: Any
    ) -> None:
        """Seed a block's own deltas. Only a finished block may be written to.

        A running block would race the turn, and a dirty one would need the seed
        folded into the turn's own diff at commit time. Seeding a finished block
        — a root, typically — is enough to give a whole subtree initial state.
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
        key = command.get("key")
        if not isinstance(key, str) or not key:
            raise HHTcpError("bad_field", "'key' must be a non-empty string")
        drop = bool(command.get("delete", False))
        if not drop and "value" not in command:
            raise HHTcpError("bad_field", "set_state needs 'value', or 'delete': true")
        replaced = key in block.merged_state(tool)
        delta = block.state_deltas.get(tool)
        if delta is None:
            delta = StateDelta()
            block.state_deltas[tool] = delta
        if drop:
            delta.changed.pop(key, None)
            if key not in delta.removed:
                delta.removed = (*delta.removed, key)
        else:
            delta.changed[key] = command["value"]
            delta.removed = tuple(name for name in delta.removed if name != key)
        if not delta.changed and not delta.removed:
            del block.state_deltas[tool]
        conn.send(
            "state_seeded",
            rid=rid,
            agent_id=agent_id,
            tool=tool,
            key=key,
            value=command.get("value"),
            deleted=drop,
            replaced=replaced,
            keys=sorted(block.merged_state(tool)),
        )
        self.persist(conn, block)

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
            "error": block.error,
            "prompt_chars": len(block.prompt),
            "prompt_preview": block.prompt[:200],
            "text_chars": len(block.text),
            "local_len": len(block.messages),
            "context_len": len(block.context()),
            "model": block.model,
            "tools": sorted(block.tools),
            "state_tools": block.state_tools(),
            "include_usage": block.include_usage,
            "verbose": block.verbose,
            "created_at": block.created_at,
            "age_ms": round((time.time() - block.created_at) * 1000, 3),
        }


def require_id(command: dict[str, Any]) -> str:
    agent_id = command.get("id")
    if not isinstance(agent_id, str) or not agent_id:
        raise HHTcpError("bad_id", "'id' must be a non-empty string")
    return agent_id


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
        elif name == "timeout":
            try:
                block.timeout = float(value)
            except (TypeError, ValueError) as exc:
                raise HHTcpError(
                    "bad_field", f"'timeout' must be a number: {exc}"
                ) from exc
        else:
            if not isinstance(value, str) or not value:
                raise HHTcpError(
                    "bad_field", f"{name!r} must be a non-empty string"
                )
            setattr(block, name, value)
    if command.get("tools") is not None:
        block.tools = {tool.name: tool for tool in select_tools(command["tools"])}


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
