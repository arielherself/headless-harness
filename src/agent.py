import copy
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Self

import requests

from tools import ToolContext, ToolEntry, builtin_tools

# Must be a model the provider serves over the OpenAI chat/completions shape;
# the `claude-*` models are rejected here and only accept /v1/messages.
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_TIMEOUT = 120.0
MAX_TOOL_ROUNDS = 120

# A hook receiving one event dict per thing that happens during a turn.
EventHook = Callable[[dict[str, Any]], None]


class HHAgentError(RuntimeError):
    """Raised when no completion can be produced: transport, HTTP, or bad payload."""


class HHAgentCancelled(HHAgentError):
    """Raised when a running turn is stopped by `HHAgent.cancel`."""


def new_id() -> str:
    """A short, collision-resistant block id."""
    return f"agent-{uuid.uuid4().hex[:12]}"


def _parse_arguments(name: str, raw_arguments: Any) -> tuple[Any, str | None]:
    """Decode a tool call's arguments, or explain why they are unusable."""
    if isinstance(raw_arguments, str):
        try:
            return (json.loads(raw_arguments) if raw_arguments.strip() else {}), None
        except json.JSONDecodeError as exc:
            return None, f"Error: arguments for '{name}' are not valid JSON: {exc}"
    if isinstance(raw_arguments, dict):
        return raw_arguments, None
    return {}, None


def _run_hook(tool: ToolEntry, context: ToolContext) -> tuple[str, str | None]:
    """Run a tool hook: returns the text for the model and why it failed."""
    try:
        return tool.hook(context, **context.arguments), None
    except TypeError as exc:
        return f"Error: bad arguments for '{tool.name}': {exc}", "bad_arguments"
    except Exception as exc:  # a failing tool must not abort the conversation
        return (
            f"Error: tool '{tool.name}' raised {type(exc).__name__}: {exc}",
            "tool_raised",
        )


def _same_value(before: Any, after: Any) -> bool:
    """Equality that never raises and never returns a non-bool.

    Exotic values (a numpy array, say) compare to something without a truth
    value; calling those changed only ever makes a delta a superset.
    """
    try:
        return bool(before == after)
    except Exception:
        return False


@dataclass
class _Turn:
    """One assistant turn, reassembled from a stream of SSE deltas."""

    content: str = ""
    calls: dict[int, dict[str, Any]] = field(default_factory=dict)

    def absorb(self, delta: dict[str, Any]) -> str:
        """Fold a delta in and return the text it carried, if any."""
        for fragment in delta.get("tool_calls") or []:
            call = self.calls.setdefault(
                fragment.get("index") or 0,
                {"id": "", "name": "", "arguments": ""},
            )
            if fragment.get("id"):
                # the id is sent once and never split, unlike the fields below
                call["id"] = fragment["id"]
            function = fragment.get("function") or {}
            call["name"] += function.get("name") or ""
            call["arguments"] += function.get("arguments") or ""
        text = delta.get("content") or ""
        self.content += text
        return text

    def message(self) -> dict[str, Any]:
        """Render the turn as an assistant message safe to send back."""
        if not self.calls:
            return {"role": "assistant", "content": self.content}
        message: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"] or "{}",
                    },
                }
                for _, call in sorted(self.calls.items())
            ],
        }
        if self.content:
            message["content"] = self.content
        return message


@dataclass
class StateDelta:
    """One block's contribution to a tool's persistent state.

    Granularity is the top-level key: whatever a tool did to `state["packages"]`
    is recorded as the new value of `packages`, which keeps nested edits (an
    in-place `append`, say) correct without having to track every write.
    """

    changed: dict[str, Any] = field(default_factory=dict)
    removed: tuple[str, ...] = ()


class HHAgent:
    """One block of a conversation chain.

    A block owns exactly one user prompt plus everything the model produced in
    reply to it. Blocks are linked through `parent`, so the conversation is a
    singly linked list and the context sent to the provider is `context()`,
    rebuilt by walking to the root. Forking stores a prompt and a pointer and
    copies nothing else.

    Tool state has the same shape: a block keeps only the deltas its own turn
    committed, under `state_deltas`, and a tool's live state is `tool_state()`,
    the chain's deltas replayed in order. Forking a block therefore rewinds
    every tool's memory to that point, and because a chain is linear there is
    never anything to merge.

    A block runs its single turn exactly once: `fork` marks it `dirty`, and the
    turn clears that flag when it ends, however it ends. Forking from a dirty
    block is refused, since its context is still growing. Only a turn that
    finished commits state deltas, so a tool's memory never advances through a
    failure, a cancellation, or an abandoned generator.
    """

    id: str
    parent: Self | None
    prompt: str
    messages: list[dict[str, Any]]
    text: str
    error: str | None
    dirty: bool
    created_at: float
    endpoint: str
    key: str
    model: str
    tools: dict[str, ToolEntry]
    timeout: float
    include_usage: bool
    verbose: bool
    on_event: EventHook | None
    state_deltas: dict[str, StateDelta]
    _cancel: threading.Event
    _run_lock: threading.Lock
    _live_states: dict[str, dict[str, Any]]
    _state_bases: dict[str, dict[str, Any]]

    @classmethod
    def root(
        cls,
        endpoint: str,
        key: str,
        model: str = DEFAULT_MODEL,
        tools: Iterable[ToolEntry] = builtin_tools,
        timeout: float = DEFAULT_TIMEOUT,
        include_usage: bool = True,
        verbose: bool = False,
        on_event: EventHook | None = None,
        id: str | None = None,
    ) -> Self:
        """Start a chain: a block with no prompt, waiting to be forked.

        `endpoint` is the base URL, e.g. `https://api.commandcode.ai/provider`;
        `/v1/chat/completions` is appended to it.

        `on_event`, when given, receives a dict for every step of a turn: the
        request that went out, each streamed text and reasoning delta, every
        tool call with its arguments and result, timings and token usage. The
        per-call `on_event` of `run`/`stream` takes precedence. `verbose` adds
        an event carrying each raw SSE chunk, and `include_usage` asks the
        provider for token accounting on the final chunk.
        """
        self = cls._blank()
        self.id = id or new_id()
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.timeout = timeout
        self.include_usage = include_usage
        self.verbose = verbose
        self.on_event = on_event
        return self

    @classmethod
    def _blank(cls) -> Self:
        """A block with every bookkeeping field at rest."""
        self = cls()
        self.id = new_id()
        self.parent = None
        self.prompt = ""
        self.messages = []
        self.text = ""
        self.error = None
        self.dirty = False
        self.created_at = time.time()
        self.on_event = None
        self.state_deltas = {}
        self._cancel = threading.Event()
        self._run_lock = threading.Lock()
        self._live_states = {}
        self._state_bases = {}
        return self

    def fork(
        self,
        message: str,
        id: str | None = None,
        on_event: EventHook | None = None,
    ) -> Self:
        """Start the next block, holding `message` as its user prompt.

        The child inherits this block's provider, model, tools and options, and
        starts out `dirty` because its turn has not run yet. Override any of
        the inherited settings on the returned block before running it.
        """
        if self.dirty:
            raise HHAgentError(
                f"agent {self.id} has not finished, fork from a finished block"
            )
        if not isinstance(message, str) or not message:
            raise HHAgentError("fork needs a non-empty prompt")
        child = type(self)._blank()
        child.id = id or new_id()
        child.parent = self
        child.prompt = message
        child.messages = [{"role": "user", "content": message}]
        child.dirty = True
        child.endpoint = self.endpoint
        child.key = self.key
        child.model = self.model
        child.tools = dict(self.tools)
        child.timeout = self.timeout
        child.include_usage = self.include_usage
        child.verbose = self.verbose
        child.on_event = self.on_event if on_event is None else on_event
        return child

    @property
    def running(self) -> bool:
        """Whether this block's turn is executing right now."""
        return self._run_lock.locked()

    @property
    def depth(self) -> int:
        """How many blocks precede this one in the chain."""
        return len(self.lineage()) - 1

    def lineage(self) -> list[Self]:
        """This block and its ancestors, root first."""
        chain: list[Self] = []
        node: Self | None = self
        while node is not None:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def path(self) -> list[str]:
        """The ids of `lineage()`, root first."""
        return [node.id for node in self.lineage()]

    def context(self) -> list[dict[str, Any]]:
        """Every message of the chain, oldest first.

        Rebuilt on demand by walking to the root, so no block stores a copy of
        anything its ancestors already hold.
        """
        # TODO(context-window): the whole lineage goes out on every request, so
        # this grows without bound as a chain deepens. See TODO.md.
        return [message for node in self.lineage() for message in node.messages]

    def state_tools(self) -> list[str]:
        """Names of the tools this chain has committed state deltas for."""
        names: list[str] = []
        for node in self.lineage():
            for name in node.state_deltas:
                if name not in names:
                    names.append(name)
        return names

    def merged_state(self, tool: str) -> dict[str, Any]:
        """A tool's state as of this block.

        The values are the stored deltas' own objects, so treat the result as
        read-only; `tool_state()` returns a copy a tool may edit in place.
        """
        state: dict[str, Any] = {}
        for node in self.lineage():
            delta = node.state_deltas.get(tool)
            if delta is None:
                continue
            state.update(delta.changed)
            for key in delta.removed:
                state.pop(key, None)
        return state

    def tool_state(self, tool: str) -> dict[str, Any]:
        """A deep copy of `merged_state`, safe to hand to a tool to mutate."""
        return copy.deepcopy(self.merged_state(tool))

    def all_states(self) -> dict[str, dict[str, Any]]:
        """Every touched tool's state as of this block, for reporting."""
        return {name: self.merged_state(name) for name in self.state_tools()}

    def _live_state(self, tool: str) -> dict[str, Any]:
        """One tool's mutable state, loaded once per turn and kept live.

        The baseline is kept next to it so the turn's diff is measured against
        the state as it stood when the turn began, not against the last call.
        """
        if tool not in self._live_states:
            self._state_bases[tool] = self.merged_state(tool)
            self._live_states[tool] = copy.deepcopy(self._state_bases[tool])
        return self._live_states[tool]

    def _release_states(self, hook: EventHook | None, outcome: str) -> None:
        """Commit this turn's tool state if it succeeded, drop it otherwise.

        Called once, from the turn's `finally`, which is what makes a block's
        state atomic: a turn lands whole or not at all.
        """
        for tool, live in self._live_states.items():
            before = self._state_bases.get(tool, {})
            changed = {
                key: value
                for key, value in live.items()
                if key not in before or not _same_value(before[key], value)
            }
            removed = tuple(key for key in before if key not in live)
            if not changed and not removed:
                continue
            if outcome != "commit":
                self._emit(
                    hook,
                    "state_discarded",
                    tool=tool,
                    changed=sorted(changed),
                    removed=list(removed),
                    reason=outcome,
                )
                continue
            self.state_deltas[tool] = StateDelta(changed=changed, removed=removed)
            self._emit(
                hook,
                "state_delta",
                tool=tool,
                changed=sorted(changed),
                removed=list(removed),
                keys=sorted(self.merged_state(tool)),
            )
        self._live_states.clear()
        self._state_bases.clear()

    def cancel(self) -> bool:
        """Ask the running turn to stop, returning whether one was running.

        The turn notices on the next streamed chunk, so it stops promptly but
        not instantly. Cancelling while idle is a no-op rather than a landmine
        for the next turn.
        """
        if not self.running:
            return False
        self._cancel.set()
        return True

    def chat(self, on_event: EventHook | None = None) -> str:
        """Run this block's turn and return the reply once it is complete."""
        return "".join(self.stream(on_event))

    def stream(self, on_event: EventHook | None = None) -> Iterator[str]:
        """Run this block's turn, yielding the reply as it is generated.

        Tool calls are executed transparently in between the text, so joining
        the yielded chunks gives exactly what `chat` returns. Reasoning deltas
        (`reasoning`, `reasoning_content`) are not part of the reply: they are
        reported through the event hook instead.

        The generator is lazy: the first iteration flips `running` on, and the
        block stops being `dirty` when the turn ends for any reason.
        """
        hook = on_event if on_event is not None else self.on_event
        if not self._run_lock.acquire(blocking=False):
            raise HHAgentError(f"agent {self.id} is already running")
        started = time.monotonic()
        parts: list[str] = []
        tool_calls = 0
        outcome = "abandoned"
        # cleared here, on the turn thread, so an idle cancel cannot leak in
        self._cancel.clear()
        try:
            if self.parent is None:
                raise HHAgentError(f"agent {self.id} is a root; fork from it first")
            if not self.dirty:
                raise HHAgentError(f"agent {self.id} has already finished its turn")
            self._emit(
                hook,
                "turn_started",
                prompt=self.prompt,
                prompt_chars=len(self.prompt),
                depth=self.depth,
                path=self.path(),
                model=self.model,
                tools=sorted(self.tools),
                context_len=len(self.context()),
                include_usage=self.include_usage,
                verbose=self.verbose,
            )
            # already in `messages` since fork; only announce it here
            self._announce(hook, self.messages[0], "fork")
            for round_no in range(1, MAX_TOOL_ROUNDS + 1):
                turn = _Turn()
                yield from self._stream_turn(turn, hook, round_no)
                parts.append(turn.content)
                reply = turn.message()
                self._remember(hook, reply, "assistant")
                calls = reply.get("tool_calls")
                self._emit(
                    hook,
                    "assistant_message",
                    round=round_no,
                    content=turn.content,
                    content_chars=len(turn.content),
                    tool_calls=[
                        {
                            "call_id": call["id"],
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"],
                        }
                        for call in calls or []
                    ],
                )
                if not calls:
                    outcome = "commit"
                    self._finish(hook, started, parts, round_no, tool_calls)
                    return
                tool_calls += len(calls)
                for tool_message in self._run_tools(calls, hook, round_no):
                    self._remember(hook, tool_message, "tool")
            raise HHAgentError(
                f"model requested tools more than {MAX_TOOL_ROUNDS} times in a row"
            )
        except HHAgentCancelled as exc:
            self.error = str(exc)
            outcome = "cancelled"
            self._emit(
                hook,
                "turn_cancelled",
                error=self.error,
                text="".join(parts),
                elapsed_ms=self._ms(started),
                context_len=len(self.context()),
                dirty=False,
            )
            raise
        except Exception as exc:
            self.error = str(exc)
            outcome = "failed"
            self._emit(
                hook,
                "turn_failed",
                error=self.error,
                error_type=type(exc).__name__,
                text="".join(parts),
                elapsed_ms=self._ms(started),
                context_len=len(self.context()),
                dirty=False,
            )
            raise
        finally:
            self._release_states(hook, outcome)
            self.dirty = False
            self._run_lock.release()

    def _finish(
        self,
        hook: EventHook | None,
        started: float,
        parts: list[str],
        rounds: int,
        tool_calls: int,
    ) -> None:
        """Record the finished turn and announce it."""
        self.text = "".join(parts)
        self._emit(
            hook,
            "turn_finished",
            text=self.text,
            text_chars=len(self.text),
            rounds=rounds,
            tool_calls=tool_calls,
            elapsed_ms=self._ms(started),
            messages=len(self.messages),
            context_len=len(self.context()),
            dirty=False,
        )

    def _stream_turn(
        self, turn: _Turn, hook: EventHook | None, round_no: int
    ) -> Iterator[str]:
        """Run one streamed /v1/chat/completions request into `turn`."""
        started = time.monotonic()
        schemas = self.tool_schemas()
        messages = self.context()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if schemas:
            # an empty list is not the same request as an absent one
            payload["tools"] = schemas
        if self.include_usage:
            payload["stream_options"] = {"include_usage": True}
        url = f"{self.endpoint}/v1/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._emit(
            hook,
            "request_started",
            round=round_no,
            url=url,
            model=self.model,
            timeout=self.timeout,
            request_bytes=len(body),
            messages=len(messages),
            tools=len(schemas),
            depth=self.depth,
            include_usage=self.include_usage,
        )
        self._emit(
            hook,
            "request_payload",
            round=round_no,
            messages=self._message_summary(messages),
        )
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                },
                data=body,
                stream=True,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            self._emit(
                hook,
                "request_failed",
                round=round_no,
                error=str(exc),
                error_type=type(exc).__name__,
                elapsed_ms=self._ms(started),
            )
            raise HHAgentError(f"request to {self.endpoint} failed: {exc}") from exc
        with response:
            # the provider declares ISO-8859-1, which would mangle UTF-8 text
            response.encoding = "utf-8"
            self._emit(
                hook,
                "response_received",
                round=round_no,
                status=response.status_code,
                reason=response.reason,
                content_type=response.headers.get("Content-Type"),
                elapsed_ms=self._ms(started),
            )
            if not response.ok:
                raise HHAgentError(
                    f"{response.status_code} {response.reason}: {response.text[:500]}"
                )
            chunks = 0
            chars = 0
            first_chunk_ms: float | None = None
            finish_reason: str | None = None
            try:
                for line in self._sse_payloads(response):
                    if self._cancel.is_set():
                        raise HHAgentCancelled(f"agent {self.id} cancelled by caller")
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        self._emit(hook, "sse_unparsed", round=round_no, line=line[:500])
                        continue
                    chunks += 1
                    chars += len(line)
                    if first_chunk_ms is None:
                        first_chunk_ms = self._ms(started)
                    if self.verbose:
                        self._emit(
                            hook,
                            "sse_chunk",
                            round=round_no,
                            index=chunks,
                            chunk=chunk,
                        )
                    usage = chunk.get("usage")
                    if usage:
                        self._emit(hook, "usage", round=round_no, usage=usage)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue  # e.g. a trailing usage-only chunk
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning") or delta.get(
                        "reasoning_content"
                    )
                    if reasoning:
                        self._emit(
                            hook,
                            "reasoning_delta",
                            round=round_no,
                            text=reasoning,
                            chars=len(reasoning),
                        )
                    text = turn.absorb(delta)
                    if text:
                        self._emit(
                            hook,
                            "content_delta",
                            round=round_no,
                            index=chunks,
                            text=text,
                            chars=len(text),
                            elapsed_ms=self._ms(started),
                        )
                        yield text
            except requests.RequestException as exc:
                self._emit(
                    hook,
                    "request_failed",
                    round=round_no,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    chunks=chunks,
                    elapsed_ms=self._ms(started),
                )
                raise HHAgentError(
                    f"stream from {self.endpoint} failed: {exc}"
                ) from exc
            self._emit(
                hook,
                "request_finished",
                round=round_no,
                status=response.status_code,
                chunks=chunks,
                payload_chars=chars,
                finish_reason=finish_reason,
                first_chunk_ms=first_chunk_ms,
                elapsed_ms=self._ms(started),
                cancelled=self._cancel.is_set(),
            )

    @staticmethod
    def _sse_payloads(response: requests.Response) -> Iterator[str]:
        """Unwrap an OpenAI-style text/event-stream into its data payloads."""
        for line in response.iter_lines():
            if not line or line.startswith(b":"):
                continue
            if line.startswith(b"data:"):
                # decoded by hand: the charset header is not trustworthy here
                yield line[5:].decode("utf-8", errors="replace").strip()

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Describe the registered tools in the OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param.name: {
                                "type": param.type,
                                "description": param.description,
                            }
                            for param in tool.params
                        },
                        "required": [param.name for param in tool.params],
                    },
                },
            }
            for tool in self.tools.values()
        ]

    @staticmethod
    def _message_summary(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Describe the outgoing context without resending every byte."""
        summary = []
        for message in messages:
            entry: dict[str, Any] = {
                "role": message.get("role"),
                "chars": len(message.get("content") or ""),
            }
            if message.get("tool_call_id"):
                entry["tool_call_id"] = message["tool_call_id"]
            if message.get("tool_calls"):
                entry["tool_calls"] = [
                    call.get("function", {}).get("name")
                    for call in message["tool_calls"]
                ]
            summary.append(entry)
        return summary

    def _run_tools(
        self, calls: list[dict[str, Any]], hook: EventHook | None, round_no: int
    ) -> list[dict[str, Any]]:
        """Execute requested tool calls and build the matching tool messages."""
        results: list[dict[str, Any]] = []
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            raw = function.get("arguments")
            call_id = call.get("id", "")
            self._emit(
                hook,
                "tool_call_requested",
                round=round_no,
                call_id=call_id,
                name=name,
                raw_arguments=raw,
                known=name in self.tools,
            )
            self._emit(
                hook,
                "tool_call_started",
                round=round_no,
                call_id=call_id,
                name=name,
            )
            started = time.monotonic()
            tool = self.tools.get(name)
            if tool is None:
                text, error = f"Error: unknown tool '{name}'", "unknown_tool"
            else:
                arguments, parse_error = _parse_arguments(name, raw)
                if parse_error is not None:
                    text, error = parse_error, "bad_arguments"
                elif not isinstance(arguments, dict):
                    text, error = (
                        f"Error: arguments for '{name}' must be a JSON object",
                        "bad_arguments",
                    )
                else:
                    context = ToolContext(
                        tool=tool,
                        agent=self,
                        call_id=call_id,
                        arguments=arguments,
                        raw_arguments=raw,
                        state=self._live_state(name),
                    )
                    self._emit(
                        hook,
                        "state_loaded",
                        round=round_no,
                        call_id=call_id,
                        tool=name,
                        keys=sorted(context.state),
                        inherited=len(self._state_bases.get(name) or {}),
                    )
                    text, error = _run_hook(tool, context)
            self._emit(
                hook,
                "tool_call_finished",
                round=round_no,
                call_id=call_id,
                name=name,
                ok=error is None,
                error=error,
                result=text,
                result_chars=len(text),
                elapsed_ms=self._ms(started),
            )
            results.append({"role": "tool", "tool_call_id": call_id, "content": text})
        return results

    def _remember(
        self, hook: EventHook | None, message: dict[str, Any], source: str
    ) -> None:
        """Record a message in this block and report what went in."""
        self.messages.append(message)
        self._announce(hook, message, source)

    def _announce(
        self, hook: EventHook | None, message: dict[str, Any], source: str
    ) -> None:
        """Announce a message that this block now holds."""
        content = message.get("content") or ""
        self._emit(
            hook,
            "history_appended",
            source=source,
            role=message.get("role"),
            chars=len(content),
            preview=content[:200],
            tool_call_id=message.get("tool_call_id"),
            tool_calls=len(message.get("tool_calls") or []),
            messages=len(self.messages),
            context_len=len(self.context()),
        )

    @staticmethod
    def _ms(started: float) -> float:
        return round((time.monotonic() - started) * 1000, 3)

    def _emit(self, hook: EventHook | None, event: str, **fields: Any) -> None:
        """Hand one event to the hook. A broken listener must not kill a turn."""
        if hook is None:
            return
        try:
            hook({"event": event, "agent_id": self.id, **fields})
        except Exception:
            pass
