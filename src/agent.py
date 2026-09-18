import copy
import hashlib
import json
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import requests

from tools import ToolCall, ToolContext, ToolEntry, ToolResult, builtin_tools

# Must be a model the provider serves over the OpenAI chat/completions shape;
# the `claude-*` models are rejected here and only accept /v1/messages.
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_TIMEOUT = 120.0
# how long a turn waits for a client to answer a local tool call
DEFAULT_LOCAL_TIMEOUT = 120.0
MAX_TOOL_ROUNDS = 120
# How many calls one tool's pipe may add before it is cut off. A pipe is a
# tool's own business and runs without the model in the loop, so the bound is
# what keeps a tool that pipes to itself from spinning forever.
MAX_PIPE_DEPTH = 16
# A pipe's intermediate values are recorded for inspection, never sent to the
# provider, so they are bounded rather than stored whole: a base64 file would
# otherwise sit in the database once per step that handled it.
PIPE_VALUE_MAX_CHARS = 4096
# how deep into a nested argument the bounded rendering goes before giving up
PIPE_VALUE_MAX_DEPTH = 6

# The default model takes max_tokens in [1, 393216], so the default cap is its own
# maximum output. A block may override it; `0` sends no cap at all and leaves the
# choice to the provider.
DEFAULT_MAX_TOKENS = 393216

# A hook receiving one event dict per thing that happens during a turn.
EventHook = Callable[[dict[str, Any]], None]

# Error codes meaning a call never reached the tool body, so nothing happened and
# there is nothing to undo. A rollback for one of these would be inventing work.
NOT_RUN = ("bad_arguments", "no_hook")

# Asked of the model when a turn fails, to leave a note for whoever reads the
# chain next. One request, no tools: it only writes the note.
SUMMARY_PROMPT = (
    "You are the harness that runs a conversation agent, and one turn of that "
    "conversation has just failed. Write a short note, for the model that will "
    "read this conversation next, describing what the turn tried to do and where "
    "it stood when it stopped. Summarise simply, in the language the conversation "
    "is in. Do not repeat the transcript or the error verbatim — the reader has "
    "both already — and do not ask questions, offer help, or go beyond what the "
    "transcript shows."
)


class HHAgentError(RuntimeError):
    """Raised when no completion can be produced: transport, HTTP, or bad payload."""


class HHAgentCancelled(HHAgentError):
    """Raised when a running turn is stopped by `HHAgent.cancel`."""


def new_id() -> str:
    """A short, collision-resistant block id."""
    return f"agent-{uuid.uuid4().hex[:12]}"


def _check_image_url(url: str, where: str) -> None:
    """Refuse anything that is not an http(s) URL or a data:image/... URI.

    A local path — or any other scheme, `file:` included — would hand the
    provider a local file to open. The harness never reads one on a client's
    behalf, and a path must not reach a provider that would.
    """
    lowered = url.lower()
    if lowered.startswith("data:image/"):
        return
    if lowered.startswith("data:"):
        raise HHAgentError(
            f"{where} must be a data:image/... URI or an http(s) URL; "
            "that data: URI is not an image"
        )
    scheme, sep, rest = url.partition("://")
    if sep and rest and scheme.lower() in ("http", "https"):
        return
    if not sep or scheme.lower() == "file":
        raise HHAgentError(
            f"{where} must be a data:image/... URI or an http(s) URL; "
            "local paths and file: URLs are never read"
        )
    raise HHAgentError(
        f"{where} must be a data:image/... URI or an http(s) URL; "
        f"the {scheme.lower()!r} scheme is not accepted"
    )


def image_content_parts(images: Any) -> list[dict[str, Any]]:
    """Normalize images into the content parts the provider expects.

    Each image is an `http(s)` URL or a `data:image/...` URI string, or a
    mapping with a non-empty `url` and an optional `detail` (`auto`, `low` or
    `high`, passed through). Local paths and other schemes (`file:` included)
    are refused: the harness never reads a file for a client, and it will not
    hand a provider a path to open. `None`, an empty sequence and a single
    entry are all accepted, as is an already-normalized `image_url` part.
    Anything else raises `HHAgentError`, so a bad image fails the fork rather
    than reaching the provider.
    """
    if images is None:
        return []
    if isinstance(images, (str, Mapping)):
        images = [images]
    if not isinstance(images, (list, tuple)):
        raise HHAgentError(
            "images must be a list of image URLs, data:image/... URIs "
            "or {url, detail} objects"
        )
    parts: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        where = f"image {index + 1}"
        if isinstance(image, Mapping) and image.get("type") == "image_url":
            part = dict(image)
            nested = part.get("image_url")
            url = nested.get("url") if isinstance(nested, Mapping) else None
            if not isinstance(url, str) or not url:
                raise HHAgentError(f"{where} needs a non-empty 'url'")
            _check_image_url(url, where)
            parts.append(part)
            continue
        if isinstance(image, str):
            url, detail = image, None
        elif isinstance(image, Mapping):
            url, detail = image.get("url"), image.get("detail")
        else:
            raise HHAgentError(
                f"{where} must be a URL string or a {{url, detail}} object"
            )
        if not isinstance(url, str) or not url:
            raise HHAgentError(f"{where} needs a non-empty 'url'")
        _check_image_url(url, where)
        if detail is not None and not isinstance(detail, str):
            raise HHAgentError(f"{where} has a non-string 'detail'")
        rendered: dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
        if detail:
            rendered["image_url"]["detail"] = detail
        parts.append(rendered)
    return parts


def message_text(content: Any) -> str:
    """The text of a message's `content`, however it is shaped.

    A message with images holds a list of content parts; anything that reports
    or summarises a message wants only its text, never a megabyte of base64.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(
            part.get("text") or ""
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return ""


def message_image_count(content: Any) -> int:
    """How many image parts a message's `content` carries."""
    if not isinstance(content, (list, tuple)):
        return 0
    return sum(
        1
        for part in content
        if isinstance(part, Mapping) and part.get("type") == "image_url"
    )


def content_with_images(text: str, parts: Iterable[dict[str, Any]]) -> Any:
    """A message's `content`: plain text, or text and images as parts.

    Without images the content stays the string it has always been; with them
    it becomes an OpenAI-style content list, and the text part is omitted when
    there is no text to send.
    """
    parts = list(parts)
    if not parts:
        return text
    return ([{"type": "text", "text": text}] if text else []) + parts


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


def _check_call(call: ToolCall) -> ToolCall:
    """Check a pipe call's shape, returning it with a plain dict of arguments.

    Shape only: whether the name is a registered tool is decided when the call
    runs, where an unknown one can be reported to the model like any other
    unknown tool rather than failing the hook that asked for it.
    """
    if not isinstance(call.name, str) or not call.name:
        raise HHAgentError("a tool call needs a non-empty 'name'")
    if not isinstance(call.arguments, Mapping):
        raise HHAgentError(f"tool call '{call.name}' needs its arguments as an object")
    try:
        arguments = dict(call.arguments)
    except Exception as exc:  # a mapping that will not copy
        raise HHAgentError(
            f"tool call '{call.name}' has unusable arguments: {exc}"
        ) from exc
    return ToolCall(name=call.name, arguments=arguments)


def _bounded(
    value: Any,
    limit: int = PIPE_VALUE_MAX_CHARS,
    depth: int = PIPE_VALUE_MAX_DEPTH,
) -> Any:
    """A JSON-safe, size-bounded rendering of a value, for traces and events.

    Only a pipe's intermediate values go through this: the model never sees
    them, and a step may legitimately carry something enormous (a file's bytes),
    so what is kept is a preview plus its true size and a digest, never the
    whole thing. Everything comes back as JSON primitives, which is what lets a
    trace be stored in a block and sent over the wire unchanged.
    """
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        # NaN and the infinities are not JSON, so they are rendered as text
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        digest = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()
        return (
            f"{value[:limit]}\u2026[+{len(value) - limit} chars, "
            f"sha256 {digest[:16]}]"
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return f"<{len(raw)} bytes, sha256 {hashlib.sha256(raw).hexdigest()[:16]}>"
    if depth <= 0:
        return (
            f"<{type(value).__name__} nested deeper than "
            f"{PIPE_VALUE_MAX_DEPTH} levels>"
        )
    if isinstance(value, Mapping):
        return {
            str(key): _bounded(item, limit, depth - 1) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_bounded(item, limit, depth - 1) for item in value]
    try:
        rendered = repr(value)[:limit]
    except Exception:  # a repr that raises is still not a reason to fail a turn
        rendered = "<unprintable>"
    return f"<{type(value).__name__}: {rendered}>"


def _run_hook(tool: ToolEntry, context: ToolContext) -> tuple["_ToolReply", str | None]:
    """Run a tool hook: returns its reply and why it failed."""
    hook = tool.hook
    if hook is None:
        return _reply(f"Error: tool '{tool.name}' has no hook"), "no_hook"
    try:
        returned = hook(context, **context.arguments)
    except TypeError as exc:
        return (
            _reply(f"Error: bad arguments for '{tool.name}': {exc}"),
            "bad_arguments",
        )
    except Exception as exc:  # a failing tool must not abort the conversation
        return (
            _reply(f"Error: tool '{tool.name}' raised {type(exc).__name__}: {exc}"),
            "tool_raised",
        )
    try:
        return _tool_reply(returned), None
    except HHAgentError as exc:
        return _reply(f"Error: tool '{tool.name}' {exc}"), "bad_result"


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
class _Pending:
    """A local tool call that is waiting for the client to answer it.

    Only a `threading.Event` is shared with the resolving thread: no registry
    lock and no database transaction is held while a client decides.
    """

    name: str
    event: threading.Event
    result: str | None = None
    error: str | None = None
    # already-normalized image parts from the client's answer
    images: list[dict[str, Any]] = field(default_factory=list)
    # a next call, when the client answered with one instead of ending the pipe
    call: ToolCall | None = None


@dataclass
class _Call:
    """A tool call this turn made, kept so its rollback can undo it."""

    tool: ToolEntry
    call_id: str
    arguments: dict[str, Any]
    raw_arguments: Any
    result: str
    ok: bool


@dataclass
class _ToolReply:
    """One tool call's answer, in the shapes the rest of the turn needs."""

    # what goes into the tool message: a string, or a content-part list
    content: Any = ""
    # the text alone, for events, failure notes and rollback reports
    text: str = ""
    image_count: int = 0
    # the normalized parts `content` was built from, so a pipe can rebuild the
    # message with its own header in front
    parts: tuple[dict[str, Any], ...] = ()
    # set when this reply is not the end of the pipe: the call to run next
    call: ToolCall | None = None


def _reply(text: str, parts: Iterable[dict[str, Any]] = ()) -> _ToolReply:
    """A reply built from text and already-normalized image parts."""
    parts = tuple(parts)
    return _ToolReply(
        content=content_with_images(text, parts),
        text=text,
        image_count=len(parts),
        parts=parts,
    )


def _tool_reply(returned: Any) -> _ToolReply:
    """Normalize whatever a hook returned: a string, `ToolResult` or `ToolCall`."""
    if isinstance(returned, ToolCall):
        returned = ToolResult(call=returned)
    if isinstance(returned, ToolResult):
        if not isinstance(returned.text, str):
            raise HHAgentError("returned a ToolResult whose text is not a string")
        if returned.call is None:
            return _reply(returned.text, image_content_parts(returned.images))
        if not isinstance(returned.call, ToolCall):
            raise HHAgentError("returned a ToolResult whose call is not a ToolCall")
        try:
            call = _check_call(returned.call)
        except HHAgentError as exc:
            raise HHAgentError(f"returned an unusable tool call: {exc}") from exc
        if returned.images:
            raise HHAgentError(
                "returned both a pipe call and images; only the last call of a "
                "pipe may return images"
            )
        return _ToolReply(text=returned.text, call=call)
    if isinstance(returned, str):
        return _reply(returned)
    raise HHAgentError(
        f"returned {type(returned).__name__}, expected a string, ToolResult or ToolCall"
    )


def _step_record(
    call_id: str,
    name: str,
    via: str,
    arguments: Any,
    reply: _ToolReply,
    error: str | None,
    elapsed_ms: float,
) -> dict[str, Any]:
    """One call inside a pipe, as the trace and the step events report it."""
    return {
        "call_id": call_id,
        "name": name,
        "via": via,
        "arguments": _bounded(arguments),
        "ok": error is None,
        "error": error,
        "text": _bounded(reply.text),
        "text_chars": len(reply.text),
        "image_count": reply.image_count,
        "next": reply.call.name if reply.call is not None else None,
        "elapsed_ms": elapsed_ms,
    }


@dataclass
class _Undo:
    """What became of one call's effects when the turn did not commit."""

    tool: str
    status: str  # "undone", "stuck" (the undo failed), or "standing"
    error: str | None = None


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
    committed, under `state_deltas` keyed by state namespace, and a tool's live
    state is `tool_state()`, the chain's deltas replayed in order. Forking a
    block therefore rewinds every tool's memory to that point, and because a
    chain is linear there is never anything to merge.

    A block runs its single turn exactly once: `fork` marks it `dirty`, and the
    turn clears that flag when it ends, however it ends. Forking from a dirty
    block is refused, since its context is still growing. Only a turn that
    finished commits state deltas, so a tool's memory never advances through a
    failure, a cancellation, or an abandoned generator.

    A tool may also answer with a call instead of text, and the harness runs it
    before the model is consulted again: a **pipe**. Only the tools the pipe
    called and the last call's output are shown to the model; every step's
    arguments and result stay in `pipe_traces`, in memory for the life of the
    process but never written to the store, and each step is recorded as a call
    of its own, so a failed turn can undo them all, newest first.
    """

    id: str
    parent: Self | None
    prompt: str
    messages: list[dict[str, Any]]
    text: str
    error: str | None
    # how the turn ended: "ok", "failed", "cancelled" or "abandoned"
    outcome: str | None
    dirty: bool
    created_at: float
    endpoint: str
    key: str
    model: str
    tools: dict[str, ToolEntry]
    timeout: float
    local_timeout: float
    # output-token cap sent with every turn request; 0 sends no cap
    max_tokens: int
    # model for failure summaries; empty means this block's own model
    summary_model: str
    include_usage: bool
    verbose: bool
    on_event: EventHook | None
    # keyed by namespace: a tool's `state_namespace`, or its name when unset
    state_deltas: dict[str, StateDelta]
    # one record per pipe this block's own turn ran, for inspection and for
    # `get_context`; the model never sees any of it (see `_run_call`), and the
    # store never keeps it
    pipe_traces: list[dict[str, Any]]
    _cancel: threading.Event
    _run_lock: threading.Lock
    _live_states: dict[str, dict[str, Any]]
    _state_bases: dict[str, dict[str, Any]]
    _pending: dict[str, _Pending]
    _called: list[_Call]

    @classmethod
    def root(
        cls,
        endpoint: str,
        key: str,
        model: str = DEFAULT_MODEL,
        tools: Iterable[ToolEntry] = builtin_tools,
        timeout: float = DEFAULT_TIMEOUT,
        local_timeout: float = DEFAULT_LOCAL_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        summary_model: str = "",
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
        provider for token accounting on the final chunk. `max_tokens` caps how
        many tokens one turn request may produce; it defaults to the default
        model's own maximum, and `0` leaves the cap to the provider. The
        failure-summary request is separate and uncapped.
        """
        self = cls._blank()
        self.id = id or new_id()
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.timeout = timeout
        self.local_timeout = local_timeout
        self.max_tokens = max_tokens
        self.summary_model = summary_model
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
        self.outcome = None
        self.dirty = False
        self.created_at = time.time()
        self.on_event = None
        self.state_deltas = {}
        self.pipe_traces = []
        self._cancel = threading.Event()
        self._run_lock = threading.Lock()
        self._live_states = {}
        self._state_bases = {}
        self.local_timeout = DEFAULT_LOCAL_TIMEOUT
        self._pending = {}
        self._called = []
        return self

    def fork(
        self,
        message: str,
        id: str | None = None,
        on_event: EventHook | None = None,
        images: Any = None,
    ) -> Self:
        """Start the next block, holding `message` as its user prompt.

        `images`, when given, is a list of `http(s)` URLs or `data:image/...`
        URIs — strings, or `{url, detail}` mappings; a local path is refused.
        The child's user message then becomes a content-part list: the text
        first, then one `image_url` part per image. `prompt` stays the text
        either way, and without images the message remains the plain string it
        has always been. Images are validated here, so a bad one fails the fork
        before a block exists.

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
        parts = image_content_parts(images)
        child = type(self)._blank()
        child.id = id or new_id()
        child.parent = self
        child.prompt = message
        child.messages = [
            {"role": "user", "content": content_with_images(message, parts)}
        ]
        child.dirty = True
        child.endpoint = self.endpoint
        child.key = self.key
        child.model = self.model
        child.tools = dict(self.tools)
        child.timeout = self.timeout
        child.local_timeout = self.local_timeout
        child.max_tokens = self.max_tokens
        child.summary_model = self.summary_model
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

    @property
    def image_count(self) -> int:
        """How many image parts this block's own messages carry."""
        return sum(
            message_image_count(message.get("content")) for message in self.messages
        )

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

    def state_namespaces(self) -> list[str]:
        """Namespace of every state delta this chain has committed."""
        names: list[str] = []
        for node in self.lineage():
            for name in node.state_deltas:
                if name not in names:
                    names.append(name)
        return names

    def merged_state(self, namespace: str) -> dict[str, Any]:
        """One namespace's state as of this block.

        The values are the stored deltas' own objects, so treat the result as
        read-only; `tool_state()` returns a copy a tool may edit in place.
        """
        state: dict[str, Any] = {}
        for node in self.lineage():
            delta = node.state_deltas.get(namespace)
            if delta is None:
                continue
            state.update(delta.changed)
            for name in delta.removed:
                state.pop(name, None)
        return state

    def tool_state(self, namespace: str) -> dict[str, Any]:
        """A deep copy of `merged_state`, safe to hand to a tool to mutate."""
        return copy.deepcopy(self.merged_state(namespace))

    def all_states(self) -> dict[str, dict[str, Any]]:
        """Every touched namespace's state as of this block, for reporting."""
        return {name: self.merged_state(name) for name in self.state_namespaces()}

    def tool_for(self, namespace: str) -> str:
        """The name of a tool that shares this namespace, or the namespace.

        Used for reporting: a delta is stored under its namespace, and the event
        that announces it should also say which tool caused it.
        """
        for tool in self.tools.values():
            if tool.namespace == namespace:
                return tool.name
        return namespace

    def _live_state(self, namespace: str) -> dict[str, Any]:
        """One namespace's mutable state, loaded once per turn and kept live.

        The baseline is kept next to it so the turn's diff is measured against
        the state as it stood when the turn began, not against the last call.
        """
        if namespace not in self._live_states:
            self._state_bases[namespace] = self.merged_state(namespace)
            self._live_states[namespace] = copy.deepcopy(self._state_bases[namespace])
        return self._live_states[namespace]

    def _release_states(self, hook: EventHook | None, outcome: str) -> None:
        """Commit this turn's tool state if it succeeded, drop it otherwise.

        Called once, from the turn's `finally`, which is what makes a block's
        state atomic: a turn lands whole or not at all.
        """
        for namespace, live in self._live_states.items():
            before = self._state_bases.get(namespace, {})
            changed = {
                name: value
                for name, value in live.items()
                if name not in before or not _same_value(before[name], value)
            }
            removed = tuple(name for name in before if name not in live)
            if not changed and not removed:
                continue
            if outcome != "commit":
                self._emit(
                    hook,
                    "state_discarded",
                    tool=self.tool_for(namespace),
                    state_namespace=namespace,
                    changed=sorted(changed),
                    removed=list(removed),
                    reason=outcome,
                )
                continue
            self.state_deltas[namespace] = StateDelta(
                changed=changed, removed=removed
            )
            self._emit(
                hook,
                "state_delta",
                tool=self.tool_for(namespace),
                state_namespace=namespace,
                changed=sorted(changed),
                removed=list(removed),
                keys=sorted(self.merged_state(namespace)),
            )
        self._live_states.clear()
        self._state_bases.clear()

    def pending_calls(self) -> list[str]:
        """Ids of local tool calls this block is waiting on right now."""
        return sorted(self._pending)

    def resolve_local_call(
        self,
        call_id: str,
        result: str,
        error: str | None = None,
        images: Any = None,
        call: ToolCall | None = None,
    ) -> bool:
        """Hand a client's answer to the turn that is waiting for it.

        `images` takes the same shapes as `fork`'s and is normalized here, so a
        bad one raises before the waiting turn is woken. An answer that reports
        an `error` is text: images sent with it are dropped.

        `call`, when given, is a `ToolCall` the client's tool wants run next, so
        the answer continues a pipe instead of ending it. It cannot be combined
        with `error` (which ends the pipe) or with `images` (which only the last
        call of a pipe may carry). A rejected answer leaves the call parked, so
        the client can answer again.

        Returns False when nothing is waiting on that id any more — the call
        timed out, was cancelled, or never existed. The turn is woken by the
        event and reads the answer on its own thread.
        """
        if call is not None:
            if not isinstance(call, ToolCall):
                raise HHAgentError("'call' must be a ToolCall")
            call = _check_call(call)
            if error is not None:
                raise HHAgentError(
                    "a local tool answer cannot be both an error and a call"
                )
        parts = image_content_parts(images)
        if call is not None and parts:
            raise HHAgentError(
                "a local tool answer cannot carry both a call and images"
            )
        pending = self._pending.get(call_id)
        if pending is None:
            return False
        pending.result = result
        pending.error = error
        pending.images = parts
        pending.call = call
        pending.event.set()
        return True

    def _answer_local(
        self,
        tool: ToolEntry,
        call_id: str,
        arguments: dict[str, Any],
        raw: Any,
        hook: EventHook | None,
        round_no: int,
        pipe: dict[str, Any] | None = None,
    ) -> tuple[_ToolReply, str | None]:
        """Announce a client-side tool call, then wait for the answer.

        `pipe`, when given, says this call is a step of a pipe rather than the
        model's own call: `parent_call_id`, `step` and `chain` go out with the
        event so the client can see where the call came from, and the client may
        answer with a `call` of its own to keep the pipe going.
        """
        return self._ask_client(
            hook,
            "local_tool_called",
            call_id,
            tool.name,
            "call",
            {
                "round": round_no,
                "arguments": arguments,
                "raw_arguments": raw,
                **(pipe or {}),
            },
            self.local_timeout,
            "timeout",
            stop_on_cancel=True,
        )

    def _ask_client(
        self,
        hook: EventHook | None,
        event: str,
        call_id: str,
        name: str,
        kind: str,
        fields: dict[str, Any],
        timeout: float,
        reason: str,
        stop_on_cancel: bool,
    ) -> tuple[_ToolReply, str | None]:
        """Announce something the client has to do and wait for its answer.

        The wait holds nothing but an event: no registry lock, no store lock and
        no open database transaction, so the rest of the server keeps running
        while a client decides. `reason` labels a give-up when `timeout` is spent,
        and a cancelled turn only raises when `stop_on_cancel` is set — a rollback
        must not replace the failure that caused it.
        """
        pending = _Pending(name=name, event=threading.Event())
        self._pending[call_id] = pending
        started = time.monotonic()
        try:
            # registered before announcing, so an answer that arrives
            # immediately is never lost
            self._emit(
                hook,
                event,
                call_id=call_id,
                name=name,
                kind=kind,
                timeout_ms=round(timeout * 1000, 3),
                **fields,
            )
            deadline = started + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._emit(
                        hook,
                        "local_tool_unresolved",
                        call_id=call_id,
                        name=name,
                        kind=kind,
                        reason=reason,
                        waited_ms=self._ms(started),
                    )
                    return (
                        _reply(
                            f"Error: local tool '{name}' was not answered ({reason})"
                        ),
                        reason,
                    )
                if pending.event.wait(min(0.2, remaining)):
                    break
                if self._cancel.is_set():
                    self._emit(
                        hook,
                        "local_tool_unresolved",
                        call_id=call_id,
                        name=name,
                        kind=kind,
                        reason="cancelled",
                        waited_ms=self._ms(started),
                    )
                    if stop_on_cancel:
                        raise HHAgentCancelled(
                            f"agent {self.id} cancelled while waiting for {name}"
                        )
                    return (
                        _reply(
                            f"Error: local tool '{name}' was not answered (cancelled)"
                        ),
                        "cancelled",
                    )
            text = pending.result or ""
            self._emit(
                hook,
                "local_tool_resolved",
                call_id=call_id,
                name=name,
                kind=kind,
                ok=pending.error is None,
                error=pending.error,
                result=text,
                result_chars=len(text),
                image_count=len(pending.images),
                next=(
                    pending.call.name
                    if kind == "call" and pending.call is not None
                    else None
                ),
                waited_ms=self._ms(started),
            )
            if pending.error is not None:
                # an error answer stays text; any images sent with it are dropped
                return (
                    _reply(text or f"Error: local tool '{name}': {pending.error}"),
                    pending.error,
                )
            reply = _reply(text, pending.images)
            if kind == "call":
                # only a forward call can continue a pipe: an undo's answer is a
                # report, so a call sent with one goes nowhere
                reply.call = pending.call
            return reply, None
        finally:
            self._pending.pop(call_id, None)

    def _rollback(self, hook: EventHook | None, outcome: str) -> list[_Undo]:
        """Undo this turn's tool calls, newest first, because it did not commit.

        Only calls that reached a tool are undone: an unknown tool, or arguments
        that never bound, did nothing. For a client-run tool that was asked and
        never answered — timed out or cancelled — the undo is still offered,
        because the client may have run it before falling silent.

        Every hook is isolated: one that fails is reported and skipped so the rest
        still run, and a failing rollback can never replace the failure that
        triggered it. Remote tools are asked over the protocol; a cancelled or
        abandoned turn is not waited on, so cancelling stays quick.

        Returns one `_Undo` per call whose effects needed attention, newest
        first, for the failure note to report: undone, undo failed, or nothing to
        run because the tool declared effects it cannot take back.
        """
        report: list[_Undo] = []
        if not self._called:
            return report
        calls = list(reversed(self._called))
        total = len(calls)
        for index, call in enumerate(calls, start=1):
            tool = call.tool
            if not tool.has_rollback:
                if tool.external_effects:
                    # nothing to run, but the world moved; only the author knows
                    # whether anything can put it back
                    self._emit(
                        hook,
                        "rollback_unavailable",
                        call_id=call.call_id,
                        tool=tool.name,
                        index=index,
                        total=total,
                    )
                    report.append(_Undo(tool.name, "standing"))
                continue
            self._emit(
                hook,
                "rollback_started",
                call_id=call.call_id,
                tool=tool.name,
                index=index,
                total=total,
            )
            started = time.monotonic()
            if tool.is_local:
                reply, error = self._ask_client(
                    hook,
                    "local_tool_rollback",
                    f"{call.call_id}:rollback",
                    tool.name,
                    "rollback",
                    {
                        "rollback_of": call.call_id,
                        "arguments": call.arguments,
                        "result": call.result,
                        "call_ok": call.ok,
                    },
                    self.local_timeout if outcome == "failed" else 0.0,
                    "timeout" if outcome == "failed" else outcome,
                    stop_on_cancel=False,
                )
                text = reply.text
            else:
                text, error = self._invoke_rollback(tool, call)
            self._emit(
                hook,
                "rollback_finished",
                call_id=call.call_id,
                tool=tool.name,
                ok=error is None,
                error=error,
                result=text,
                result_chars=len(text),
                elapsed_ms=self._ms(started),
            )
            report.append(
                _Undo(tool.name, "undone" if error is None else "stuck", error)
            )
        return report

    def _record_failure(
        self,
        hook: EventHook | None,
        outcome: str,
        rolled_back: list[_Undo],
    ) -> None:
        """Leave a note in the block saying why it ended and what was undone.

        The note is an ordinary message, so a block forked from this one carries
        it into its context — which is the point: the transcript above it may
        still describe tool effects the rollback has since undone, and the next
        model should not trust that blindly.

        Only a failed turn is summarised by the model. A cancelled or abandoned
        one gets a fixed note: summarising would put an extra request in the way
        of the stop the caller just asked for, and an abandoned generator must
        not start network calls during teardown at all.
        """
        summary = None
        if outcome == "failed" and self._worth_summarizing():
            summary = self._summarize_failure(hook, rolled_back)
        note = self._failure_note(outcome, summary, rolled_back)
        self._remember(hook, {"role": "user", "content": note}, "failure")

    def _worth_summarizing(self) -> bool:
        """Whether a summary would say more than the raw error already does.

        A turn that called nothing and produced no text is fully described by the
        error, so summarising it would be a wasted request.
        """
        return bool(self._called or self.text)

    def _failure_note(
        self,
        outcome: str,
        summary: str | None,
        rolled_back: list[_Undo],
    ) -> str:
        """The text left behind: bracketed, so it reads as a harness note."""
        head = {
            "cancelled": "the previous turn was cancelled by the caller",
            "abandoned": "the previous turn was abandoned before it finished",
        }.get(outcome, "the previous turn failed")
        parts = [f"[harness] {head}; the state it changed was discarded."]
        undone = [u.tool for u in rolled_back if u.status == "undone"]
        stuck = [u.tool for u in rolled_back if u.status == "stuck"]
        standing = [u.tool for u in rolled_back if u.status == "standing"]
        if undone:
            parts.append(f"Undone: {', '.join(undone)}.")
        if stuck:
            parts.append(f"Could not be undone: {', '.join(stuck)}.")
        if standing:
            # declared effects with nothing to take them back
            parts.append(f"May still be in effect: {', '.join(standing)}.")
        if self.error:
            # kept verbatim: the summary is written by a model and may distort it
            parts.append(f"Reported error: {self.error}")
        if summary:
            parts.append(f"In short: {summary}")
        elif outcome == "failed":
            parts.append("No summary was available.")
        return "\n".join(parts)

    def _summary_prompt(
        self, rolled_back: list[_Undo]
    ) -> str:
        """The transcript of this turn, for the summariser to read."""
        lines = ["The turn that failed, as it was recorded:", ""]
        for message in self.messages:
            content = message_text(message.get("content"))[:4000]
            if message.get("tool_calls"):
                calls = "; ".join(
                    f"{call['function']['name']}({call['function']['arguments']})"
                    for call in message["tool_calls"]
                )
                content = f"{content} [tool calls: {calls}]".strip()
            lines.append(f"{message.get('role')}: {content}")
        if self.text:
            lines += ["", f"assistant, cut off mid-answer: {self.text[:4000]}"]
        lines += [
            "",
            f"The turn ended as: {self.outcome}",
            f"The reported error: {self.error}",
        ]
        undone = [u.tool for u in rolled_back if u.status == "undone"]
        stuck = [u.tool for u in rolled_back if u.status == "stuck"]
        standing = [u.tool for u in rolled_back if u.status == "standing"]
        if undone:
            lines.append(f"Effects that were undone: {', '.join(undone)}")
        if stuck:
            lines.append(f"Effects that could NOT be undone: {', '.join(stuck)}")
        if standing:
            lines.append(
                "Effects with no undo available, which may still be in effect: "
                + ", ".join(standing)
            )
        return "\n".join(lines)

    def _summarize_failure(
        self, hook: EventHook | None, rolled_back: list[_Undo]
    ) -> str | None:
        """Ask the model to write the note. Returns None if that fails too.

        Deliberately a plain request with no tools: the summariser cannot call
        anything, it only writes. A failure here is reported and then ignored —
        the note falls back to the raw error, which is the part that matters.
        """
        model = self.summary_model or self.model
        url = f"{self.endpoint}/v1/chat/completions"
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": self._summary_prompt(rolled_back)},
                ],
                "stream": True,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self._emit(
            hook,
            "failure_summary_started",
            model=model,
            error=self.error,
            outcome=self.outcome,
            messages=len(self.messages),
            request_bytes=len(body),
            timeout=self.timeout,
        )
        started = time.monotonic()
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
                "failure_summary_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                elapsed_ms=self._ms(started),
            )
            return None
        with response:
            response.encoding = "utf-8"
            if not response.ok:
                self._emit(
                    hook,
                    "failure_summary_failed",
                    status=response.status_code,
                    error=response.text[:500],
                    error_type="HTTPError",
                    elapsed_ms=self._ms(started),
                )
                return None
            parts: list[str] = []
            chunks, usage, finish = 0, None, None
            try:
                for line in self._sse_payloads(response):
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunks += 1
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    if choices[0].get("finish_reason"):
                        finish = choices[0]["finish_reason"]
                    text = (choices[0].get("delta") or {}).get("content") or ""
                    if text:
                        parts.append(text)
            except requests.RequestException as exc:
                self._emit(
                    hook,
                    "failure_summary_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    chunks=chunks,
                    elapsed_ms=self._ms(started),
                )
                return None
        summary = "".join(parts).strip()
        self._emit(
            hook,
            "failure_summary_finished",
            status=response.status_code,
            summary=summary,
            chars=len(summary),
            chunks=chunks,
            finish_reason=finish,
            usage=usage,
            elapsed_ms=self._ms(started),
        )
        return summary or None

    def _invoke_rollback(self, tool: ToolEntry, call: _Call) -> tuple[str, str | None]:
        """Run one server-side rollback hook, containing whatever it throws."""
        rollback = tool.rollback
        if rollback is None:
            return "", None
        context = ToolContext(
            tool=tool,
            agent=self,
            call_id=call.call_id,
            arguments=call.arguments,
            raw_arguments=call.raw_arguments,
            # a copy, so whatever a rollback does to it cannot muddy the
            # `state_discarded` report this same turn is about to emit
            state=copy.deepcopy(self._live_states.get(tool.namespace, {})),
            result=call.result,
            tools=dict(self.tools),
        )
        try:
            returned = rollback(context, **call.arguments)
        except TypeError as exc:
            return (
                f"Error: rollback for '{tool.name}' had bad arguments: {exc}",
                "bad_arguments",
            )
        except Exception as exc:
            # warned and ignored: the remaining rollbacks must still run
            failure = f"{type(exc).__name__}: {exc}"
            return (f"Error: rollback for '{tool.name}' raised {failure}", "rollback_raised")
        if isinstance(returned, str):
            return returned, None
        # an undo reports what it did; it cannot itself pipe, so anything else
        # is a bug worth naming rather than a value to store
        return (
            f"Error: rollback for '{tool.name}' returned "
            f"{type(returned).__name__}, expected a string",
            "bad_result",
        )

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
                image_count=self.image_count,
                depth=self.depth,
                path=self.path(),
                model=self.model,
                tools=sorted(self.tools),
                context_len=len(self.context()),
                include_usage=self.include_usage,
                verbose=self.verbose,
                max_tokens=self.max_tokens,
            )
            # already in `messages` since fork; only announce it here
            self._announce(hook, self.messages[0], "fork")
            for round_no in range(1, MAX_TOOL_ROUNDS + 1):
                turn = _Turn()
                try:
                    yield from self._stream_turn(turn, hook, round_no)
                finally:
                    # captured even when the round fails mid-stream, so a failed
                    # turn still reports what the model had said so far
                    if turn.content:
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
            rolled_back = (
                self._rollback(hook, outcome) if outcome != "commit" else []
            )
            self._release_states(hook, outcome)
            self.outcome = "ok" if outcome == "commit" else outcome
            if outcome != "commit":
                # recorded after the rollback, so the note can say what was
                # undone, and before `dirty` clears, so a fork from this block
                # can never miss it
                self.text = "".join(parts)
                self._record_failure(hook, outcome, rolled_back)
            self._called.clear()
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
            # calls the model asked for are counted above; these are the ones
            # the tools added themselves by piping
            pipe_steps=sum(len(trace["steps"]) - 1 for trace in self.pipe_traces),
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
        if self.max_tokens > 0:
            # unlike an absent field, an explicit cap is enforced by the provider
            payload["max_tokens"] = self.max_tokens
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
            max_tokens=self.max_tokens,
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
            content = message.get("content")
            entry: dict[str, Any] = {
                "role": message.get("role"),
                "chars": len(message_text(content)),
            }
            images = message_image_count(content)
            if images:
                entry["image_count"] = images
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
        return [self._run_call(call, hook, round_no) for call in calls]

    def _run_call(
        self, call: dict[str, Any], hook: EventHook | None, round_no: int
    ) -> dict[str, Any]:
        """Run one model-requested call, following any pipe it starts.

        A tool answers with text, which ends the call, or with a `ToolCall`,
        which becomes the next step of a *pipe*: the harness runs that call and
        carries on until one of them returns text. Only the tools the pipe
        called and the last call's output reach the model; each step's own
        arguments and result are kept in `pipe_traces` and reported through
        events, so they can be inspected without ever entering the transcript —
        or the database, which they are deliberately never written to.

        Every step is also a `_Call` of its own, recorded as it runs, so a turn
        that does not commit offers each of them an undo, newest first.
        """
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
            hook, "tool_call_started", round=round_no, call_id=call_id, name=name
        )
        started = time.monotonic()
        # the checks keep the order they always had: a name nobody knows is
        # reported as such even when its arguments are junk too
        arguments: Any = {}
        if name not in self.tools:
            reply, error, via = (
                _reply(f"Error: unknown tool '{name}'"),
                "unknown_tool",
                "server",
            )
        else:
            arguments, parse_error = _parse_arguments(name, raw)
            if parse_error is not None:
                reply, error, via = _reply(parse_error), "bad_arguments", "server"
            elif not isinstance(arguments, dict):
                reply, error, via = (
                    _reply(f"Error: arguments for '{name}' must be a JSON object"),
                    "bad_arguments",
                    "server",
                )
            else:
                reply, error, via = self._invoke(
                    name, arguments, raw, call_id, hook, round_no, {}
                )
        # the trace keeps the model's own call as step 0 and every piped call
        # after it, in the order they ran
        steps = [
            _step_record(
                call_id,
                name,
                via,
                arguments if isinstance(arguments, dict) else {},
                reply,
                error,
                self._ms(started),
            )
        ]
        while error is None and reply.call is not None:
            if len(steps) > MAX_PIPE_DEPTH:
                # the pipe's own calls are not the model's turns, so the guard
                # is here: a tool that keeps asking for itself must still end
                reply, error = (
                    _reply(
                        f"Error: tool pipe from '{name}' stopped after "
                        f"{MAX_PIPE_DEPTH} steps"
                    ),
                    "pipe_depth",
                )
                break
            step_call = reply.call
            step_no = len(steps)
            chain = [step["name"] for step in steps] + [step_call.name]
            step_id = f"{call_id}:pipe:{step_no}"
            pipe = {"parent_call_id": call_id, "step": step_no, "chain": chain}
            step_tool = self.tools.get(step_call.name)
            step_via = (
                "client"
                if step_tool is not None and step_tool.is_local
                else "server"
            )
            step_started = time.monotonic()
            self._emit(
                hook,
                "pipe_step_started",
                round=round_no,
                call_id=step_id,
                name=step_call.name,
                via=step_via,
                known=step_tool is not None,
                arguments=_bounded(step_call.arguments),
                **pipe,
            )
            step_reply, step_error, step_via = self._invoke(
                step_call.name,
                step_call.arguments,
                step_call.arguments,
                step_id,
                hook,
                round_no,
                pipe,
            )
            steps.append(
                _step_record(
                    step_id,
                    step_call.name,
                    step_via,
                    step_call.arguments,
                    step_reply,
                    step_error,
                    self._ms(step_started),
                )
            )
            self._emit(
                hook, "pipe_step_finished", round=round_no, **steps[-1], **pipe
            )
            reply, error = step_reply, step_error
        extra: dict[str, Any] = {}
        if len(steps) > 1:
            chain = [step["name"] for step in steps]
            header = f"[tool pipe] {' -> '.join(chain)}"
            text = f"{header}\n{reply.text}" if reply.text else header
            content = content_with_images(text, reply.parts)
            self.pipe_traces.append(
                {
                    "call_id": call_id,
                    "round": round_no,
                    "chain": chain,
                    "ok": error is None,
                    "error": error,
                    "result": _bounded(reply.text),
                    "result_chars": len(reply.text),
                    "steps": steps,
                    "elapsed_ms": self._ms(started),
                }
            )
            extra = {"chain": chain, "steps": len(steps) - 1}
        else:
            text, content = reply.text, reply.content
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
            image_count=reply.image_count,
            elapsed_ms=self._ms(started),
            **extra,
        )
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    def _invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        raw: Any,
        call_id: str,
        hook: EventHook | None,
        round_no: int,
        pipe: dict[str, Any],
    ) -> tuple[_ToolReply, str | None, str]:
        """Run one tool call — server-side or by asking the client — and record it.

        Returns the reply, an error code when it failed, and where it ran
        (`server` or `client`). Every call that reached a tool is appended to
        `_called`, a pipe's steps included, so a failed turn can undo them all.
        `pipe` is empty for the model's own call and names the parent, step and
        chain for a piped one, which is what a parked client is told.
        """
        tool = self.tools.get(name)
        if tool is None:
            return _reply(f"Error: unknown tool '{name}'"), "unknown_tool", "server"
        if tool.is_local:
            # no hook and no state on this side: the client runs it
            # recorded before asking, because a client may run a tool and
            # then fail to answer — a cancelled or timed-out call still
            # needs its undo offered
            call = _Call(tool, call_id, arguments, raw, "", False)
            self._called.append(call)
            reply, error = self._answer_local(
                tool, call_id, arguments, raw, hook, round_no, pipe
            )
            call.result, call.ok = reply.text, error is None
            return reply, error, "client"
        context = ToolContext(
            tool=tool,
            agent=self,
            call_id=call_id,
            arguments=arguments,
            raw_arguments=raw,
            state=self._live_state(tool.namespace),
            tools=dict(self.tools),
        )
        self._emit(
            hook,
            "state_loaded",
            round=round_no,
            call_id=call_id,
            tool=name,
            state_namespace=tool.namespace,
            keys=sorted(context.state),
            inherited=len(self._state_bases.get(tool.namespace) or {}),
        )
        reply, error = _run_hook(tool, context)
        if error not in NOT_RUN:
            self._called.append(
                _Call(tool, call_id, arguments, raw, reply.text, error is None)
            )
        return reply, error, "server"

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
        text = message_text(content)
        self._emit(
            hook,
            "history_appended",
            source=source,
            role=message.get("role"),
            chars=len(text),
            preview=text[:200],
            image_count=message_image_count(content),
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
