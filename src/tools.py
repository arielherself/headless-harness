import json
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import requests

from protocol import PROTOCOL_VERSION

if TYPE_CHECKING:  # only for the annotation below; agent imports this module
    from agent import HHAgent


@dataclass
class ToolParam:
    name: str
    type: str
    description: str


@dataclass
class ToolResult:
    """What a hook hands back: text, optionally with images.

    A hook may return a plain string, as before, or one of these when the model
    should see images too. `images` takes the same shapes `fork` accepts: an
    `http(s)` URL or `data:image/...` URI string, or a `{url, detail}` mapping.
    A local path is refused — the harness never reads a file for a tool, and it
    will not hand a provider one to open. The text is what events, failure
    summaries and rollback reports carry; the image bytes only ever go into the
    model's tool message.
    """

    text: str = ""
    images: tuple[Any, ...] = ()


@dataclass
class ToolEntry:
    """A tool the model may call: its schema plus the hook that runs it.

    A tool with no `hook` runs on the *client*: the server cannot execute it, so
    it announces the call and waits for an answer instead (see `local_tools` on
    `create_agent` and the `resolve_tool` command). Local tools are visible to
    the model exactly like any other, but they carry no state on the server.

    `state_namespace` is where the tool's state lives. Left empty, every tool
    gets a private namespace named after itself; tools that should share one
    memory — a get/set pair, say — declare the same `state_namespace`.
    """

    name: str
    description: str
    params: list[ToolParam]
    hook: Callable[..., str | ToolResult] | None = None
    state_namespace: str = ""
    # undo for this tool's calls, run when a turn does not commit; for a
    # client-run tool there is nothing callable here, so `remote_rollback`
    # says instead that the client has one to ask for
    rollback: Callable[..., str] | None = None
    remote_rollback: bool = False
    # Declares that calls to this tool reach beyond the state: they installed
    # something, wrote a file, sent a request. State rolls back on its own, so
    # this is what lets a failure note warn that such effects may still stand.
    # It says nothing about whether an undo exists — an irreversible effect is a
    # legitimate declaration, not a mistake to be flagged.
    external_effects: bool = False

    @property
    def is_local(self) -> bool:
        """Whether the client answers this tool instead of a server-side hook."""
        return self.hook is None

    @property
    def has_rollback(self) -> bool:
        """Whether a call to this tool can be undone at all."""
        return self.rollback is not None or self.remote_rollback

    @property
    def namespace(self) -> str:
        """The namespace this tool's state actually lives in.

        This is the resolved form; `state_namespace` is what the author declared.
        """
        return self.state_namespace or self.name


@dataclass
class ToolContext:
    """What a hook is handed besides its own arguments.

    `state` is the tool's persistent memory: a mutable dict already overlaid
    with every delta its ancestors committed. A tool may edit it in place,
    including nested values; whatever differs when a *successful* turn ends is
    frozen into the running agent block, and blocks forked from that point
    inherit it. Turns that fail, are cancelled, or are abandoned contribute
    nothing, so a tool's memory never advances through them.

    `state` holds one namespace's worth of memory: the namespace the tool's
    `state_namespace` names, which defaults to the tool's own name. So by default
    no two tools see each other's keys, and tools that declare the same
    `state_namespace` share one memory.

    Values may be any Python object, but two things follow from that. State is
    copied with `deepcopy` so each turn works on its own copy, so a value that
    cannot be deep-copied will fail its turn. And anything not JSON-serialisable
    reaches TCP clients as its `str()`, so keep state JSON-shaped if you want it
    read faithfully from outside.

    Granularity is the top-level key: a tool that buries everything under one
    key makes every block store that whole value.
    """

    tool: ToolEntry
    agent: "HHAgent"
    call_id: str
    arguments: dict[str, Any]
    raw_arguments: Any
    state: dict[str, Any]
    # set only for a rollback: the text the call being undone returned
    result: str | None = None


def get_system_info_executor(context: ToolContext, **arguments: Any) -> str:
    # `context.agent` is the live block; a bare context (a test's) has none
    model = getattr(context.agent, "model", "") or "<unknown>"
    return (
        f"Headless Harness protocol version {PROTOCOL_VERSION}\n"
        f"https://github.com/arielherself/headless-harness\n"
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"Host: {platform.node()}\n"
        f"Python: {platform.python_version()}\n"
        f"CPU count: {os.cpu_count()}\n"
        f"Model: {model}"
    )


get_system_info_tool = ToolEntry(
    name="get_system_info",
    description=(
        "Get information about the machine the harness runs on "
        "(OS, host, Python, CPUs, underlying model)."
    ),
    params=[],
    hook=get_system_info_executor,
)


def get_current_time_executor(context: ToolContext, **arguments: Any) -> str:
    utc = datetime.now(timezone.utc)
    local = utc.astimezone()
    return (
        f"UTC: {utc.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Local: {local.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')}"
    )


get_current_time_tool = ToolEntry(
    name="get_current_time",
    description="Get current time (UTC and local).",
    params=[],
    hook=get_current_time_executor,
)


def set_magic_number_executor(context: ToolContext, **arguments: Any) -> str:
    new_magic = arguments["magic"]
    context.state["magic"] = new_magic
    return f"Magic is set to {new_magic}"


set_magic_number_tool = ToolEntry(
    name="set_magic_number",
    description="Set the magic number for the current chat.",
    params=[ToolParam(name="magic", type="string", description="the new magic number")],
    hook=set_magic_number_executor,
    state_namespace="magic",
)


def get_magic_number_executor(context: ToolContext, **arguments: Any) -> str:
    old_magic = "<null>"
    if "magic" in context.state:
        old_magic = context.state["magic"]
    return old_magic


get_magic_number_tool = ToolEntry(
    name="get_magic_number",
    description="Get the magic number for the current chat.",
    params=[],
    hook=get_magic_number_executor,
    state_namespace="magic",
)

def _jsonrpc_replies(text: str) -> list[dict[str, Any]]:
    """The JSON-RPC payloads in an MCP answer: SSE frames or one JSON body."""
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        replies = []
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                replies.append(json.loads(line[5:].strip()))
            except json.JSONDecodeError:
                continue
        return replies


def _truncate(text: str, limit: int) -> str:
    """Cut a tool result to `limit` characters, saying how much was dropped."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[truncated: showing the first {limit} of {len(text)} characters]"


# Jina's reader renders a URL as Markdown for a model to read. An API key is
# optional — it lifts the anonymous rate limit — so it is read from the
# environment when the operator has one.
JINA_READER = "https://r.jina.ai/"
WEB_FETCH_TIMEOUT = 180.0
# a tool result stays in the transcript of every later request, so a huge page is
# cut off rather than carried forever
WEB_FETCH_MAX_CHARS = 20_000


def web_fetch_executor(context: ToolContext, url: str) -> str:
    target = url.strip()
    if "://" not in target:
        target = f"https://{target}"
    headers = {"Accept": "text/plain", "X-Return-Format": "markdown"}
    key = os.environ.get("JINA_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        response = requests.get(
            f"{JINA_READER}{target}", headers=headers, timeout=WEB_FETCH_TIMEOUT
        )
    except requests.RequestException as exc:
        return f"Error: could not reach Jina Reader: {exc}"
    text = response.content.decode("utf-8", "replace").strip()
    if not response.ok:
        detail = " ".join(text.split())[:200] or response.reason
        return f"Error: Jina Reader answered HTTP {response.status_code}: {detail}"
    if not text:
        return "Error: Jina Reader returned an empty document."
    return _truncate(text, WEB_FETCH_MAX_CHARS)


web_fetch_tool = ToolEntry(
    name="web_fetch",
    description=(
        "Read a web page as Markdown, through Jina AI's reader (r.jina.ai). What "
        "comes back is Jina's extraction of the page — its main content rendered "
        "and condensed as Markdown, not the raw HTML — so scripts, styles, "
        "navigation and anything Jina cannot render are missing. Use it to read "
        "documentation, articles and API references; it cannot reach pages behind "
        "a login, and it is not a search engine."
    ),
    params=[
        ToolParam(
            name="url",
            type="string",
            description="the page to read, e.g. https://example.com/docs",
        )
    ],
    hook=web_fetch_executor,
    # it reads: the request changes nothing a later turn would have to be warned
    # about, so no external effect is declared and a failed turn stays quiet
    external_effects=False,
)


# Exa's search is an MCP endpoint that answers a JSON-RPC `tools/call` over plain
# HTTP — no key, no session — so one POST is the whole conversation. Their free
# tier rate-limits by IP.
EXA_MCP = "https://mcp.exa.ai/mcp"
WEB_SEARCH_TIMEOUT = 60.0
# a handful of results is what a model can act on, and a second, sharper query
# beats a longer list
WEB_SEARCH_RESULTS = 5
WEB_SEARCH_MAX_CHARS = 20_000


def web_search_executor(context: ToolContext, query: str, objective: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search_exa",
            "arguments": {
                "query": query,
                "objective": objective,
                "numResults": WEB_SEARCH_RESULTS,
            },
        },
    }
    try:
        response = requests.post(
            EXA_MCP,
            json=payload,
            headers={"Accept": "application/json, text/event-stream"},
            timeout=WEB_SEARCH_TIMEOUT,
        )
    except requests.RequestException as exc:
        return f"Error: could not reach Exa: {exc}"
    replies = [reply for reply in _jsonrpc_replies(response.text) if isinstance(reply, dict)]
    failure = next((reply["error"] for reply in replies if "error" in reply), None)
    if failure is not None:
        message = failure.get("message") if isinstance(failure, dict) else failure
        return f"Error: Exa refused the search: {message}"
    if not response.ok:
        detail = " ".join(response.text.split())[:200]
        return f"Error: Exa answered HTTP {response.status_code}: {detail}"
    result = next((reply["result"] for reply in replies if "result" in reply), None)
    if not isinstance(result, dict):
        return "Error: Exa's answer held no result."
    text = "\n".join(
        block.get("text", "")
        for block in result.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    if result.get("isError"):
        return f"Error: Exa could not run the search: {text or 'no detail'}"
    if not text:
        return "Error: Exa returned no results."
    return _truncate(text, WEB_SEARCH_MAX_CHARS)


web_search_tool = ToolEntry(
    name="web_search",
    description=(
        "Search the web through Exa and get back short, read-ready excerpts of "
        "the top results — title, URL and highlights, not whole pages. Use it when "
        "you need current information or a good page to read but have no URL yet, "
        "then follow up with web_fetch to read a result in full. It cannot search "
        "inside private or login-gated sources."
    ),
    params=[
        ToolParam(
            name="query",
            type="string",
            description=(
                "a description of the page you want, not keywords — e.g. 'blog "
                "post comparing React and Vue performance'"
            ),
        ),
        ToolParam(
            name="objective",
            type="string",
            description=(
                "what this search is for: which documents should rank first and "
                "which facts to pull out of them"
            ),
        ),
    ],
    hook=web_search_executor,
    # a read as well: nothing it does leaves a later turn anything to trip over
    external_effects=False,
)


builtin_tools = [
    get_system_info_tool,
    get_current_time_tool,
    set_magic_number_tool,
    get_magic_number_tool,
    web_fetch_tool,
    web_search_tool,
]
