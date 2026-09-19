import json
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import requests

import sandbox_tools
from protocol import PROTOCOL_VERSION

if TYPE_CHECKING:  # only for the annotation below; agent imports this module
    from agent import HHAgent


@dataclass
class ToolParam:
    name: str
    type: str
    description: str


@dataclass
class ToolCall:
    """A tool call one tool asks the harness to run before it is done.

    A hook returns text when it has finished. Returning one of these instead —
    on its own, or as `ToolResult(...).call` — puts the call at the end of a
    *pipe*: the harness runs `name` with `arguments` as if the model had asked
    for it, and hands the result to whatever that tool returns in turn.

    `arguments` is an ordinary Python dict, so a value too big or too awkward
    for a model to quote (a file's bytes, a page of HTML) can travel through a
    pipe without ever entering the transcript. Only the tools a pipe called and
    the *last* call's output reach the model; what the earlier calls returned,
    arguments included, is recorded in memory for inspection (see
    `HHAgent.pipe_traces`) and reported through events, but never sent to the
    provider and never written to the store.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """What a hook hands back: text, optionally with images or a next call.

    A hook may return a plain string, as before, or one of these when the model
    should see images too. `images` takes the same shapes `fork` accepts: an
    `http(s)` URL or `data:image/...` URI string, or a `{url, detail}` mapping.
    A local path is refused — the harness never reads a file for a tool, and it
    will not hand a provider one to open. The text is what events, failure
    summaries and rollback reports carry; the image bytes only ever go into the
    model's tool message.

    `call`, when set, makes this result the first half of a pipe instead: the
    text is recorded for inspection but not shown, `images` may not be set (the
    model only ever sees the last call's), and the named tool runs next. A
    `ToolCall` returned on its own means the same thing with no text at all.
    """

    text: str = ""
    images: tuple[Any, ...] = ()
    call: ToolCall | None = None


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
    # Every tool the block offers, keyed by name: what a hook may pipe to, with
    # the parameters each expects. A name that is not here cannot be piped to —
    # the harness answers with an unknown-tool error — so a piping tool can
    # check first. The mapping is a copy; editing it changes nothing.
    tools: Mapping[str, ToolEntry] = field(default_factory=dict)


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


# The sandbox tools. The sandbox itself is the `sandbox` package at the project
# root; `sandbox_tools.py` owns the registry and the fixed configuration, and
# these entries are the model's side of it. Ids live in memory and in the
# conversation, never in tool state: forking a block rewinds tool memory but
# cannot resurrect a sandbox, which `nix_sandbox_status` makes observable.


def nix_spawn_sandbox_executor(context: ToolContext) -> str:
    return sandbox_tools.SANDBOXES.spawn()


nix_spawn_sandbox_tool = ToolEntry(
    name="nix_spawn_sandbox",
    description=(
        "Create an isolated Nix sandbox and return its id; every other nix_* "
        "tool takes that id. The sandbox has a fixed shape that cannot be "
        "changed: 256M memory, 512M disk, 256 processes, one CPU, no network "
        "access, and a writable /workspace that is also the working directory. "
        "It cannot reach the network — there is no DNS either — so anything "
        "that fetches at runtime (curl, pip install, git clone) fails. "
        "nix_add_dependency is unaffected, because Nix builds run in the "
        "harness process, outside the sandbox. It starts with only bash and "
        "coreutils. Use it — rather than the machine the harness runs on — "
        "for untrusted code, package installs and anything else that should "
        "be contained, and destroy it with nix_destroy_sandbox when the work "
        "is done."
    ),
    params=[],
    hook=nix_spawn_sandbox_executor,
    external_effects=True,
)


def nix_sandbox_status_executor(context: ToolContext, sandbox_id: str) -> str:
    return sandbox_tools.SANDBOXES.status(sandbox_id)


nix_sandbox_status_tool = ToolEntry(
    name="nix_sandbox_status",
    description=(
        "Report whether a sandbox is still live, with its age, last use and "
        "installed packages. Sandboxes are released when destroyed, after ten "
        "minutes without a call, or when the harness restarts, because ids are "
        "kept in memory only — so ask before assuming a sandbox you created "
        "earlier still exists. Calling this also counts as using the sandbox "
        "and restarts its idle clock."
    ),
    params=[
        ToolParam(name="sandbox_id", type="string", description="the id nix_spawn_sandbox returned"),
    ],
    hook=nix_sandbox_status_executor,
    external_effects=False,
)


def nix_add_dependency_executor(context: ToolContext, sandbox_id: str, package: str) -> str:
    return sandbox_tools.SANDBOXES.add_dependency(sandbox_id, package)


nix_add_dependency_tool = ToolEntry(
    name="nix_add_dependency",
    description=(
        "Install one Nix package into a live sandbox (python312, git, "
        "ripgrep, ...). The change applies to the next nix_exec; if Nix cannot "
        "build the package, the sandbox keeps the environment it had. Add one "
        "package per call."
    ),
    params=[
        ToolParam(name="sandbox_id", type="string", description="the id nix_spawn_sandbox returned"),
        ToolParam(
            name="package",
            type="string",
            description="Nix package name, e.g. python312, git or ripgrep",
        ),
    ],
    hook=nix_add_dependency_executor,
    external_effects=True,
)


def nix_remove_dependency_executor(context: ToolContext, sandbox_id: str, package: str) -> str:
    return sandbox_tools.SANDBOXES.remove_dependency(sandbox_id, package)


nix_remove_dependency_tool = ToolEntry(
    name="nix_remove_dependency",
    description=(
        "Remove one Nix package from a live sandbox. bash and coreutils are "
        "always present and cannot be removed. The change applies to the next "
        "nix_exec."
    ),
    params=[
        ToolParam(name="sandbox_id", type="string", description="the id nix_spawn_sandbox returned"),
        ToolParam(
            name="package",
            type="string",
            description="Nix package name to remove, as passed to nix_add_dependency",
        ),
    ],
    hook=nix_remove_dependency_executor,
    external_effects=True,
)


def nix_exec_executor(
    context: ToolContext, sandbox_id: str, command: str, timeout: int
) -> str:
    return sandbox_tools.SANDBOXES.exec(sandbox_id, command, timeout)


nix_exec_tool = ToolEntry(
    name="nix_exec",
    description=(
        "Run one shell command in a live sandbox and wait for it, returning "
        "its exit code, stdout and stderr. `command` is passed to `bash -c`. "
        "`timeout` is required and may not exceed 600 seconds; a command "
        "that outlives it is killed. Each call runs in fresh namespaces: "
        "background processes from an earlier call are gone, and only files "
        "under /workspace persist."
    ),
    params=[
        ToolParam(name="sandbox_id", type="string", description="the id nix_spawn_sandbox returned"),
        ToolParam(
            name="command",
            type="string",
            description="the shell command line to run, e.g. 'python3 /workspace/main.py'",
        ),
        ToolParam(
            name="timeout",
            type="integer",
            description="seconds to wait before the command is killed; at most 600",
        ),
    ],
    hook=nix_exec_executor,
    external_effects=True,
)


def nix_add_file_executor(
    context: ToolContext, sandbox_id: str, path: str, content_base64: str
) -> str:
    return sandbox_tools.SANDBOXES.add_file(sandbox_id, path, content_base64)


nix_add_file_tool = ToolEntry(
    name="nix_add_file",
    description=(
        "Write one file into a live sandbox at `path`, which must be under "
        "the writable /workspace. `content_base64` is the file's bytes, "
        "base64-encoded, so text and binary files both work, up to 200 MiB "
        "per call; a file whose content starts with '#!' is made executable. "
        "The file lives as long as the sandbox."
    ),
    params=[
        ToolParam(name="sandbox_id", type="string", description="the id nix_spawn_sandbox returned"),
        ToolParam(
            name="path",
            type="string",
            description="absolute destination path inside the sandbox, e.g. /workspace/main.py",
        ),
        ToolParam(
            name="content_base64",
            type="string",
            description="the file's bytes as a base64 string (standard alphabet; padding optional)",
        ),
    ],
    hook=nix_add_file_executor,
    external_effects=True,
)


def nix_destroy_sandbox_executor(context: ToolContext, sandbox_id: str) -> str:
    return sandbox_tools.SANDBOXES.destroy(sandbox_id)


nix_destroy_sandbox_tool = ToolEntry(
    name="nix_destroy_sandbox",
    description=(
        "Destroy a sandbox now: stop anything still running, delete its files "
        "and free its slot. Call it as soon as you are done; an idle sandbox "
        "is released automatically, but not instantly."
    ),
    params=[
        ToolParam(name="sandbox_id", type="string", description="the id nix_spawn_sandbox returned"),
    ],
    hook=nix_destroy_sandbox_executor,
    external_effects=True,
)


builtin_tools = [
    get_system_info_tool,
    get_current_time_tool,
    set_magic_number_tool,
    get_magic_number_tool,
    web_fetch_tool,
    web_search_tool,
    nix_spawn_sandbox_tool,
    nix_sandbox_status_tool,
    nix_add_dependency_tool,
    nix_remove_dependency_tool,
    nix_exec_tool,
    nix_add_file_tool,
    nix_destroy_sandbox_tool,
]
