from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # only for the annotation below; agent imports this module
    from agent import HHAgent


@dataclass
class ToolParam:
    name: str
    type: str
    description: str


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
    hook: Callable[..., str] | None = None
    state_namespace: str = ""

    @property
    def is_local(self) -> bool:
        """Whether the client answers this tool instead of a server-side hook."""
        return self.hook is None

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

builtin_tools = [get_current_time_tool, set_magic_number_tool, get_magic_number_tool]
