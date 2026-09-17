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
    """A tool the model may call: its schema plus the hook that runs it."""

    name: str
    description: str
    params: list[ToolParam]
    hook: Callable[..., str]


@dataclass
class ToolContext:
    """What a hook is handed besides its own arguments.

    `state` is the tool's private, persistent memory: a mutable dict already
    overlaid with every delta its ancestors committed. A tool may edit it in
    place, including nested values; whatever differs when a *successful* turn
    ends is frozen into the running agent block, and blocks forked from that
    point inherit it. Turns that fail, are cancelled, or are abandoned
    contribute nothing, so a tool's memory never advances through them.

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

builtin_tools = [get_current_time_tool]
