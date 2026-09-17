from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ToolParam:
    name: str
    type: str
    description: str


@dataclass
class ToolEntry:
    name: str
    description: str
    params: list[ToolParam]
    hook: Callable[..., str]


def get_current_time_executor() -> str:
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
