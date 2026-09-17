# Writing a tool

A tool is a `ToolEntry` in `src/tools.py`: a name, a description, its parameters,
and a hook the model's call is dispatched to.

## The shape of a tool

```python
from tools import ToolContext, ToolEntry, ToolParam


def add_executor(context: ToolContext, a: int, b: int) -> str:
    return str(a + b)


add_tool = ToolEntry(
    name="add",
    description="Add two integers.",
    params=[
        ToolParam(name="a", type="integer", description="first addend"),
        ToolParam(name="b", type="integer", description="second addend"),
    ],
    hook=add_executor,
)
```

| Field | Meaning |
|---|---|
| `name` | what the model calls; also the default state namespace |
| `description` | shown to the model — this is prompt text, so write it well |
| `params` | `ToolParam(name, type, description)`; every one is sent as `required` |
| `hook` | `hook(context, **arguments) -> str` |
| `state_namespace` | where the tool's state lives; empty means its own name (see below) |

The hook is always called with the context first and the model's arguments as
keywords, so a tool with no parameters is `def hook(context: ToolContext) -> str`.

Register it by adding it to `builtin_tools` at the bottom of `tools.py`. Clients
then choose which tools a block gets:

```jsonc
{"command":"create_agent","id":"root"}                    // every builtin
{"command":"create_agent","id":"root","tools":["add"]}     // just this one
{"command":"fork","id":"root","prompt":"…","tools":[]}     // no tools at all
```

## What the hook receives

`ToolContext` carries everything the tool needs:

| Field | Meaning |
|---|---|
| `tool` | the `ToolEntry`, so a shared hook can branch on `tool.name` |
| `agent` | the running block — `agent.context()` is the conversation, `agent.depth`, `agent.path()`, `agent.id` |
| `call_id` | the provider's id for this call |
| `arguments` | the decoded arguments dict |
| `raw_arguments` | exactly what the model sent, before decoding |
| `state` | this namespace's memory, already overlaid with its ancestors' deltas |

## Persistent state

`context.state` is a mutable `dict`. Edit it freely — including nested values —
and whatever differs when a **successful** turn ends is frozen into the running
block as a delta, inherited by everything forked from that point.

```python
def remember_executor(context: ToolContext, key: str, value: str) -> str:
    context.state[key] = value
    return f"remembered {key}"
```

Because deltas are stored per block and replayed by walking the chain, forking a
block rewinds tool memory to exactly that point:

```
block A  installs python3-base      → A holds {packages: [base]}
block B  (forked from A) installs python3-requests
fork A again → the new branch sees [base] and no requests
```

A tool that never touches `state` produces no delta at all, so stateless tools
cost nothing.

### Namespaces

State is keyed by **namespace**, not by tool name. By default a tool's namespace
is its own name, so no two tools see each other's keys. Tools that should share
one memory declare the same `state_namespace`:

```python
set_magic_number_tool = ToolEntry(..., state_namespace="magic")
get_magic_number_tool = ToolEntry(..., state_namespace="magic")
```

That is how a get/set pair works: isolation is the default, sharing is explicit.
`ToolEntry.namespace` is the resolved value (`state_namespace` or the name).

Keep namespaces cheap to `deepcopy` and, if you want to read them from outside,
JSON-shaped. Nothing else constrains the values.

## Constraints

These are the sharp edges; each one exists for a reason.

**State is deep-copied before your hook runs.** Your edits cannot corrupt the
deltas stored in ancestor blocks — which is also what lets sibling branches run
concurrently without locks. The cost: a value that cannot be deep-copied fails
its turn. Do not put sockets, locks, or generators in state.

**Granularity is the top-level key.** Editing `state["packages"]` stores the new
value of `packages` for that block. Bury everything under one key and every block
stores that whole value; keep state flat.

**State is invisible to the model.** The model only sees the string your hook
returns. If it should know something happened, say so in the result — state will
not tell it.

**A failed turn discards state.** If the turn errors, is cancelled, or the caller
walks away from the stream, no delta is committed, and `state_discarded` reports
what was dropped. Design tools so that re-running a turn is safe.

**Persistence pickles state per namespace.** A namespace whose values refuse to
pickle loses that one namespace on save — `persist_warning` names it — rather than
failing the block. Changing a tool's `state_namespace` makes its historical state
in old blocks unreachable.

**`state` is not the place for per-call scratch space.** Use locals; anything left
in `state` at the end of a successful turn is committed.

## What happens when a tool fails

You do not need to defend against bad input: the harness catches the common
failures and reports them to the model as the tool's result, with `ok: false` and
an `error` code.

| `error` | Cause |
|---|---|
| `unknown_tool` | the model called a name that is not registered |
| `bad_arguments` | the arguments were not a JSON object, or did not bind to the hook's parameters (`TypeError`) |
| `tool_raised` | the hook raised anything else |

Do not signal failure by raising for control flow; return a string the model can
act on. Either way the turn continues — a failing tool never aborts a
conversation.

## The builtin tools

| Tool | Namespace | What it does |
|---|---|---|
| `get_current_time` | `get_current_time` | reports UTC and local time |
| `set_magic_number` | `magic` | stores a number in the shared `magic` namespace |
| `get_magic_number` | `magic` | reads it back |

The `magic` pair is the worked example of a shared namespace, and the pair used to
test that forking rewinds state. Before they declared `state_namespace="magic"`
they could not see each other's writes at all — which is the isolation default
working as intended.

## Checklist

- [ ] `description` reads well to a model, and says when *not* to use the tool.
- [ ] Parameters are described individually; they are all marked required.
- [ ] The hook returns a string that stands on its own — the model sees nothing else.
- [ ] State is flat, JSON-shaped where practical, and free of unpicklable values.
- [ ] If another tool needs the same memory, both declare the same `state_namespace`.
- [ ] Re-running the turn is safe, because a failed turn commits no state.
