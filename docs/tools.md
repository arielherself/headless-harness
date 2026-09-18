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
| `hook` | `hook(context, **arguments) -> str`, or a `ToolResult` / `ToolCall` when it returns images or pipes (see below) |
| `state_namespace` | where the tool's state lives; empty means its own name (see below) |

The hook is always called with the context first and the model's arguments as
keywords, so a tool with no parameters is `def hook(context: ToolContext) -> str`.

### Returning images

A hook may return a plain string, as above, or a `ToolResult` when the model
should see images too:

```python
import base64

from tools import ToolResult


def screenshot_executor(context: ToolContext, url: str) -> ToolResult:
    png = take_screenshot(url)  # bytes, whatever your tool produces
    encoded = base64.b64encode(png).decode("ascii")
    return ToolResult(f"Captured {url}", images=[f"data:image/png;base64,{encoded}"])
```

`images` takes the same shapes `fork` accepts:
[`protocol.md#images`](protocol.md#images) — an `http(s)` URL or
`data:image/...` URI string, or a `{url, detail}` mapping. The text is what
events, failure summaries and rollback reports carry; the image bytes only ever
go into the model's tool message. A malformed `ToolResult` (a bad image, a local
path, non-string text) comes back to the model as a `bad_result` error and the
turn continues.

### Piping into another tool

A hook's third answer is a `ToolCall`: run this other tool first, and only tell
the model what *that* one says.

```python
from tools import ToolCall, ToolResult


def fetch_executor(context: ToolContext, url: str) -> ToolResult:
    path, data = download(url)
    return ToolResult(
        f"downloaded {len(data)} bytes",          # for the record, not the model
        call=ToolCall("write_file", {
            "sandbox_id": context.state["sandbox"],
            "path": path,
            "data": data,                          # bytes, never quoted by a model
        }),
    )


def write_file_executor(context: ToolContext, sandbox_id: str, path: str, data: bytes) -> str:
    put_file(sandbox_id, path, data)              # what the previous call piped over
    return f"wrote {path} ({len(data)} bytes)"    # the only text the model sees
```

Returning a `ToolCall` on its own is the same thing with no note. The harness runs
the named tool — server-side or, if it has no hook, by asking the client — and
follows whatever *it* returns, until a call answers with text. A pipe may mix the
two freely.

**The model sees the ends, never the middle.** The tool message becomes the chain
of names plus the last result:

```
[tool pipe] fetch -> write_file
wrote /workspace/photo.jpg (184320 bytes)
```

The intermediate arguments and results are recorded on the block (`get_context`
returns them as `pipe_traces`) and reported through `pipe_step_started` /
`pipe_step_finished` events, but they are never sent to the provider — which is
the whole point, since that is where a base64 file or a megabyte of HTML would
otherwise cost tokens. Long values are stored bounded (a prefix, the true length
and a `sha256` prefix), so inspecting a pipe never bloats the database either.

What a piping hook needs to know:

- **`context.tools` is the registry** — every tool the block offers, by name,
  with the params each expects. Check it before piping: a name that is not there
  still runs, but comes back as `unknown tool 'x'` in the model's result. Editing
  the mapping changes nothing.
- **A pipe is bounded to 16 calls** (`MAX_PIPE_DEPTH`). Hitting the limit is not a
  turn failure: the model gets `Error: tool pipe from 'x' stopped after 16 steps`
  and `error: "pipe_depth"` on `tool_call_finished`.
- **Only the last call may return images.** A `ToolResult` that carries both a
  `call` and `images` is a `bad_result` — the earlier images would have nowhere to
  go. Text on a piping `ToolResult` is for the trace only, so it can be as
  detailed as you like about what the step did.
- **Every step is a call of its own.** It is rolled back with the turn, newest
  first, and `context.result` in a rollback hook is that step's own text — which
  for a piping call is its note, not the final answer.
- **Piped arguments are arbitrary Python values.** They only have to survive
  being passed to the next hook; the recorded form is bounded and JSON-shaped.

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
| `tools` | every tool this block offers, by name — the registry a piping hook inspects |

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

**State is invisible to the model.** The model only sees your hook's result —
its text, plus any images. If it should know something happened, say so in the
result — state will not tell it.

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
| `bad_result` | the hook returned something that was not a string, `ToolResult` or `ToolCall`; a `ToolResult` whose images were malformed; or a pipe call with no name, non-object arguments, or images on a step that piped |
| `pipe_depth` | the pipe reached `MAX_PIPE_DEPTH` calls and was cut off |

Do not signal failure by raising for control flow; return a string the model can
act on. Either way the turn continues — a failing tool never aborts a
conversation.

## Rolling back

Discarding a turn's state delta undoes the state. Anything a tool did *outside*
the state — installed a package, wrote a file, called an API — needs its own undo,
so a tool may declare one:

```python
def install_executor(context: ToolContext, package: str) -> str:
    run(f"apt install -y {package}")
    return f"installed {package}"


def install_rollback(context: ToolContext, package: str) -> str:
    # `context.result` is the text the call returned, in case the undo needs it
    run(f"apt remove -y {package}")
    return f"removed {package}"


install_tool = ToolEntry(
    name="install",
    description="Install a system package.",
    params=[ToolParam(name="package", type="string", description="package name")],
    hook=install_executor,
    rollback=install_rollback,
    external_effects=True,      # it really does install things
)
```

**`external_effects` is a separate declaration from `rollback`, and both are
needed for the note to be honest.** State rolls back on its own, so the harness
has no way to know whether a tool touched anything else; `external_effects` is how
you tell it. A failure note then partitions the calls that declared it into
`Undone`, `Could not be undone`, and — for a tool that declared effects but has no
undo — `May still be in effect`. A tool that declares nothing is simply not
mentioned, so:

- Declare `external_effects` on anything that reaches outside the state, **even if
  it cannot be undone.** "I sent an email and there is no taking it back" is a
  legitimate pair of declarations, and the note should say so rather than stay
  silent.
- Forgetting it is the failure mode to watch for: a tool that moved the world
  without declaring it leaves the next model assuming a clean environment. The
  harness cannot detect this for you.

When a turn fails, is cancelled, or is abandoned, the rollback of **every call the
turn made** runs, newest first — so a tool called three times is undone three
times, each with its own arguments and result. A successful turn rolls back
nothing.

Rules the harness enforces so your hook can assume them:

- **A failing rollback is contained.** If yours raises, it is reported as
  `rollback_finished` with `ok: false` and skipped; the other rollbacks still run,
  and the turn keeps its original failure.
- **Calls that never reached a tool are not rolled back.** An unknown tool, or
  arguments that did not bind (a `TypeError` before your hook body ran), did
  nothing and is not in the list.
- **The state you see is a copy.** A rollback gets a deep copy of the turn's live
  state, so nothing it does there can muddy the report of what was discarded —
  and since a failing turn commits nothing, state edits would have been thrown
  away anyway.
- **A cancelled turn does not wait for a client.** For a local tool's undo the
  request is sent but not waited on, so cancelling stays quick.

Builtin state needs no rollback hook at all: a failing turn commits no deltas, so
the state is already back where it was.

After the rollbacks, the turn appends one `[harness]` message to its block saying
what failed and what was undone — summarised by a model from the turn's own
messages, with the raw error kept verbatim (see
[`protocol.md`](protocol.md#failed-turns-leave-a-note)). You get that for free;
the only thing worth knowing as a tool author is that a rollback failing is
reported there as `Could not be undone`, which is the cue for whoever reads the
chain next that an effect is still standing.

## Tools that run on the client

Not every tool belongs on the server. A tool with no `hook` is a **local
tool**: the server offers it to the model but cannot run it, so it parks the turn
and asks the client for the result.

Local tools are declared per block over the protocol rather than in this module,
because their implementation is somewhere else entirely:

```jsonc
{"command":"create_agent","id":"root","local_tools":[
  {"name":"ask_operator","description":"Ask the human operator.",
   "params":[{"name":"question","type":"string","description":"what to ask"}]}
]}
```

The model sees them exactly like a builtin. When one is called the server emits
`local_tool_called` and waits for a `resolve_tool` command with the result, which
becomes the tool message on the next request. In the library, that wait is
`HHAgent.resolve_local_call(call_id, result, error=None)` — with `images=` for
images, and `call=ToolCall(...)` to pipe into another tool instead of ending the
call — called from whatever thread can answer.

`ToolEntry(name=..., description=..., params=..., hook=None)` builds one in code;
`tool.is_local` is true for it. The rest of this document applies as usual, with
two exceptions:

- **No state.** `ToolContext` exists on the server, so a local tool cannot use
  `state`. Keep whatever memory it needs on your own side — and note that forking
  rewinds server-side state but cannot rewind yours.
- **A timeout is not a failure.** If no answer arrives within the block's
  `local_timeout`, the tool result becomes an explanatory error string and the
  turn continues, so the model can react as it would to any other failing tool.

Nothing else is blocked while the client decides: no lock, no database
transaction. See [`design.md`](design.md) for why that is guaranteed rather than
lucky.

## The builtin tools

| Tool | Namespace | What it does |
|---|---|---|
| `get_system_info` | `get_system_info` | reports the host: OS, hostname, Python, CPUs, model, protocol version |
| `get_current_time` | `get_current_time` | reports UTC and local time |
| `set_magic_number` | `magic` | stores a number in the shared `magic` namespace |
| `get_magic_number` | `magic` | reads it back |
| `web_fetch` | `web_fetch` | reads a URL through Jina's reader and returns Jina's Markdown of the page |
| `web_search` | `web_search` | searches through Exa and returns short excerpts (title, URL, highlights) of the top results |
| `nix_spawn_sandbox` | `nix_spawn_sandbox` | creates the one sandbox shape and returns its id |
| `nix_sandbox_status` | `nix_sandbox_status` | reports whether an id is still live, with its age, last use and packages |
| `nix_add_dependency` | `nix_add_dependency` | installs one Nix package for the next command |
| `nix_remove_dependency` | `nix_remove_dependency` | removes a package; bash and coreutils cannot go |
| `nix_exec` | `nix_exec` | runs one shell line; the timeout is required and at most 600s |
| `nix_add_file` | `nix_add_file` | writes one base64 file into the writable `/workspace` |
| `nix_destroy_sandbox` | `nix_destroy_sandbox` | stops it and deletes its files now |

The `magic` pair is the worked example of a shared namespace, and the pair used to
test that forking rewinds state. Before they declared `state_namespace="magic"`
they could not see each other's writes at all — which is the isolation default
working as intended.

`web_fetch` reads a page through Jina's reader (`r.jina.ai`), so the URL leaves
the host and what comes back is Jina's extraction of the page as Markdown — the
main content rendered and condensed, not the raw HTML. It is a read: the request
leaves nothing behind that a later turn would have to trip over, so it declares no
external effects and a failed turn says nothing about it. Set `JINA_API_KEY` in the
server's environment to authenticate the call; without it Jina rate-limits the
caller by IP. A page longer than `WEB_FETCH_MAX_CHARS` (20 000 characters) comes
back truncated with a note saying how much was cut.

`web_search` is its other half: it asks Exa for the best pages matching a query and
returns their titles, URLs and excerpts, which is what you want before you have a
URL at all. It needs no key either, Exa rate-limits the free endpoint by IP, and —
like `web_fetch` — it only reads, so it declares no external effects.

### The sandbox tools

The `nix_*` tools are the harness's side of the `sandbox/` package: they run
untrusted commands in a bubblewrap sandbox built from a Nix environment. Their
implementation lives in `src/sandbox_tools.py`, and they behave differently from
the tools above in four ways worth knowing.

**There is one fixed configuration, and spawn takes no parameters.** Every
sandbox gets 256M of memory, 512M of disk, 256 pids, one CPU, the host network,
and a writable `/workspace` that is also the working directory. The model chooses
what runs in a sandbox, never how much of the machine it gets. A new sandbox
starts with bash and coreutils; `nix_add_dependency` installs more Nix packages,
one per call, into the environment the next `nix_exec` sees.

**Ids are memory, not state.** `nix_spawn_sandbox` returns an id like
`sbx-3f2a9c1b7d0e`, and every other `nix_*` tool takes it. The registry is
process-wide and in-memory, so nothing is written to a block's state: forking a
block rewinds tool memory but cannot resurrect a sandbox, and a restarted server
has forgotten every id. `nix_sandbox_status` is how the model tells whether a
sandbox it created earlier still exists — it reports age, last use and packages,
or says the sandbox was released.

**Idle sandboxes are released, and there is a cap.** A daemon thread sweeps every
20 minutes and destroys every sandbox that no tool call has named for 10 minutes;
any call about a sandbox — even one that fails — restarts its clock, and a call
that is still running protects it for its whole duration. At most 10
sandboxes exist at once, and one more `nix_spawn_sandbox` is refused rather than
evicting anybody's work. `nix_destroy_sandbox` ends one immediately.

**`nix_exec` always takes a timeout, and `nix_add_file` takes bytes as base64.**
The timeout is required and must be between 1 and 600 seconds; a command that
outlives it is killed and the partial output is returned with exit code 124. Each
command runs in fresh namespaces, so background processes do not survive from one
call to the next and only files under `/workspace` persist — which is where
`nix_add_file` can put a file, handed over as a base64 string because a tool call
is JSON. A file whose content starts with `#!` is made executable.

All of them but `nix_sandbox_status` declare `external_effects`: they create,
destroy and change things outside the tool state, so a failed turn's note lists
them rather than letting the next model assume the environment is clean.

## Checklist

- [ ] `description` reads well to a model, and says when *not* to use the tool.
- [ ] Parameters are described individually; they are all marked required.
- [ ] The hook returns a string that stands on its own — unless it pipes, in which case only the last call's result is read.
- [ ] A pipe is bounded: check `context.tools` before naming a target, and keep the chain short.
- [ ] State is flat, JSON-shaped where practical, and free of unpicklable values.
- [ ] If another tool needs the same memory, both declare the same `state_namespace`.
- [ ] Re-running the turn is safe, because a failed turn commits no state.
