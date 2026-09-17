# headless-harness

A conversation agent whose history is a **chain of blocks**, exposed over a local
TCP port as line-delimited JSON, with SQLite persistence and forking as the
primary way to branch or rewind.

The idea is simple: a message is never appended to a growing transcript. Each
prompt creates a new **block** holding just that prompt and whatever the model
produced in reply, linking to its parent. The request context is rebuilt by
walking to the root, so a block stores only its own delta and nothing is ever
copied. Forking a block rewinds the conversation — and every tool's memory —
to exactly that point, because tool state is stored the same way.

## Quick start

```bash
# talk to the provider directly
python src/server.py                    # 127.0.0.1:8765, persists to ./harness.db

# or drive it with the bundled interactive client (it starts a server if none is up)
python test.py
```

`test.py` is the interactive client used for manual testing; it also holds the
provider credentials the server reads on startup, which is why it is gitignored.

A minimal session, over any TCP client:

```jsonc
{"command":"create_agent","rid":"1","id":"root"}          // a root block, no prompt yet
{"command":"fork","rid":"2","id":"root","prompt":"你好","new_id":"a"}
{"command":"run","rid":"3","id":"a"}                      // executes a's one turn
```

Events come back on the same connection, one JSON object per line:

```jsonc
{"event":"content_delta","agent_id":"a","text":"你","seq":9,...}
{"event":"turn_finished","agent_id":"a","text":"你好！","rounds":1,...}
{"event":"command_finished","rid":"3","command":"run","status":"ok",...}
```

## Running the server

```bash
python src/server.py [--host H] [--port P] [--endpoint URL] [--key K]
                     [--model M] [--db PATH] [--max-db-bytes N]
```

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | bind address; loopback only by choice, there is no auth |
| `--port` | `8765` | listen port |
| `--endpoint` | — | default provider base URL for new blocks |
| `--key` | — | default provider key for new blocks; never written to disk |
| `--model` | `deepseek/deepseek-v4.1-flash` | default model |
| `--db` | `<project root>/harness.db` | SQLite file; blocks are restored from it at startup |
| `--max-db-bytes` | `67108864` (64 MiB) | evict whole subtrees, oldest first, once the file exceeds this; `0` disables the limit |

If `--endpoint` or `--key` is omitted the server falls back to a sibling
`test.py` that defines them — which is why that file is gitignored. Without
credentials it still starts, and agents must then be created with an explicit
`endpoint` and `key`.

The bundled client has its own flags: `--host`, `--port`, `--agent ID` (resume at
an existing block), `--db` (forwarded when it spawns a server), `--no-spawn`,
`--think`, `--raw`, `--no-color`.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/protocol.md`](docs/protocol.md) | Every command, every event, and the wire-level rules |
| [`docs/design.md`](docs/design.md) | Why the chain model, fork semantics, atomic turns, state namespaces |
| [`docs/tools.md`](docs/tools.md) | Writing a tool: `ToolContext`, persistent state, constraints |
| [`TODO.md`](TODO.md) | Known open work |

## Layout

```
src/agent.py    HHAgent — one block of the chain; provider calls; state deltas
src/server.py   HHServer — TCP listener, registry, JSONL protocol, eviction
src/store.py    HHStore  — SQLite mirror, size budget, subtree eviction
src/tools.py    ToolEntry / ToolContext and the builtin tools
src/main.py     empty placeholder
test.py         interactive client (gitignored: it holds the API key)
docs/           this documentation
```

## The moving parts

**Blocks form a chain.** `HHAgent.root(...)` makes an empty root;
`block.fork(prompt)` makes a child holding that prompt. A block runs its single
turn exactly once: it is `dirty` from `fork` until the turn ends, and forking
from a dirty block is refused because its context is still growing. Rewinding is
just forking from an earlier block again — the blocks beyond it stay untouched.

**Tool state is incremental too.** A tool is handed a mutable `dict` already
overlaid with its ancestors' deltas; whatever it changed when a *successful* turn
ends is frozen into that block. Forking therefore rewinds tool memory as well. A
tool's state lives in the namespace named by its `state_namespace` (its own name
by default), so tools are isolated unless they explicitly share one.

**Everything is observable.** A turn emits ~20 events — the request that went
out, every streamed text and reasoning delta, each tool call with its arguments
and result, state commits and discards, timings and token usage. Nothing has to
be guessed from the final text.

**Some tools can live on the client.** A block may declare `local_tools`: tools
it does not implement, only describes. The model sees them like any other, and
when one is called the server emits `local_tool_called` and parks that turn until
the client answers with `resolve_tool`. The wait holds no lock and no database
transaction, so the rest of the server keeps working while a client decides.

**And a turn that fails can undo itself.** A tool may declare a `rollback` hook —
or, if it runs on the client, promise that the client has one — and when a turn
does not commit, every call it made is offered an undo, newest first. A rollback
that fails is reported and skipped, so the others still run.

## Requirements

Python ≥ 3.10 and `requests` (the provider is called through it). SQLite comes
from the standard library. `uv` manages the environment:

```bash
uv sync
```
