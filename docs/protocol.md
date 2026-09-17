# Protocol

The server speaks **newline-delimited JSON over TCP**: one JSON object per line,
in both directions, UTF-8. Default `127.0.0.1:8765`, protocol version `2`.

A line longer than 8 MiB is rejected with `command_too_large` and the connection
closes. The server binds to loopback by default and has no authentication — it is
a local harness, not a service.

## Framing

```
client → server   {"command":"ping","rid":"1"}\n
server → client   {"event":"pong","rid":"1","seq":2,"ts":1789...}\n
```

Every event carries:

| Field | Meaning |
|---|---|
| `event` | the event name |
| `seq` | monotonic per connection, starting at 1 — a gap means a lost event |
| `ts` | `time.time()` when it was written |
| `agent_id` | on every event a block produced (turn, request, tool, state) |
| `rid` | on events belonging to a command |

`rid` is yours: send any JSON value on a command and it comes back on that
command's events, which is how you correlate replies when several turns stream at
once. A client that omits it gets no correlation.

## Command lifecycle

Every well-formed command produces, in order:

1. **`command_received`** — `rid`, `command`, `keys` (the fields you supplied),
   `target` (the block id you addressed, if any).
2. whatever the command does, possibly streaming many events;
3. **`command_finished`** — `rid`, `command`, `target`, `status`, `elapsed_ms`.

Two exceptions, both worth internalising:

- **`run` never sends a generic `command_finished`.** It is answered by the turn
  thread, which sends its own `command_finished` when the turn ends (`status` is
  `ok`, `error`, or `cancelled`). `run` sends `run_accepted` immediately so you
  know it started.
- **A rejected command sends `error` only, and no `command_finished`.** So a
  client that waits solely for `command_finished` hangs on a rejection. Wait for
  "`command_finished` or `error` for this `rid`", or add a timeout.

## Commands

| Command | Purpose |
|---|---|
| `create_agent` | create a root block (no prompt yet) |
| `fork` | append a block holding a user prompt |
| `run` | execute a block's single turn |
| `cancel` | ask a running turn to stop |
| `get_context` | the flattened message list of a chain |
| `get_state` | a block's rebuilt tool state |
| `set_state` | seed or drop one key of a block's state |
| `resolve_tool` | answer a local tool call a turn is waiting on |
| `list_agents` | every block, with links and flags |
| `destroy_agent` | drop a block and everything forked from it |
| `ping` | liveness, counters, store stats |

### `create_agent`

Creates a root. It holds no prompt until forked from, never runs a turn, and is
never dirty.

| Field | Notes |
|---|---|
| `id` | optional; auto-generated `agent-<12 hex>` when omitted |
| `endpoint` | optional; falls back to the server default |
| `key` | optional; falls back to the server default |
| `model` | optional; falls back to the server default |
| `tools` | optional list of names; omitted means every builtin |
| `local_tools` | optional definitions of tools the **client** runs (see below) |
| `timeout` | optional read timeout in seconds |
| `local_timeout` | optional seconds to wait for a local tool to be answered (default 120) |
| `include_usage` | optional bool; asks the provider for token accounting |
| `verbose` | optional bool; additionally emits raw `sse_chunk` events |

→ `agent_created` (`agent_id`, `parent: null`, `depth: 0`, `dirty`, `model`,
`endpoint`, `timeout`, `include_usage`, `verbose`, `tools`, `local_tools`,
`tool_schemas`).

Errors: `bad_id`, `missing_credentials`, `duplicate_agent`, `unknown_tool`,
`bad_tools`, `bad_field`.

### `fork`

Appends a block holding `prompt`, linked to `id`. The child inherits the parent's
endpoint, key, model, tools and options, and **starts dirty**.

| Field | Notes |
|---|---|
| `id` | **required** — the parent to fork from |
| `prompt` | **required** — non-empty string |
| `new_id` | optional id for the new block; supply it to avoid waiting for the reply |
| `model` `tools` `local_tools` `timeout` `local_timeout` `include_usage` `verbose` | optional overrides for the child only |

`tools` and `local_tools` are separate axes: supplying either replaces that half
and carries the other half over, so a fork can add a local tool without losing
the inherited builtins.

→ `agent_forked` (`agent_id`, `parent`, `depth`, `dirty`, `prompt`,
`prompt_chars`, `path` (ids root → child), `context_len`, `model`, `timeout`,
`include_usage`, `verbose`, `tools`, `local_tools`, `state_namespaces`).

Errors: `bad_id`, `bad_prompt`, `unknown_agent`, `parent_dirty`,
`duplicate_agent`, `unknown_tool`, `bad_field`.

Because the child's id is yours to choose, `fork` and `run` can be pipelined on
two consecutive lines without waiting for `agent_forked`.

### `run`

Executes the block's one turn. Events stream back on this connection; the turn
runs on its own thread, so the connection stays responsive.

| Field | Notes |
|---|---|
| `id` | **required** — a forked block that has not run yet |

→ `run_accepted` (`agent_id`, `depth`, `context_len`), then the turn events, then
`command_finished` with `command: "run"`, `status`, `error`, `error_type`,
`elapsed_ms`, `dirty`, `messages`, `context_len`, `state_committed` (the
namespaces this turn committed).

Errors: `bad_id`, `unknown_agent`, `root_agent` (a root has no prompt),
`agent_finished`, `agent_running`.

### `cancel`

| Field | Notes |
|---|---|
| `id` | **required** |

→ `cancel_result` (`cancelled`: whether a turn was running, `running`, `dirty`).
The turn notices on the next streamed chunk and ends with `status: "cancelled"`
and a `state_discarded` event. Cancelling an idle block is a no-op, not a
landmine for the next turn.

### `get_context`

| Field | Notes |
|---|---|
| `id` | **required** |

→ `context` (`depth`, `path`, `context` — the flattened messages that would be
sent, `context_len`, `local_len` — how many of them this block owns, `messages` —
this block's own list).

### `get_state` / `set_state`

`get_state`

| Field | Notes |
|---|---|
| `id` | **required** |
| `tool` | optional; a tool name *or* a namespace. Omitted means every namespace |

→ `state` (`tool` echoed, `keys` — the namespaces covered, `state` — namespace →
dict, plus `depth` and `path`).

`set_state` seeds a block's own deltas. Only a finished block may be written to:
a running one would race its turn, and a dirty one would need the seed folded
into the turn's diff at commit time. Seeding a root is the way to give a whole
subtree an initial environment.

| Field | Notes |
|---|---|
| `id` | **required** — a block that is neither running nor dirty |
| `tool` | **required** — resolved to the namespace it shares |
| `key` | **required** — the field *inside* that namespace |
| `value` | required unless `delete` is true |
| `delete` | optional bool; drops the key instead |

→ `state_seeded` (`tool`, `state_namespace`, `key`, `value`, `deleted`,
`replaced` — whether a value was there before, `keys` — the namespace's fields
now).

Errors: `agent_running`, `agent_dirty`, `bad_field`.

### `list_agents`

No fields. → `agents_listed` (`count`, `roots`, `dirty` — ids currently dirty,
`agents` — one entry per block).

Each entry: `agent_id`, `parent`, `depth`, `dirty`, `running`, `error`,
`prompt_chars`, `prompt_preview`, `text_chars`, `local_len`, `context_len`,
`model`, `tools`, `local_tools`, `state_namespaces`, `waiting_on` (local calls
this block is parked on), `include_usage`, `verbose`, `created_at`, `age_ms`.

### `destroy_agent`

| Field | Notes |
|---|---|
| `id` | **required** |

Drops the block **and every block forked from it** — a block is only reachable
through its parent, so its whole subtree goes with it. Any turn inside is
cancelled. → `agent_destroyed` (`agent_id`, `dropped` — every id removed, `count`,
`cancelled` — ids whose turn was running, `age_ms`, `remaining`).

Errors: `bad_id`, `unknown_agent`.

### `resolve_tool`

Answers a local tool call that a turn is parked on. **Deliberately allowed on a
running block**, unlike `set_state`: the block being parked is the whole point.

| Field | Notes |
|---|---|
| `id` | **required** — the block whose turn is waiting, from the event |
| `call_id` | **required** — from the `local_tool_called` event |
| `result` | the string handed back to the model; a non-string is JSON-encoded |
| `error` | optional; marks the call failed and is reported to the model as such |

→ `local_tool_answered` (`call_id`, `ok`, `result_chars`), and the parked turn
resumes.

Errors: `bad_id`, `unknown_agent`, `bad_field`, `unknown_call` — nothing is
waiting on that id any more (it timed out, was cancelled, or never existed), and
`detail.pending` lists what is still outstanding.

### `ping`

| Field | Notes |
|---|---|
| `echo` | optional; echoed back |

→ `pong` (`echo`, `server_time`, `uptime_ms`, `agents`, `dirty`, `running`,
`threads`, `store` — `{path, blocks, trees, bytes, max_bytes}` or null).

## Events

### Session

| Event | Fields | When |
|---|---|---|
| `session_hello` | `protocol`, `server`, `peer`, `started_at`, `defaults` (`endpoint`, `model`, `has_key`), `commands` (self-describing: name, summary, fields), `tools` (name, description, `state_namespace`, params), `store` (`path`, `blocks`, `trees`, `bytes`, `max_bytes`, `warnings`) or null | first thing after connecting |
| `session_closing` | `commands`, `events_sent` | the connection is going away |

`hooks`' `defaults` never include the key — only `has_key`. `store.warnings`
reports what restore could not resolve (a tool name that no longer exists, an
unreadable row).

### Lifecycle

| Event | Fields |
|---|---|
| `command_received` | `rid`, `command`, `keys`, `target` |
| `command_finished` | `rid`, `command`, `target`, `status`, `elapsed_ms` — or, for `run`: `agent_id`, `status`, `error`, `error_type`, `dirty`, `messages`, `context_len`, `state_committed` |
| `run_accepted` | `rid`, `agent_id`, `depth`, `context_len` |
| `cancel_result` | `rid`, `agent_id`, `cancelled`, `running`, `dirty` |
| `agent_created` | see the command above |
| `agent_forked` | see the command above |
| `agent_destroyed` | `rid`, `agent_id`, `dropped`, `count`, `cancelled`, `age_ms`, `remaining` |
| `agents_listed` | see the command above |
| `context` | see the command above |
| `state` | see the command above |
| `state_seeded` | see the command above |
| `local_tool_answered` | `rid`, `agent_id`, `call_id`, `ok`, `result_chars` |
| `pong` | see the command above |
| `error` | `code`, `message`, `rid`, `command`, plus a case-specific `detail`; `internal_error` adds `traceback`, `bad_json` adds `line`, `command_too_large` adds `bytes`, `unknown_command` adds `commands` |

### Turn

| Event | Fields |
|---|---|
| `turn_started` | `prompt`, `prompt_chars`, `depth`, `path`, `model`, `tools`, `context_len`, `include_usage`, `verbose` |
| `turn_finished` | `text`, `text_chars`, `rounds`, `tool_calls`, `elapsed_ms`, `messages`, `context_len`, `dirty: false` |
| `turn_failed` | `error`, `error_type`, `text` (partial), `elapsed_ms`, `context_len`, `dirty: false` |
| `turn_cancelled` | `error`, `text` (partial), `elapsed_ms`, `context_len`, `dirty: false` |

A turn is one or more rounds: each round is one request, and a round that
requests tools is followed by another round until the model answers with text.

### Request (`round` is 1-based within the turn)

| Event | Fields |
|---|---|
| `request_started` | `round`, `url`, `model`, `timeout`, `request_bytes`, `messages`, `tools`, `depth`, `include_usage` |
| `request_payload` | `round`, `messages` — a per-message summary: `role`, `chars`, and `tool_call_id` / `tool_calls` when present |
| `response_received` | `round`, `status`, `reason`, `content_type`, `elapsed_ms` |
| `request_finished` | `round`, `status`, `chunks`, `payload_chars`, `finish_reason`, `first_chunk_ms`, `elapsed_ms`, `cancelled` |
| `request_failed` | `round`, `error`, `error_type`, `elapsed_ms` (plus `chunks` when it failed mid-stream) |
| `sse_chunk` | `round`, `index`, `chunk` — the raw provider chunk; only with `verbose` |
| `sse_unparsed` | `round`, `line` — a stream line that was not JSON |

### Stream

| Event | Fields |
|---|---|
| `content_delta` | `round`, `index`, `text`, `chars`, `elapsed_ms` — the reply, as it arrives |
| `reasoning_delta` | `round`, `text`, `chars` — the model's reasoning, kept out of the reply |
| `usage` | `round`, `usage` — the provider's token accounting, when it sends any |
| `assistant_message` | `round`, `content`, `content_chars`, `tool_calls` (`call_id`, `name`, `arguments`) — the round's message, assembled |
| `history_appended` | `source` (`fork` / `assistant` / `tool`), `role`, `chars`, `preview` (first 200), `tool_call_id`, `tool_calls`, `messages`, `context_len` |

### Tool

| Event | Fields |
|---|---|
| `tool_call_requested` | `round`, `call_id`, `name`, `raw_arguments`, `known` |
| `tool_call_started` | `round`, `call_id`, `name` |
| `tool_call_finished` | `round`, `call_id`, `name`, `ok`, `error`, `result`, `result_chars`, `elapsed_ms` |

`error` is a code, not a message: `unknown_tool`, `bad_arguments`, `tool_raised`,
`timeout`, or `no_hook`. A tool that fails still produces a result, which is fed
back to the model so it can correct itself.

Calls to a **local tool** — one this client declared — announce themselves
instead of running:

| Event | Fields |
|---|---|
| `local_tool_called` | `round`, `call_id`, `name`, `arguments`, `raw_arguments`, `timeout_ms` — the turn is now parked on this |
| `local_tool_resolved` | `call_id`, `name`, `ok`, `error`, `result`, `result_chars`, `waited_ms` |
| `local_tool_unresolved` | `call_id`, `name`, `reason` (`timeout` / `cancelled`), `waited_ms` |

### State

| Event | Fields |
|---|---|
| `state_loaded` | `round`, `call_id`, `tool`, `state_namespace`, `keys` (fields present), `inherited` (how many came from ancestors) |
| `state_delta` | `tool`, `state_namespace`, `changed`, `removed`, `keys` — committed at the end of a successful turn |
| `state_discarded` | `tool`, `state_namespace`, `changed`, `removed`, `reason` — `failed`, `cancelled`, or `abandoned` |

### Store

| Event | Fields |
|---|---|
| `evicted` | `agent_id` (the subtree's root), `dropped`, `nodes`, `newest`, `bytes`, `max_bytes` |
| `persist_warning` | `agent_id`, `dropped_tools`, `message` — state that could not be stored |

## Local tools

A block may declare tools the **client** runs. Only a schema travels — name,
description, parameters — because the implementation is on the client's side, and
the model sees it in the same `tools` array as the builtins.

```jsonc
{"command":"create_agent","id":"root","local_tools":[
  {"name":"ask_operator","description":"Ask the human operator.",
   "params":[{"name":"question","type":"string","description":"what to ask"}]}
]}
```

They inherit through `fork` (pass `local_tools` to replace the set), are reported
as `local_tools` by `agent_created`, `agent_forked` and `list_agents`, and survive
a restart because the definitions are stored whole.

When the model calls one, the turn parks and the client is asked:

```
← {"event":"tool_call_requested","name":"ask_operator","raw_arguments":"{\"question\":\"今天午饭吃什么？\"}"}
← {"event":"local_tool_called","agent_id":"a1","call_id":"call_1","name":"ask_operator",
   "arguments":{"question":"今天午饭吃什么？"},"timeout_ms":120000}
→ {"command":"resolve_tool","id":"a1","call_id":"call_1","result":"红烧肉"}
← {"event":"local_tool_answered","call_id":"call_1","ok":true,"result_chars":9}
← {"event":"local_tool_resolved","call_id":"call_1","name":"ask_operator","ok":true,"waited_ms":401}
← {"event":"tool_call_finished","name":"ask_operator","ok":true,"result":"红烧肉"}
```

Things to know:

- **Only that one turn is parked.** The wait holds no lock and no database
  transaction, so other connections keep being served normally, writes included.
- **The answer is addressed by `agent_id` and `call_id`**, both from the
  `local_tool_called` event — the block owns the pending call.
- **Local tools have no server-side state.** `ToolContext.state` lives on the
  server, so such a tool keeps its own memory; forking rewinds builtin state but
  not a client's private memory.
- **`cancel` releases a parked turn**, reporting `reason: "cancelled"`.

## Error codes

| Code | Meaning |
|---|---|
| `bad_json` | the line was not JSON |
| `bad_command` | not a JSON object, or `command` is not a string |
| `command_too_large` | over 8 MiB |
| `unknown_command` | no such command; the reply lists the valid ones |
| `bad_id` | `id` / `new_id` missing or not a non-empty string |
| `bad_prompt` | `prompt` missing or empty |
| `bad_field` | a field had the wrong type or value |
| `bad_tools` | `tools` / `local_tools` was malformed, or a name is declared twice |
| `unknown_tool` | a name is not in the catalogue |
| `unknown_call` | `resolve_tool` for a call nothing is waiting on |
| `unknown_agent` | no such block; `detail.agents` lists what exists |
| `duplicate_agent` | that id is taken |
| `missing_credentials` | no endpoint/key on the command and no server default |
| `parent_dirty` | the block being forked from has not finished its turn |
| `root_agent` | `run` on a root, which has no prompt |
| `agent_finished` | that block already ran its turn |
| `agent_running` | that block is mid-turn |
| `agent_dirty` | `set_state` on a block that has not run yet |
| `internal_error` | a bug; carries `traceback` |

## Semantics to rely on

**A block runs once.** `fork` → `run`, and never again. Rewinding is forking
from an earlier block, not re-running one.

**Forking rewinds tool state too.** A child sees its ancestors' deltas only, so
a branch cannot see what a sibling's turn committed.

**A failed turn is transparent for state.** It commits no deltas, keeping the
environment consistent with what a fork from it would produce. Its *messages* are
kept as they stand, partial replies included — a log can be partial, a value
cannot.

**State is addressed by namespace.** `get_state`/`set_state` accept a tool name
and resolve it to the namespace that tool shares; an unknown name is treated as a
namespace, so state stays reachable after a tool is removed.

**The database budget is soft.** A subtree with a turn in flight is skipped, so
the file can exceed `--max-db-bytes` until that turn ends.

**Local tools park one turn and nothing else.** While a client is deciding, no
lock and no database transaction is held, so every connection stays fully
serviceable — writes included. A call with no answer inside `local_timeout` does
not fail the turn; the model is told the tool produced an error and can react.

**Durability precedes the completion event.** When you see a turn's
`command_finished`, its result is already on disk if the server was started with
`--db`.

## Example

```jsonc
{"command":"create_agent","rid":"1","id":"root"}
// ← session_hello, command_received, agent_created, command_finished

{"command":"fork","rid":"2","id":"root","prompt":"用工具查现在时间","new_id":"a1"}
// ← command_received, agent_forked, command_finished

{"command":"run","rid":"3","id":"a1"}
// ← run_accepted, turn_started, request_started, request_payload,
//   response_received, content_delta ×n, tool_call_requested,
//   state_loaded, tool_call_started, tool_call_finished,
//   history_appended, request_finished, assistant_message,
//   request_started (round 2), content_delta ×n, usage, request_finished,
//   turn_finished, command_finished

{"command":"get_state","rid":"4","id":"a1"}
// ← command_received, state, command_finished
```
