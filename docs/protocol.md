# Protocol

The server speaks **newline-delimited JSON over TCP**: one JSON object per line,
in both directions, UTF-8. Default `127.0.0.1:8765`, protocol version `4`.

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
| `max_tokens` | optional output-token cap for every turn request; defaults to the default model's own maximum (393216), and `0` sends no cap (the failure-summary request is separate and uncapped) |
| `summary_model` | optional model for failure summaries; defaults to the block's own |
| `include_usage` | optional bool; asks the provider for token accounting |
| `verbose` | optional bool; additionally emits raw `sse_chunk` events |

→ `agent_created` (`agent_id`, `parent: null`, `depth: 0`, `dirty`, `model`,
`endpoint`, `timeout`, `max_tokens`, `include_usage`, `verbose`, `tools`,
`local_tools`, `tool_schemas`).

Errors: `bad_id`, `missing_credentials`, `duplicate_agent`, `unknown_tool`,
`bad_tools`, `bad_field`.

### `fork`

Appends a block holding `prompt`, linked to `id`. The child inherits the parent's
endpoint, key, model, tools and options, and **starts dirty**.

| Field | Notes |
|---|---|
| `id` | **required** — the parent to fork from |
| `prompt` | **required** — non-empty string |
| `images` | optional — images sent with the prompt (see [Images](#images)) |
| `new_id` | optional id for the new block; supply it to avoid waiting for the reply |
| `model` `tools` `local_tools` `timeout` `local_timeout` `max_tokens` `summary_model` `include_usage` `verbose` | optional overrides for the child only |

`tools` and `local_tools` are separate axes: supplying either replaces that half
and carries the other half over, so a fork can add a local tool without losing
the inherited builtins.

→ `agent_forked` (`agent_id`, `parent`, `depth`, `dirty`, `prompt`,
`prompt_chars`, `image_count`, `path` (ids root → child), `context_len`, `model`,
`timeout`, `max_tokens`, `include_usage`, `verbose`, `tools`, `local_tools`,
`state_namespaces`).

Errors: `bad_id`, `bad_prompt`, `bad_image`, `unknown_agent`, `parent_dirty`,
`duplicate_agent`, `unknown_tool`, `bad_field`.

Because the child's id is yours to choose, `fork` and `run` can be pipelined on
two consecutive lines without waiting for `agent_forked`.

#### Images

`fork` may carry images alongside the prompt. `images` is a list whose entries
are either a string — an `http(s)` URL or a `data:image/...` URI — or an object
with a `url` and an optional `detail` (`auto`, `low` or `high`):

```jsonc
{"command":"fork","rid":"3","id":"root","new_id":"a1","prompt":"what is this?",
 "images":["data:image/png;base64,iVBOR…",
           {"url":"https://example.test/cat.webp","detail":"low"}]}
```

The block's user message then becomes an OpenAI-style content-part list: a
`text` part first, then one `image_url` part per image. Without `images` the
message stays the plain string it has always been, and `prompt` is the text in
both cases:

```jsonc
{"role":"user","content":[{"type":"text","text":"what is this?"},
                          {"type":"image_url","image_url":{"url":"data:…"}}]}
```

A local path is refused (`bad_image`), as is any other scheme — `file:`
included. The server never reads a file on a client's behalf, and it will not
hand a provider a path to open either. A client that has a local file reads it
itself and sends a `data:image/...` URI, which is what `test.py` does.

Images ride in the same single JSON line as everything else, so a `data:` URI
counts against the 8 MiB limit (base64 adds about a third). `prompt` stays
required and non-empty: images ride along with text, they do not replace it.
`image_count` reports how many parts a block holds without echoing the bytes.

Tool results can carry images too: a local tool answers with `images` on
`resolve_tool`, and a server-side hook returns a `ToolResult` (see
[`tools.md`](tools.md)). The same shapes apply, and events report them by
`image_count` as well.

All of this arrived in protocol `3`: a `2` server does not know the `images`
fields and would ignore them, so check `session_hello.protocol` before sending
images to an unknown server.

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
this block's own list, `pipe_traces` — one record per tool pipe this block's own
turn ran, see [Tool pipes](#tool-pipes)).

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

Each entry: `agent_id`, `parent`, `depth`, `dirty`, `running`, `outcome`
(`ok` / `failed` / `cancelled` / `abandoned`, null before the turn has run),
`error`, `prompt_chars`, `prompt_preview`, `image_count`, `text_chars`,
`local_len`, `context_len`, `model`, `summary_model`, `tools`, `local_tools`,
`state_namespaces`, `waiting_on` (local calls this block is parked on),
`pipe_traces` (how many this block's own turn ran), `max_tokens`, `include_usage`,
`verbose`, `created_at`, `age_ms`.

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
| `result` | the text handed back to the model; a non-string is JSON-encoded; optional when `images` or `call` is given |
| `images` | optional list of http(s) URLs or data:image/... URIs the tool returned (see [Images](#images)) |
| `error` | optional; marks the call failed and is reported to the model as such |
| `call` | optional `{name, arguments}` for the tool to run next, continuing a [pipe](#tool-pipes); cannot be combined with `error` or `images` |

→ `local_tool_answered` (`call_id`, `ok`, `result_chars`, `image_count`, `next` —
the tool a piped `call` names, null otherwise), and the parked turn resumes. The
images become part of the tool message the model sees, the same way a fork's
images become part of the user message; an answer that carries `error` stays text
and its images, if any, are dropped.

Errors: `bad_id`, `unknown_agent`, `bad_field`, `bad_call` — the `call` was not
`{name, arguments}` with a non-empty name, or it was combined with `error` or
`images` — and `unknown_call`: nothing is waiting on that id any more (it timed
out, was cancelled, or never existed), with `detail.pending` listing what is still
outstanding. A rejected answer leaves the call parked, so it can be retried.

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
| `local_tool_answered` | `rid`, `agent_id`, `call_id`, `ok`, `result_chars`, `image_count`, `next` |
| `pong` | see the command above |
| `error` | `code`, `message`, `rid`, `command`, plus a case-specific `detail`; `internal_error` adds `traceback`, `bad_json` adds `line`, `command_too_large` adds `bytes`, `unknown_command` adds `commands` |

### Turn

| Event | Fields |
|---|---|
| `turn_started` | `prompt`, `prompt_chars`, `image_count`, `depth`, `path`, `model`, `tools`, `context_len`, `max_tokens`, `include_usage`, `verbose` |
| `turn_finished` | `text`, `text_chars`, `rounds`, `tool_calls` (the model's own), `pipe_steps` (calls tools added by piping), `elapsed_ms`, `messages`, `context_len`, `dirty: false` |
| `turn_failed` | `error`, `error_type`, `text` (partial), `elapsed_ms`, `context_len`, `dirty: false` |
| `turn_cancelled` | `error`, `text` (partial), `elapsed_ms`, `context_len`, `dirty: false` |

A turn is one or more rounds: each round is one request, and a round that
requests tools is followed by another round until the model answers with text.

### Request (`round` is 1-based within the turn)

| Event | Fields |
|---|---|
| `request_started` | `round`, `url`, `model`, `timeout`, `request_bytes`, `messages`, `tools`, `depth`, `max_tokens`, `include_usage` |
| `request_payload` | `round`, `messages` — a per-message summary: `role`, `chars`, `image_count` when the message has images, and `tool_call_id` / `tool_calls` when present |
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
| `history_appended` | `source` (`fork` / `assistant` / `tool`), `role`, `chars`, `preview` (first 200), `image_count`, `tool_call_id`, `tool_calls`, `messages`, `context_len` |

### Tool

| Event | Fields |
|---|---|
| `tool_call_requested` | `round`, `call_id`, `name`, `raw_arguments`, `known` |
| `tool_call_started` | `round`, `call_id`, `name` |
| `tool_call_finished` | `round`, `call_id`, `name`, `ok`, `error`, `result` (the text), `result_chars`, `image_count`, `elapsed_ms`, plus `chain` and `steps` when the call piped |
| `pipe_step_started` | `round`, `call_id` (the step's own, `<call_id>:pipe:<n>`), `parent_call_id` (the model's call), `step`, `name`, `via` (`server` / `client`), `known`, `arguments` (bounded), `chain` |
| `pipe_step_finished` | the same, plus `ok`, `error`, `text` (bounded), `text_chars`, `image_count`, `next` (the tool it piped to, or null), `elapsed_ms` |

`error` is a code, not a message: `unknown_tool`, `bad_arguments`, `tool_raised`,
`bad_result`, `pipe_depth` (the pipe hit its step limit), `timeout`, or `no_hook`.
A tool that fails still produces a result, which is fed back to the model so it
can correct itself.

Calls to a **local tool** — one this client declared — announce themselves
instead of running:

| Event | Fields |
|---|---|
| `local_tool_called` | `round`, `call_id`, `name`, `kind` (`call`), `arguments`, `raw_arguments`, `timeout_ms` — the turn is now parked on this; a piped step adds `parent_call_id`, `step` and `chain` |
| `local_tool_rollback` | `call_id` (of the undo), `name`, `kind` (`rollback`), `rollback_of`, `arguments`, `result`, `call_ok`, `timeout_ms` |
| `local_tool_resolved` | `call_id`, `name`, `kind`, `ok`, `error`, `result` (the text), `result_chars`, `image_count`, `next` (the tool a piped answer named, or null), `waited_ms` |
| `local_tool_unresolved` | `call_id`, `name`, `kind`, `reason` (`timeout` / `cancelled`), `waited_ms` |

`kind` distinguishes a call the model asked for from an undo the harness is asking
for. Answer either with the same `resolve_tool` command.

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
   "params":[{"name":"question","type":"string","description":"what to ask"}],
   "rollback": true, "external_effects": true}
]}
```

They inherit through `fork` (pass `local_tools` to replace the set), are reported
as `local_tools` by `agent_created`, `agent_forked` and `list_agents`, and survive
a restart because the definitions are stored whole. `"rollback": true` promises
that the client can undo a call to this tool; without it the server has no reason
to ask (see below). `"external_effects": true` declares that calls to this tool
reach beyond the state, which is what lets a failure note warn that such effects
may still stand when there is no undo for them.

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

## Tool pipes

A tool normally answers with text. A **server-side hook** may instead return a
`ToolCall` (or a `ToolResult` carrying one), and a **local tool** may answer
`resolve_tool` with `call`; either way the harness runs that call next, before the
model is consulted again, and keeps following calls until one returns text. A pipe
may run through any mix of server-run and client-run tools.

All of this arrived in protocol `4`: a `3` server does not know `resolve_tool`'s
`call` (without a `result` it answers `bad_field`, with one it quietly drops the
call), and it never sends the `pipe_step_*` events or `pipe_traces`. Check
`session_hello.protocol` before piping to an unknown server.

The point is what the model does **not** see. Only the tools the pipe called and
the *last* call's output become the tool message:

```
[tool pipe] tg_download_file_to_sandbox -> nix_add_file
wrote /workspace/photo.jpg (184320 bytes)
```

Everything in between — the arguments each step was given, what it returned and
when — is recorded on the block and reported through events, but never sent to the
provider. A client can therefore download a file and pipe its bytes into
`nix_add_file` without the model ever quoting a base64 payload.

A server hook does it in code:

```python
def download_executor(context: ToolContext, url: str) -> ToolResult:
    path, data = fetch(url)
    return ToolResult(
        f"downloaded {len(data)} bytes",     # for the trace, not the model
        call=ToolCall("nix_add_file", {
            "sandbox_id": context.state["sandbox"],
            "path": path,
            "content_base64": base64.b64encode(data).decode("ascii"),
        }),
    )
```

...or a client, answering a local call:

```jsonc
→ {"command":"resolve_tool","id":"a1","call_id":"call_1",
   "call":{"name":"nix_add_file","arguments":{"sandbox_id":"sbx-7f",
            "path":"/workspace/photo.jpg","content_base64":"…"}}}
← {"event":"local_tool_answered","call_id":"call_1","ok":true,"next":"nix_add_file"}
← {"event":"pipe_step_started","call_id":"call_1:pipe:1","parent_call_id":"call_1",
   "step":1,"name":"nix_add_file","via":"server","known":true,
   "chain":["tg_download_file_to_sandbox","nix_add_file"],"arguments":{"sandbox_id":"sbx-7f",…}}
← {"event":"pipe_step_finished","call_id":"call_1:pipe:1","name":"nix_add_file",
   "ok":true,"error":null,"text":"wrote /workspace/photo.jpg (184320 bytes)","next":null,…}
← {"event":"tool_call_finished","call_id":"call_1","name":"tg_download_file_to_sandbox",
   "ok":true,"chain":["tg_download_file_to_sandbox","nix_add_file"],"steps":1,
   "result":"[tool pipe] tg_download_file_to_sandbox -> nix_add_file\nwrote /workspace/photo.jpg (184320 bytes)"}
```

Rules worth knowing:

- **Each step gets a derived `call_id`**: `<the model's call id>:pipe:<n>`, numbered
  from 1. It is what a piped local call is answered with, and what a rollback of
  that step reports.
- **A pipe is bounded** to `MAX_PIPE_DEPTH` (16) calls. Hitting the limit is not a
  turn failure: the model gets an explanatory result with `error: "pipe_depth"`.
- **An unknown piped tool** is the same: the model is told `unknown tool 'x'` and
  the turn continues. `pipe_step_started.known` says so up front.
- **Only the last call may return images.** A step that both pipes and returns
  images is a `bad_result`; so is a call with no name or with non-object
  arguments.
- **Intermediate values are recorded bounded**, not whole: a string over
  `PIPE_VALUE_MAX_CHARS` (4096) is kept as a prefix plus `+n chars` and a
  `sha256` prefix, and bytes as a size and digest. `get_context` returns the
  records as `pipe_traces`, each holding `call_id`, `round`, `chain`, `ok`,
  `error`, `result`, `result_chars`, `elapsed_ms` and one `steps` entry per call
  (`call_id`, `name`, `via`, `arguments`, `ok`, `error`, `text`, `text_chars`,
  `image_count`, `next`, `elapsed_ms`).
- **Every step rolls back with the turn.** A failed, cancelled or abandoned turn
  offers each call it made an undo, the pipe's steps included, newest first.
- **A client's piped `call` travels over this connection**, so the 8 MiB command
  limit applies to it; a server-side pipe has no such limit.

## Rollback

When a turn does not commit — `failed`, `cancelled` or abandoned — every tool call
it made is offered an undo, **newest call first**. A server-side tool's `rollback`
hook is called directly; a client-run tool is asked over the protocol, exactly like
a forward call:

| Event | Fields |
|---|---|
| `rollback_started` | `call_id`, `tool`, `index`, `total` |
| `rollback_unavailable` | `call_id`, `tool`, `index`, `total` — no undo exists and the tool declared effects, so they may still stand |
| `rollback_finished` | `call_id`, `tool`, `ok`, `error`, `result`, `result_chars`, `elapsed_ms` |
| `failure_summary_started` | `model`, `error`, `outcome`, `messages`, `request_bytes`, `timeout` |
| `failure_summary_finished` | `status`, `summary`, `chars`, `chunks`, `finish_reason`, `usage`, `elapsed_ms` |
| `failure_summary_failed` | `error`, `error_type`, `elapsed_ms` (plus `status` on an HTTP refusal, `chunks` if it broke mid-stream) |

```
← {"event":"rollback_started","call_id":"call_2","tool":"ask_operator","index":1,"total":2}
→ {"command":"resolve_tool","id":"a1","call_id":"call_2:rollback","result":"undone"}
← {"event":"rollback_finished","call_id":"call_2","tool":"ask_operator","ok":true,
   "error":null,"result":"undone","result_chars":6,"elapsed_ms":3.1}
```

Rules worth relying on:

- **A failing undo is contained.** It comes back as `rollback_finished` with
  `ok: false` (and `error` either `rollback_raised`, `bad_arguments`, or whatever
  the client reported), the rest still run, and the turn's own failure is
  unaffected. A rollback never turns a failed turn into a different error.
- **Only calls that reached a tool are undone.** An unknown tool or arguments that
  never bound did nothing, so they are not in the list. A client-run call that was
  asked and never answered *is* listed, because the client may have run it.
- **Undos are not waited on for a cancelled turn.** The request is sent with
  `timeout_ms: 0`, so `cancel` returns promptly; expect
  `local_tool_unresolved` with `reason: "cancelled"` for those.
- **A merely failed turn does wait**, for `local_timeout`, because the caller is
  still there. Answer promptly or the undo is abandoned and reported.

## Failed turns leave a note

A turn that does not commit rolls back and then appends **one ordinary message**
to its block, so a block forked from it carries the truth in its context: the
transcript above may still describe tool effects that the rollback has undone.

```
← {"event":"history_appended","source":"failure","role":"user",
   "preview":"[harness] the previous turn failed; the state it changed was discarded.…"}
```

The message is `user`-role and prefixed `[harness]`, and looks like:

```
[harness] the previous turn failed; the state it changed was discarded.
Undone: get_current_time.
Could not be undone: terminal.
May still be in effect: sendmail.
Reported error: HHAgentError: stream from … failed: Connection broken
In short: <what the turn was doing and where it stood>
```

The three lists are the whole truth about the environment, and they partition the
calls that declared `external_effects`:

| Line | Means |
|---|---|
| `Undone:` | the tool had an undo and it ran |
| `Could not be undone:` | the tool had an undo and it failed |
| `May still be in effect:` | the tool declared effects but has no undo at all |

A tool that declares nothing is not mentioned: its state was still rolled back
with the rest of the turn, and it claimed no effects outside it. So a tool with
real-world effects should declare `external_effects` — builtin as a field,
client-run in its definition — or the note will quietly leave them out.

- **A `failed` turn is summarised** by a model — one request, no tools, sent to
  `summary_model` or the block's own model. `failure_summary_started` /
  `_finished` / `_failed` report it. If that request fails, the summary is
  dropped and the note keeps the raw error, which is always included verbatim.
- **A `cancelled` or abandoned turn uses fixed text** and makes no request, so
  cancelling stays prompt and teardown does no network I/O.
- **A turn with no tool call and no text is not summarised**: the error is the
  whole story.
- **The partial answer is not part of the note.** It is kept as the block's `text`
  and fed to the summariser, but not added to `messages`: a truncated fragment
  read as a finished reply misleads.
- **`outcome` on the block** says which of `ok` / `failed` / `cancelled` /
  `abandoned` it was, and is persisted, so this does not have to be inferred from
  the error string.

## Error codes

| Code | Meaning |
|---|---|
| `bad_json` | the line was not JSON |
| `bad_command` | not a JSON object, or `command` is not a string |
| `command_too_large` | over 8 MiB |
| `unknown_command` | no such command; the reply lists the valid ones |
| `bad_id` | `id` / `new_id` missing or not a non-empty string |
| `bad_prompt` | `prompt` missing or empty |
| `bad_image` | `images` was malformed: not a list, an entry without a non-empty `url`, or a non-string `detail` |
| `bad_result` | a tool hook returned something other than a string, `ToolResult` or `ToolCall`, a `ToolResult` whose images were malformed, or a pipe `call` that was unusable |
| `bad_call` | `resolve_tool`'s `call` was not `{name, arguments}`, or it came with `error` or `images` |
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

**A pipe is invisible to the model except at its ends.** The tools it called and
the last call's output appear in the tool message; every intermediate argument and
result stays on the block. A turn that fails rolls back each step like any other
call.

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
