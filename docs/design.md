# Design

Why the harness is shaped the way it is. For the wire format see
[`protocol.md`](protocol.md); for tool authoring see [`tools.md`](tools.md).

## 1. A chain of blocks, not a transcript

The obvious design is one object holding a `messages` list that grows. This
harness instead makes each prompt its own object:

- `HHAgent.root(...)` creates a root: configuration, no prompt, not dirty.
- `block.fork(prompt)` creates a child holding that prompt, linked by `parent`.
- A block owns exactly the messages its own turn produced; nothing else.

`block.context()` (`agent.py`) rebuilds the full request by walking `parent` to
the root and concatenating each block's `messages`. The walk is the whole of the
storage story: a block never copies anything an ancestor already holds, so a
thousand-turn chain still stores each message once.

Two consequences fall out of that, and they are the point of the design:

- **Siblings are invisible.** `context()` follows `parent` pointers, so it
  returns one path, not a tree walk. A branch's messages never leak into a
  sibling's request even though both live in the same registry.
- **Rewinding is free.** Forking from an ancestor gives you that ancestor's
  context exactly, because a context *is* the path. There is no undo logic
  anywhere, and no block is mutated to rewind. The abandoned branch's blocks
  remain valid and can be returned to.

The context is rebuilt on every request rather than cached. It costs O(depth) of
pointer chasing, and it removes an entire class of bugs: there is no cache to
invalidate when a block's messages change.

## 2. A block runs once: the dirty bit

A freshly forked block is `dirty` — it holds a prompt but has not produced a
reply yet. `run` executes its turn and clears the flag when the turn ends,
however it ends.

`dirty` is not decoration; it is an invariant the API enforces:

- **Forking from a dirty block is refused** (`parent_dirty`). Its context is
  still growing, so a child built on it would inherit a half-built baseline.
- **Running a finished block is refused** (`agent_finished`). One block, one
  turn, forever.
- **Running the same block twice at once is refused** (`agent_running`). The
  authoritative guard is the block's own non-blocking run lock inside
  `stream()`; the pre-check in `cmd_run` is only a fast path, so a tight race
  degrades into an asynchronous error rather than a doubled turn.

So a chain in use never contains a dirty block except at its tip, which is what
makes "fork from any block you can see" safe.

## 3. A turn is atomic; a block is its own transaction

`stream()` (`agent.py`) tracks one `outcome` and funnels every path — success,
cancellation, failure, and the caller abandoning the generator — through a single
`finally` that calls `_release_states`. Only the success path sets
`outcome = "commit"`.

**Messages and state follow different rules, deliberately:**

| | on a successful turn | on failure / cancel / abandon |
|---|---|---|
| messages | recorded as produced | recorded as produced (partial) |
| tool state | committed as a delta | **discarded entirely** |

The asymmetry is not an oversight. A transcript is a *log*: a half-written log is
still a valid log, and knowing how far it got is useful. State is a *value*: three
of five install steps is not a smaller environment, it is an invalid one that
neither the model nor the tool can reason about. So a turn's state lands whole or
not at all — a failed block is *transparent* for state, and forking from it yields
the environment as of its parent.

Discards are not silent: `state_discarded` names the namespace, the keys, and the
reason (`failed`, `cancelled`, `abandoned`).

### Rolling back effects outside the state

Discarding a delta takes care of the state, but a tool may have done something
else — installed a package, written a file, called an API. So a tool may declare a
`rollback` hook (or, if it runs on the client, promise that the client has one),
and when a turn does not commit, every call it made is offered an undo **newest
call first**, mirroring an undo stack.

Two rules make rollback safe to rely on:

- **A failing rollback is contained.** It is reported as `rollback_finished` with
  `ok: false`, and the remaining rollbacks still run. An undo that throws must not
  strand its neighbours, and it must never replace the failure that triggered the
  whole thing.
- **Only calls that reached a tool are undone.** An unknown tool, or arguments
  that never bound, did nothing and is left out — undoing it would be inventing
  work. A client-run call that was asked and never answered *is* included, because
  the client may have run it before going quiet.
- **Declared effects with no undo are named, not omitted.** A rollback hook can
  only exist for effects the author knows how to take back, so the note would
  otherwise be silent about the rest — and its opening line, "the state it changed
  was discarded", invites reading that as "the environment is as it was". A tool
  therefore declares `external_effects`, and the calls that declared it are
  partitioned into undone, undo failed, and *may still be in effect*. Silence is
  reserved for tools that claimed nothing, which is the honest reading.

A cancelled or abandoned turn asks for its undos without waiting: the request goes
out, no reply is waited on, and `cancel` stays responsive. A merely failed turn
waits, because the caller is still there.

### Leaving a note behind

Rollback restores the environment, which makes a failed block an *attractive*
place to fork from — and that is exactly what makes a note necessary. The
transcript above it still says `tool: installed python3-requests` while the
environment no longer has it, so a child that forks from a failed block would
trust a narrative that has stopped being true.

So a turn that does not commit appends one ordinary message to its block:

```
[harness] the previous turn failed; the state it changed was discarded.
Undone: get_current_time.
Could not be undone: terminal.
Reported error: HHAgentError: stream from … failed: Connection broken
In short: 这一轮的目标是安装 …；安装已生效且无法回滚，取时间的调用被撤销。
```

Design points behind that shape:

- **An ordinary message, not a side channel.** It is a plain `user` message, so
  it flows into every descendant's context through `context()` with no special
  casing, and `history_appended` reports it like any other (`source: "failure"`).
  The `[harness]` prefix is what marks it as the harness talking: mid-conversation
  `system` messages are refused by some OpenAI-compatible backends, while
  consecutive `user` messages are accepted (verified).
- **The raw error is kept verbatim.** The summary is written by a model and can
  distort the cause; the error is the part that is certainly true.
- **The summary is written by a model, one request, no tools.** It sees the
  turn's own transcript, the error, and what the rollback managed to undo. A
  `summary_model` can point it at something cheaper or more reliable than the
  model that just failed. If that request fails too, the note falls back to the
  raw error — which is why the error is in the note regardless.
- **The note is written before `dirty` clears**, so forking from the block can
  never miss it. That is also why a failure costs one extra request before it is
  reported.
- **Only a failed turn is summarised.** A cancelled turn gets fixed text: putting
  an LLM call in the way of the stop the caller just asked for would undo the
  point of cancelling. An abandoned generator gets fixed text too — it must not
  start network calls during teardown at all.
- **A turn with nothing to summarise is not summarised.** If it called no tool and
  produced no text, the error *is* the whole story.

The partial answer, if the model had started writing, is kept on the block as
`text` but deliberately **not** added to `messages`: a truncated fragment read as
a finished reply misleads. It goes into the summary instead.

## 4. Tool state: deltas, namespaces, and deep copies

Tool state uses the same shape as messages. A block stores only
`state_deltas: dict[namespace, StateDelta]`, and a tool's live state is the
lineage's deltas replayed in order. `StateDelta.changed` holds the new value of
each touched top-level key; `removed` names the keys that disappeared.

Three decisions are worth explaining.

**Namespaces, not tool names.** State is keyed by the namespace a tool declares
in `state_namespace`, which defaults to the tool's own name. Two tools that
should share one memory — a get/set pair — declare the same namespace; everything
else stays isolated by default. The isolation is deliberate: without it, any tool
could stomp on any other tool's keys.

**Diff at turn end, not write tracking.** The alternative — a dict subclass that
records assignments — misses nested mutation. `state["packages"].append("x")`
performs no `__setitem__` on the outer dict, so the delta would come out empty
and the install would vanish silently. Diffing the live state against a baseline
captured at first touch catches nested edits for free. The cost is that
granularity is the top-level key: bury everything under one key and every block
stores that whole value.

**Deep copy on read.** The state handed to a tool is a `deepcopy` of the merged
deltas. Without it, a tool editing `state["packages"]` in place would corrupt the
*deltas stored in ancestor blocks* — the history would be rewritten and fork
semantics destroyed. The same copy is what makes concurrent turns on sibling
branches safe: each works on its own baseline, so no lock is needed for state.

Equality during the diff is guarded (`_same_value`): a comparison that cannot be
reduced to a bool, as with a numpy array, counts as changed. A delta may then be
a superset, never a wrong one.

## 5. Local tools

A block may also carry `local_tools`: tools the **client** implements. Only the
schema travels — name, description, parameters — because the hook is on the
client's side, and the model sees them in the same `tools` array as everything
else. `ToolEntry.hook is None` is what marks one.

When the model calls one, nothing is executed here. The harness emits
`local_tool_called` and parks that turn's thread until a `resolve_tool` command
brings the result, which then becomes the tool message fed back to the model —
indistinguishable, from the model's point of view, from a server-side tool.

**The wait holds nothing.** It is one `threading.Event` and a deadline: no
registry lock, no store lock, no open SQLite transaction while a client decides.
Other connections can create, fork, run, seed and evict blocks the whole time,
and the database is written throughout. The tests assert this directly — both
locks acquirable, `_db.in_transaction` false, and a write committing mid-park —
rather than inferring it from things being fast.

Three consequences worth stating:

- **A local tool has no state on the server.** `ToolContext.state` lives here, so
  a client-run tool cannot use it; whatever memory it needs is the client's own.
  It follows that forking rewinds builtin state but cannot rewind a client's
  private memory.
- **A slow client only hurts its own turn.** Other turns, other connections and
  the database are untouched; the parked turn simply occupies one thread.
- **Definitions are persisted, pending calls are not.** A turn is only written
  when it ends, so a process that dies while a client is thinking loses that turn
  — exactly as it would lose one that died mid-request. The definitions are
  stored whole (unlike builtin tools, which are stored by name), so a restarted
  server still offers the tool and can ask again.

**A timeout is not a failure.** With no answer inside `local_timeout`, the tool
result becomes an explanatory error string and the turn carries on, so the model
can react just as it would to a tool that raised. `local_tool_unresolved` reports
the reason (`timeout`, `cancelled`), and `cancel` releases a parked turn like any
other.

## 6. Tool pipes

A tool can answer with a call instead of text:

```python
return ToolCall("nix_add_file", {
    "sandbox_id": sandbox,
    "path": path,
    "content_base64": base64.b64encode(data).decode("ascii"),
})
```

The harness runs that call next — server-side or, for a tool with no hook, by
parking the turn and asking the client — and follows whatever it returns until a
call answers with text. One tool hands off to the next; the model is not
consulted in between.

The motivation is the transcript. A file's bytes are exactly the kind of thing a
tool can hold but a model should never have to quote: routing them through a
tool-call argument would make every later request carry the base64, at token
cost, for information the model cannot use. A pipe keeps the value where it
belongs — in the process, and in the block's own record — while still letting the
model ask for the work in one call and see it done.

Three decisions shape the implementation.

**Only the ends reach the model.** The tool message is the chain of names plus
the last call's output (`[tool pipe] fetch -> write_file\nwrote 1024 bytes`).
Anything else the steps said is dropped from the transcript, not summarised, so
there is no way for an intermediate value to leak back in — including through
the tool message, which is built from the final reply alone.

**The middle is still recorded.** `HHAgent.pipe_traces` keeps one record per
pipe: the chain, each step's call id, `via`, arguments, its own result and
whether it failed. `get_context` serves them, and `pipe_step_started` /
`pipe_step_finished` report them live, so "what actually ran" is inspectable
even though "what the model saw" is deliberately small. Values are stored
bounded — a prefix, the true length and a digest — since a step may legitimately
carry a megabyte, and the database's budget is sized for conversations, not for
payloads that live in the sandbox anyway.

**Every step is a call in its own right.** Each one is recorded in the turn's
call list with its own derived id (`call_1:pipe:1`), so rollback walks them
newest-first like any other call, `state_loaded` is emitted per step, and a
client-run step is answered with the same `resolve_tool` command — with `call`
instead of `result` when the client wants to pipe onward itself. A pipe is
bounded to `MAX_PIPE_DEPTH` calls, because it runs without the model in the loop
and a tool that keeps asking for itself would otherwise never end.

## 7. Persistence

`HHStore` mirrors the registry into SQLite. Per block it stores identity and
linkage, the conversation the block owns, its state deltas, its lifecycle flags
and the settings that shape a turn.

**What is deliberately not stored:**

- **The API key.** Restored blocks inherit the key the server was started with.
  A per-block key passed to `create_agent` is lost across a restart, by design.
- **Tools themselves.** A hook is a function and cannot be serialised, so blocks
  store tool *names* and the loader resolves them against the catalogue. A name
  that no longer resolves is reported in `session_hello.store.warnings` rather
  than swallowed.
- **State as JSON.** `ToolContext.state` may hold any Python object, so deltas
  are pickled. Each namespace is checked individually: one tool whose state
  refuses to pickle costs only that tool's state, and `persist_warning` says so.

**Write points.** A block is written twice: once at `fork` (holding just the
prompt and any images) and once when its turn ends. Nothing is written during a turn, so a
process that dies mid-turn leaves the block looking forked-but-never-run and the
whole turn is retried from scratch.

**Durability before the completion event.** `_run_turn` persists *before* it
sends `command_finished`. A client that sees a turn finish and then loses the
process must not lose the turn. This was a real bug once: the write came after
the event, so killing the server the instant a turn completed dropped it.

**Deletion cascades.** `parent_id` is a self-referencing foreign key with
`ON DELETE CASCADE`, so removing a subtree cannot leave an orphan — and a
config of `PRAGMA foreign_keys = ON` is set per connection.

## 8. Eviction

The file has a budget (`--max-db-bytes`, default 64 MiB, `0` disables). After
every write the size is measured and whole subtrees are evicted until it fits.

**A candidate is a root tree, or a block whose parent has other forks.** That is
the definition the design settled on, and the reasoning is worth recording:

- Any *single* block as a candidate degenerates. A subtree's "newest block" is
  minimised at a leaf, so the rule would erode chains one tip at a time — eating
  exactly the most recent context of every conversation.
- A fork point is where storage actually diverges, so it is the natural unit for
  reclaiming it. A linear stretch has no independent existence: it is just the
  tail of its parent.

The consequence is that **a conversation with no forks is atomic** — either it
survives whole or it disappears whole, and a chat is never truncated mid-way.

**Age is the newest block inside the subtree**, so a chain that keeps growing
stays young and only quiet branches are dropped. The candidate set is recomputed
after each eviction, which has a visible effect: once a branch is the last
survivor under its parent, that parent no longer has forks, so the branch loses
candidacy and can only go together with its tree.

**The budget is soft.** A subtree with a turn in flight is skipped, so the file
can overshoot until that turn ends; the next write tries again. If only protected
trees remain, eviction stops rather than emptying the store.

Two limits worth knowing: SQLite's own schema occupies ~24 KiB, so a budget below
that can never be met (the loop empties the store and stops); and eviction only
runs after a write, so an over-budget file stays that way until the next one.

## 9. Concurrency

Threaded, with a few deliberate serialisation points.

**Concurrent:** each connection gets a thread (`ThreadingTCPServer`), and each
`run` spawns its own turn thread, so a connection stays responsive while its turn
streams — `ping` and `cancel` are answered mid-turn. Sixteen simultaneous turns
complete in the time of one.

**Serialised:**

- *Commands on one connection* are handled one at a time by that connection's
  handler thread. `run` returns immediately, so this rarely matters; `list_agents`
  (which walks every chain) and a write that triggers eviction are the cases where
  it can.
- *The registry* is guarded by `_registry_lock`.
- *The store* is one SQLite connection with an `RLock`; every statement is
  serialised. It sits on the path of `persist`, which now precedes
  `command_finished`, so a slow disk delays completion events.
- *CPU work* is subject to the GIL: serialising a large context, deep-copying
  tool state.

**One lock ordering:** registry → store. Nothing takes them the other way round,
which is what keeps the atomicity below deadlock-free.

**Two writes must not interleave.** `persist` checks that a block is still
registered *and* writes it under one lock hold, and `enforce_limit` picks a
victim, deletes its rows and drops it from the registry under one hold as well.
Splitting either pair lets an eviction land between another turn's check and its
insert, and the insert then fails its foreign key — the turn thread dies after the
client was told nothing, and the client waits forever. This was a real bug; the
fix is the shared hold, and the test that catches it injects a delay into `save`
and evicts in the middle.

## 10. Known limits and open work

- **Context grows with the chain.** Nothing is trimmed, summarised or capped, so
  the request body grows linearly with depth. Tool state is deliberately exempt:
  it is not LLM-facing, so there is nothing to trim. See `TODO.md`.
- **One process per database file.** A second opener fails fast (an exclusive
  `flock` on `<path>.lock`) because two servers would keep two registries and
  silently disagree about which blocks exist.
- **Tool history is not recoverable when code changes.** A delta is keyed by
  namespace, so changing a tool's `state_namespace` makes its historical state in
  old blocks unreachable, and renaming a tool loses that name's resolution.
  Unpickling state also depends on the classes still being importable.
- **Nothing is capped.** A client can create blocks without limit until the
  database budget evicts them.
