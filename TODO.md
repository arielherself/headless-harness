# TODO

## Unbounded context growth

**Status:** open — nothing implemented yet.

`HHAgent.context()` in `src/agent.py` rebuilds the request context by walking the
block chain to the root and concatenating every block's messages. Nothing is
trimmed, summarised or capped, so the request body grows linearly with the depth
of the chain: turn N sends all N blocks.

Storage is incremental (each message is stored once, no copies between blocks),
but the *transport* is not. A long enough chain will exceed the provider's
context window and the request will be rejected. The default model
`deepseek/deepseek-v4.1-flash` advertises a 1M token window.

Where to look today: `request_started.messages` (count), `request_payload`
(role/chars summary of what went out), `context_len` on most turn events, and the
`get_context` command for the full list.

### Candidate directions

1. **Budgeted truncation / summarisation.** Pick a token or message budget; once
   a chain exceeds it, drop or summarise the oldest ancestor blocks.
2. **Pinned + summarised blocks.** Give each block a flag: `pinned` blocks go out
   verbatim, others are sent as a summary. `context()` walks the chain and spends
   the budget on pinned blocks first. Fits the chain shape — a summary is just
   another message list the block owns, and forking is a natural place to
   generate one.
3. **Manual compaction.** Leave the request path alone and let the client fork a
   fresh chain from an early block. Works today, but discards history instead of
   compressing it.

### Constraints to respect when implementing

- Tool-call pairing must survive: an `assistant` message carrying `tool_calls`
  cannot be separated from the `tool` messages answering it.
- `_stream_turn` recomputes `context()` once per tool round, so a budget has to
  stay stable within a turn or the model sees the context shift mid-turn.
- A context that references tool calls while `tools` is absent must keep working;
  the empty-`tools` case is already handled and the provider accepts it.
