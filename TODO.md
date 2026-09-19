# TODO

## Context truncation and what it leaves open

**Status:** truncation implemented; summarisation and pinning are open.

`HHAgent._plan_context()` in `src/agent.py` walks the chain from the block back
to the root and takes whole blocks until the budget — `context_window` minus
`max_tokens` — is spent, so the newest survive and the oldest are dropped from
the request. The store keeps them: `context()` still returns the whole chain for
reporting, and `context_truncated` / `request_started` / `get_context` report
both sides of the cut.

The cut is planned once per turn and reused by every tool round, so the context
cannot shift under the model mid-turn. Sizes are estimated by `estimate_tokens()`
from characters, with a fixed cost per message and per image (a `data:` URI's
length says nothing about what the model sees), because no tokenizer is
available; the estimate rounds up.

### Candidate directions

1. **Summarise instead of dropping.** A dropped block could be replaced by a
   summary the next request carries. Fits the chain shape — a summary is just
   another message list the block owns, and forking is a natural place to
   generate one.
2. **Pinned + summarised blocks.** Give each block a flag: `pinned` blocks go out
   verbatim, others are sent as a summary. `_plan_context()` walks the chain and
   spends the budget on pinned blocks first.
3. **Manual compaction.** Leave the request path alone and let the client fork a
   fresh chain from an early block. Works today, but discards history instead of
   compressing it.

### Constraints to respect when implementing

- Tool-call pairing survives by construction today: the cut only ever falls
  between blocks, so an `assistant` message carrying `tool_calls` cannot be
  separated from the `tool` messages answering it. A summariser must keep that.
- A budget has to stay stable within a turn or the model sees the context shift
  mid-turn; `stream()` plans once for this reason.
- A context that references tool calls while `tools` is absent must keep working;
  the empty-`tools` case is already handled and the provider accepts it.

The default model `deepseek/deepseek-v4.1-flash` advertises a 1M token window,
which is where `DEFAULT_CONTEXT_WINDOW` comes from.

### Tool state is deliberately never trimmed

Settled, not open: any future budget applies to `messages` only and must leave
`HHAgent.state_deltas` alone. Tool state is not LLM-facing — it never enters the
request payload; only `messages` and the tool schemas do — so it is an
environment, not a transcript. Trimming it would corrupt the one thing the
chain exists to reproduce faithfully.

The accepted cost sits on the other side: a trimmed-away message may be the only
place a state change is *explained*, so the model can lose the memory of doing
something whose effect is still present in the environment. Two practical
consequences: a tool should describe its effects in its own result text, and a
caller must not assume the model remembers a change just because the state still
shows it.
