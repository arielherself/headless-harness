"""The wire-protocol version, in a module of its own.

`tools` is imported by `agent`, `store` and `server`, so a constant both the
tools and the server need has to live below `tools`: a constant defined in
`server` cannot be imported back from `tools` without closing an import cycle.
This module imports nothing, which is what lets it sit under all of them.
"""

PROTOCOL_VERSION = 5
