"""SQLite persistence for agent blocks.

The registry lives in memory; this module mirrors it to a SQLite file so a
restart can rebuild the same chains.

Per block it stores identity and linkage (`id`, `parent_id`), the conversation
the block owns (`prompt`, `messages`, `text`, `error`), its tool state deltas,
its lifecycle flags (`dirty`, `created_at`) and the settings that shape a turn
(`endpoint`, `model`, `timeout`, `include_usage`, `verbose`). Tools are stored by
*name* and resolved against the catalogue on load, because a hook is a function
and cannot be persisted. The API key is never written, so blocks restored from
disk inherit whatever key the server was started with.

A block is written once when it is forked and once when its turn ends, so a
process that dies mid-turn leaves the block looking forked-but-never-run and the
whole turn is retried from scratch. A turn that ends — even badly — is written
as it stands in memory, partial messages and `error` included.

Growth is bounded by `max_bytes`. The caller drains `candidates()`, which lists
the subtrees that may be reclaimed, oldest first: a candidate is a root tree, or
a block whose parent has other forks, and its age is the *newest* block inside
it. So a conversation stays young while it keeps growing, a plain chain is
all-or-nothing, and only quiet branches are dropped.
"""

import json
import pickle
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:  # POSIX only; without it the store still works, just without the lock
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

from agent import HHAgent, StateDelta
from tools import ToolEntry, ToolParam, builtin_tools

SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks (
    id            TEXT PRIMARY KEY,
    parent_id     TEXT REFERENCES blocks (id) ON DELETE CASCADE,
    prompt        TEXT NOT NULL,
    messages      TEXT NOT NULL,
    state_deltas  BLOB NOT NULL,
    text          TEXT NOT NULL,
    error         TEXT,
    dirty         INTEGER NOT NULL,
    created_at    REAL NOT NULL,
    endpoint      TEXT NOT NULL,
    model         TEXT NOT NULL,
    timeout       REAL NOT NULL,
    include_usage INTEGER NOT NULL,
    verbose       INTEGER NOT NULL,
    tool_names    TEXT NOT NULL,
    local_tools   TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS blocks_parent ON blocks (parent_id);
CREATE INDEX IF NOT EXISTS blocks_created ON blocks (created_at);
"""

COLUMNS = (
    "id, parent_id, prompt, messages, state_deltas, text, error, dirty, "
    "created_at, endpoint, model, timeout, include_usage, verbose, tool_names, "
    "local_tools"
)

# Columns refreshed when a block that already has a row changes.
UPDATABLE = (
    "prompt, messages, state_deltas, text, error, dirty, created_at, endpoint, "
    "model, timeout, include_usage, verbose, tool_names, local_tools"
)

# A subtree that may be reclaimed: a root, or a block whose parent has forks.
CANDIDATES = """
WITH RECURSIVE
candidates(id) AS (
    SELECT b.id FROM blocks b
    WHERE b.parent_id IS NULL
       OR (SELECT COUNT(*) FROM blocks s WHERE s.parent_id = b.parent_id) > 1
),
tree(root, id, created_at) AS (
    SELECT c.id, b.id, b.created_at
    FROM candidates c JOIN blocks b ON b.id = c.id
    UNION ALL
    SELECT t.root, b.id, b.created_at
    FROM blocks b JOIN tree t ON b.parent_id = t.id
)
SELECT root, COUNT(*) AS nodes, MAX(created_at) AS newest
FROM tree
GROUP BY root
ORDER BY newest ASC, root ASC
"""


class HHStoreError(RuntimeError):
    """Raised when the database cannot be opened or is already in use."""


class HHStore:
    """A SQLite mirror of the in-memory registry, with a size budget."""

    def __init__(
        self,
        path: str | Path,
        max_bytes: int = 0,
        tools: Iterable[ToolEntry] = builtin_tools,
    ) -> None:
        self.path = str(path)
        self.max_bytes = int(max_bytes or 0)
        self.tools = {tool.name: tool for tool in tools}
        self._lock = threading.RLock()
        # one file, one server: a second process would keep its own registry and
        # the two would silently disagree about which blocks exist
        self._handle = self._claim_file()
        # turns run on their own threads, so the connection is shared and every
        # statement is serialized by the lock rather than by the connection
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute("PRAGMA journal_mode = DELETE")
            # reclaim freed pages, otherwise eviction would never shrink the file
            self._db.execute("PRAGMA auto_vacuum = FULL")
            if self._db.execute("PRAGMA auto_vacuum").fetchone()[0] != 1:
                self._db.execute("VACUUM")
            self._db.executescript(SCHEMA)
            self._migrate()
            self._db.commit()

    def _migrate(self) -> None:
        """Bring a file written by an older version up to the current schema.

        SQLite has no `ADD COLUMN IF NOT EXISTS`, so the column list is checked
        first. Only additive changes are handled here; anything structural would
        need a real migration step.
        """
        columns = {row["name"] for row in self._db.execute("PRAGMA table_info(blocks)")}
        if "local_tools" not in columns:
            self._db.execute(
                "ALTER TABLE blocks ADD COLUMN local_tools TEXT NOT NULL DEFAULT '[]'"
            )

    def _claim_file(self) -> Any:
        if fcntl is None:
            return None
        handle = open(f"{self.path}.lock", "w")  # noqa: SIM115 - kept for the process
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise HHStoreError(
                f"{self.path} is already open in another process"
            ) from exc
        return handle

    def close(self) -> None:
        with self._lock:
            self._db.close()
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # --- writing ---------------------------------------------------------
    def save(self, block: HHAgent) -> list[str]:
        """Insert or refresh one block. Returns tool names whose state was dropped.

        State values are arbitrary Python objects, so they go through pickle; a
        value that will not pickle costs that one tool its state instead of the
        whole block.
        """
        blob, dropped = self._encode_deltas(block.state_deltas)
        values: tuple[Any, ...] = (
            block.id,
            block.parent.id if block.parent else None,
            block.prompt,
            json.dumps(block.messages, ensure_ascii=False, default=str),
            blob,
            block.text,
            block.error,
            int(block.dirty),
            block.created_at,
            block.endpoint,
            block.model,
            float(block.timeout),
            int(block.include_usage),
            int(block.verbose),
            json.dumps(sorted(t.name for t in block.tools.values() if not t.is_local)),
            json.dumps(self._encode_local_tools(block), ensure_ascii=False),
        )
        assignments = ", ".join(
            f"{name.strip()} = excluded.{name.strip()}"
            for name in UPDATABLE.split(",")
        )
        with self._lock, self._db:
            self._db.execute(
                f"INSERT INTO blocks ({COLUMNS}) "
                f"VALUES ({', '.join('?' * len(values))}) "
                f"ON CONFLICT(id) DO UPDATE SET {assignments}",
                values,
            )
        return dropped

    def delete(self, block_id: str) -> int:
        """Delete a block and, by cascade, everything forked from it."""
        with self._lock, self._db:
            cursor = self._db.execute("DELETE FROM blocks WHERE id = ?", (block_id,))
        return cursor.rowcount

    # --- reading ---------------------------------------------------------
    def load(self, key: str) -> tuple[list[HHAgent], list[str]]:
        """Rebuild every stored block, linking parents.

        Returns the blocks and human-readable warnings. A row that will not
        decode is *kept* with the broken part emptied rather than dropped:
        dropping it would orphan its descendants, and an orphan silently loads
        as a root, which would rewrite that chain's context.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM blocks ORDER BY created_at, id"
            ).fetchall()
        blocks: dict[str, HHAgent] = {}
        warnings: list[str] = []
        for row in rows:
            names = json.loads(row["tool_names"])
            missing = [name for name in names if name not in self.tools]
            if missing:
                warnings.append(f"{row['id']}: missing tools {missing}")
            local = [self._decode_local_tool(entry) for entry in json.loads(row["local_tools"])]
            block = HHAgent.root(
                row["endpoint"],
                key,
                model=row["model"],
                tools=[self.tools[name] for name in names if name in self.tools] + local,
                timeout=row["timeout"],
                include_usage=bool(row["include_usage"]),
                verbose=bool(row["verbose"]),
                id=row["id"],
            )
            try:
                block.messages = json.loads(row["messages"])
            except ValueError as exc:
                warnings.append(f"{row['id']}: messages unreadable ({exc})")
                block.messages = []
            deltas, dropped = self._decode_deltas(row["state_deltas"])
            if dropped is not None:
                warnings.append(f"{row['id']}: state unreadable ({dropped})")
            block.state_deltas = deltas
            block.prompt = row["prompt"]
            block.text = row["text"]
            block.error = row["error"]
            block.dirty = bool(row["dirty"])
            block.created_at = row["created_at"]
            blocks[row["id"]] = block
        for row in rows:
            parent_id = row["parent_id"]
            if not parent_id:
                continue
            if parent_id in blocks:
                blocks[row["id"]].parent = blocks[parent_id]
            else:
                warnings.append(f"{row['id']}: parent {parent_id} is gone, treated as root")
        return [blocks[row["id"]] for row in rows], warnings

    def size_bytes(self) -> int:
        """The size of the database file, as SQLite accounts for it."""
        with self._lock:
            pages = self._db.execute("PRAGMA page_count").fetchone()[0]
            page_size = self._db.execute("PRAGMA page_size").fetchone()[0]
        return pages * page_size

    def candidates(self) -> list[sqlite3.Row]:
        """Reclaimable subtrees, oldest first, each with `root`, `nodes`, `newest`."""
        with self._lock:
            return self._db.execute(CANDIDATES).fetchall()

    def subtree_ids(self, root_id: str) -> list[str]:
        """Every block under `root_id`, including it."""
        with self._lock:
            rows = self._db.execute(
                """
                WITH RECURSIVE tree(id) AS (
                    SELECT id FROM blocks WHERE id = ?
                    UNION ALL
                    SELECT b.id FROM blocks b JOIN tree ON b.parent_id = tree.id
                )
                SELECT id FROM tree
                """,
                (root_id,),
            ).fetchall()
        return [row["id"] for row in rows]

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            blocks = self._db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        return {
            "path": self.path,
            "blocks": blocks,
            "trees": len(self.candidates()),
            "bytes": self.size_bytes(),
            "max_bytes": self.max_bytes,
        }

    # --- local tools -----------------------------------------------------
    @staticmethod
    def _encode_local_tools(block: HHAgent) -> list[dict[str, Any]]:
        """The definitions of the tools the client runs, for a later restart.

        A local tool is only a schema — the hook lives on the client — so unlike
        a builtin it can be stored whole rather than by name.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "params": [
                    {"name": p.name, "type": p.type, "description": p.description}
                    for p in tool.params
                ],
                "rollback": tool.remote_rollback,
            }
            for tool in block.tools.values()
            if tool.is_local
        ]

    @staticmethod
    def _decode_local_tool(entry: dict[str, Any]) -> ToolEntry:
        return ToolEntry(
            name=entry["name"],
            description=entry.get("description", ""),
            params=[
                ToolParam(
                    name=p["name"], type=p.get("type", "string"),
                    description=p.get("description", ""),
                )
                for p in entry.get("params") or []
            ],
            hook=None,
            remote_rollback=bool(entry.get("rollback")),
        )

    # --- encoding --------------------------------------------------------
    @staticmethod
    def _encode_deltas(deltas: dict[str, StateDelta]) -> tuple[bytes, list[str]]:
        """Pickle the deltas, dropping only the tools whose state refuses."""
        try:
            return pickle.dumps(deltas, protocol=pickle.HIGHEST_PROTOCOL), []
        except Exception:
            pass
        kept: dict[str, StateDelta] = {}
        dropped: list[str] = []
        for name, delta in deltas.items():
            try:
                pickle.dumps(delta, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                dropped.append(name)
                continue
            kept[name] = delta
        return pickle.dumps(kept, protocol=pickle.HIGHEST_PROTOCOL), sorted(dropped)

    @staticmethod
    def _decode_deltas(blob: bytes) -> tuple[dict[str, StateDelta], str | None]:
        try:
            deltas = pickle.loads(blob)
        except Exception as exc:
            return {}, str(exc)
        if not isinstance(deltas, dict):
            return {}, f"expected a dict, got {type(deltas).__name__}"
        return deltas, None
