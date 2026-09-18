"""Tests for `src/store.py`: the SQLite mirror, its budget and its migrations."""

import json
import pathlib
import pickle
import sqlite3
import tempfile
import unittest
from unittest import mock

from tests.support import agent, store, tools

# The schema as it stood before the columns in `store.ADDED_COLUMNS` existed,
# so the migration is tested against a file a real older version would write.
OLD_SCHEMA = """
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
    tool_names    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS blocks_parent ON blocks (parent_id);
CREATE INDEX IF NOT EXISTS blocks_created ON blocks (created_at);
"""

OLD_COLUMNS = (
    "id, parent_id, prompt, messages, state_deltas, text, error, dirty, "
    "created_at, endpoint, model, timeout, include_usage, verbose, tool_names"
)


def build(agent_id, parent=None, **fields):
    """A block with the fields a round trip could carry, set explicitly."""
    block = agent.HHAgent.root(
        "http://provider.test",
        "secret-key",
        id=agent_id,
        tools=fields.pop("tools", []),
    )
    block.parent = parent
    block.prompt = fields.pop("prompt", f"prompt-{agent_id}")
    block.messages = fields.pop("messages", [{"role": "user", "content": block.prompt}])
    block.text = fields.pop("text", "")
    block.error = fields.pop("error", None)
    block.outcome = fields.pop("outcome", None)
    block.dirty = fields.pop("dirty", False)
    block.created_at = fields.pop("created_at", block.created_at)
    block.state_deltas = fields.pop("state_deltas", {})
    block.pipe_traces = fields.pop("pipe_traces", [])
    block.summary_model = fields.pop("summary_model", "")
    block.include_usage = fields.pop("include_usage", True)
    block.verbose = fields.pop("verbose", False)
    block.timeout = fields.pop("timeout", 5.0)
    block.max_tokens = fields.pop("max_tokens", agent.DEFAULT_MAX_TOKENS)
    block.model = fields.pop("model", "test-model")
    block.local_timeout = fields.pop("local_timeout", agent.DEFAULT_LOCAL_TIMEOUT)
    if fields:
        raise AssertionError(f"unexpected fields {sorted(fields)}")
    return block


class StoreTestCase(unittest.TestCase):
    """A temp database path plus explicitly managed stores."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hh-store-")
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "harness.db"
        self._stores = []

    def tearDown(self):
        for opened in self._stores:
            opened.close()

    def open_store(self, **kwargs):
        opened = store.HHStore(self.path, **kwargs)
        self._stores.append(opened)
        return opened

    def close_store(self, opened):
        opened.close()
        if opened in self._stores:
            self._stores.remove(opened)

    def raw(self):
        """A plain connection: no foreign keys, no store lock."""
        return sqlite3.connect(self.path)


# --- construction, locking, schema --------------------------------------


class ConstructionTests(StoreTestCase):
    def test_opening_creates_the_file_and_its_lock(self):
        opened = self.open_store()
        self.assertTrue(self.path.exists())
        self.assertTrue(pathlib.Path(f"{self.path}.lock").exists())
        self.assertEqual(opened.max_bytes, 0)

    def test_a_second_store_on_the_same_file_is_refused(self):
        self.open_store()
        with self.assertRaises(store.HHStoreError) as caught:
            store.HHStore(self.path)
        self.assertIn(str(self.path), str(caught.exception))

    def test_the_lock_is_released_when_the_store_closes(self):
        first = self.open_store()
        self.close_store(first)
        second = self.open_store()
        self.assertEqual(second.stats()["blocks"], 0)

    def test_the_schema_is_created_with_every_column(self):
        opened = self.open_store()
        columns = {
            row["name"]
            for row in opened._db.execute("PRAGMA table_info(blocks)")
        }
        for expected in (
            "id", "parent_id", "prompt", "messages", "state_deltas", "text",
            "error", "outcome", "dirty", "created_at", "endpoint", "model",
            "timeout", "max_tokens", "summary_model", "include_usage", "verbose",
            "tool_names", "local_tools",
        ):
            self.assertIn(expected, columns)
        # pipe traces are deliberately not stored, so the column does not exist
        self.assertNotIn("pipe_traces", columns)

    def test_pragmas_are_the_ones_eviction_relies_on(self):
        opened = self.open_store()
        self.assertEqual(opened._db.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertEqual(opened._db.execute("PRAGMA auto_vacuum").fetchone()[0], 1)
        self.assertEqual(
            opened._db.execute("PRAGMA journal_mode").fetchone()[0], "delete"
        )

    def test_max_bytes_is_kept(self):
        self.assertEqual(self.open_store(max_bytes=1024).max_bytes, 1024)

    def test_a_file_written_before_the_new_columns_loads_and_migrates(self):
        connection = sqlite3.connect(self.path)
        with connection:
            connection.executescript(OLD_SCHEMA)
            connection.execute(
                f"INSERT INTO blocks ({OLD_COLUMNS}) VALUES "
                "(?, NULL, ?, ?, ?, ?, NULL, 0, ?, ?, ?, 30.0, 1, 0, ?)",
                (
                    "old",
                    "an older prompt",
                    json.dumps([{"role": "user", "content": "an older prompt"}]),
                    pickle.dumps(
                        {"mem": agent.StateDelta(changed={"k": "v"})},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ),
                    "an older answer",
                    1234.5,
                    "http://provider.test",
                    "old-model",
                    json.dumps(["remember"]),
                ),
            )
        connection.close()

        opened = self.open_store(tools=[tools.ToolEntry("remember", "", [])])
        columns = {row["name"] for row in opened._db.execute("PRAGMA table_info(blocks)")}
        self.assertIn("local_tools", columns)
        self.assertIn("outcome", columns)
        self.assertIn("summary_model", columns)
        self.assertIn("max_tokens", columns)

        blocks, warnings = opened.load("restored-key")
        self.assertEqual(warnings, [])
        self.assertEqual([block.id for block in blocks], ["old"])
        block = blocks[0]
        self.assertEqual(block.prompt, "an older prompt")
        self.assertEqual(block.text, "an older answer")
        self.assertEqual(block.model, "old-model")
        self.assertEqual(block.timeout, 30.0)
        self.assertEqual(block.key, "restored-key")
        self.assertIsNone(block.outcome)
        self.assertEqual(block.summary_model, "")
        self.assertEqual(block.max_tokens, agent.DEFAULT_MAX_TOKENS)
        self.assertEqual(block.merged_state("mem"), {"k": "v"})
        # traces are never restored: a loaded block starts with none
        self.assertEqual(block.pipe_traces, [])

    def test_without_fcntl_the_store_still_works(self):
        # the exclusive lock is POSIX-only; the import fallback must leave the
        # store usable, just without the one-process-per-file guarantee
        with mock.patch.object(store, "fcntl", None):
            first = store.HHStore(self.path)
            first.save(build("a1"))
            second = store.HHStore(self.path)
            blocks, warnings = second.load("k")
            try:
                self.assertEqual([block.id for block in blocks], ["a1"])
                self.assertEqual(warnings, [])
                self.assertFalse(pathlib.Path(f"{self.path}.lock").exists())
            finally:
                first.close()
                second.close()

    def test_migration_is_idempotent(self):
        self.close_store(self.open_store())
        self.close_store(self.open_store())  # a second open must not re-add columns
        opened = self.open_store()
        self.assertEqual(opened._db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0], 0)

    def test_a_file_with_stored_pipe_traces_loses_the_column_and_its_bytes(self):
        # an older version kept a block's pipe traces in a column of their own,
        # which is what this version stopped doing: opening such a file drops
        # the column, and since the values are what made the file big, the
        # freed pages are truncated too
        opened = self.open_store()
        opened.save(build("a1", prompt="hello"))
        self.close_store(opened)

        connection = self.raw()
        with connection:
            connection.execute(
                "ALTER TABLE blocks ADD COLUMN pipe_traces TEXT NOT NULL DEFAULT '[]'"
            )
            connection.execute(
                "UPDATE blocks SET pipe_traces = ?",
                (json.dumps([{"call_id": "c1", "steps": [{"text": "x" * 200_000}]}]),),
            )
        connection.close()
        big = self.path.stat().st_size

        opened = self.open_store()
        columns = {row["name"] for row in opened._db.execute("PRAGMA table_info(blocks)")}
        self.assertNotIn("pipe_traces", columns)
        self.assertLess(opened.size_bytes(), big)
        blocks, warnings = opened.load("restored-key")
        self.assertEqual(warnings, [])
        self.assertEqual([block.id for block in blocks], ["a1"])
        self.assertEqual(blocks[0].prompt, "hello")


# --- saving and loading --------------------------------------------------


class SaveLoadTests(StoreTestCase):
    def test_round_trip_keeps_every_field(self):
        state = agent.StateDelta(changed={"colour": "blue", "n": 3}, removed=("old",))
        original = build(
            "a1",
            prompt="hello",
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            text="hi",
            outcome="ok",
            dirty=False,
            created_at=99.5,
            state_deltas={"mem": state},
            # on the block, but not something the store is allowed to keep
            pipe_traces=[{"call_id": "c1", "chain": ["fetch", "write_file"], "steps": []}],
            summary_model="summariser",
            include_usage=False,
            verbose=True,
            model="other-model",
            timeout=12.5,
            max_tokens=1234,
            tools=[tools.ToolEntry("remember", "d", [])],
        )
        store = self.open_store(tools=[tools.ToolEntry("remember", "d", [])])
        self.assertEqual(store.save(original), [])

        blocks, warnings = store.load("server-key")
        self.assertEqual(warnings, [])
        self.assertEqual(len(blocks), 1)
        loaded = blocks[0]
        self.assertEqual(loaded.id, "a1")
        self.assertIsNone(loaded.parent)
        self.assertEqual(loaded.prompt, "hello")
        self.assertEqual(loaded.messages, original.messages)
        self.assertEqual(loaded.text, "hi")
        self.assertIsNone(loaded.error)
        self.assertEqual(loaded.outcome, "ok")
        self.assertFalse(loaded.dirty)
        self.assertEqual(loaded.created_at, 99.5)
        self.assertEqual(loaded.endpoint, "http://provider.test")
        self.assertEqual(loaded.model, "other-model")
        self.assertEqual(loaded.timeout, 12.5)
        self.assertEqual(loaded.max_tokens, 1234)
        self.assertEqual(loaded.summary_model, "summariser")
        self.assertFalse(loaded.include_usage)
        self.assertTrue(loaded.verbose)
        self.assertEqual(loaded.key, "server-key")
        self.assertEqual(sorted(loaded.tools), ["remember"])
        self.assertEqual(loaded.state_deltas, {"mem": state})
        # the trace was on the original block but is not persisted at all
        self.assertEqual(loaded.pipe_traces, [])
        # the local timeout is not persisted; blocks fall back to the default
        self.assertEqual(loaded.local_timeout, agent.DEFAULT_LOCAL_TIMEOUT)

    def test_saving_the_same_block_twice_refreshes_every_column(self):
        remembered = tools.ToolEntry("remember", "d", [], hook=lambda c: "")
        other = tools.ToolEntry("other", "d", [], hook=lambda c: "")
        ask = tools.ToolEntry("ask_operator", "d", [], hook=None)
        block = build("a1", tools=[remembered], prompt="first")
        store_ = self.open_store(tools=[remembered, other, ask])
        store_.save(block)

        # change every column `save` is supposed to refresh: one left out of
        # `UPDATABLE` would silently keep its first value for ever
        block.prompt = "a new prompt"
        block.messages = [{"role": "assistant", "content": "changed"}]
        block.state_deltas = {"mem": agent.StateDelta(changed={"k": "v"}, removed=("gone",))}
        block.text = "the new text"
        block.error = "the new error"
        block.outcome = "failed"
        block.dirty = True
        block.created_at = 4321.0
        block.endpoint = "http://elsewhere.test"
        block.model = "another-model"
        block.timeout = 3.5
        block.max_tokens = 4321
        block.summary_model = "summariser"
        block.include_usage = False
        block.verbose = True
        block.tools = {"other": other, "ask_operator": ask}
        self.assertEqual(store_.save(block), [])

        with store_._lock:
            rows = store_._db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        self.assertEqual(rows, 1)  # an upsert, not a second row
        restored, warnings = store_.load("server-key")
        self.assertEqual(warnings, [])
        block = restored[0]
        self.assertEqual(block.prompt, "a new prompt")
        self.assertEqual(block.messages, [{"role": "assistant", "content": "changed"}])
        self.assertEqual(
            block.state_deltas,
            {"mem": agent.StateDelta(changed={"k": "v"}, removed=("gone",))},
        )
        self.assertEqual(block.text, "the new text")
        self.assertEqual(block.error, "the new error")
        self.assertEqual(block.outcome, "failed")
        self.assertTrue(block.dirty)
        self.assertEqual(block.created_at, 4321.0)
        self.assertEqual(block.endpoint, "http://elsewhere.test")
        self.assertEqual(block.model, "another-model")
        self.assertEqual(block.timeout, 3.5)
        self.assertEqual(block.max_tokens, 4321)
        self.assertEqual(block.summary_model, "summariser")
        self.assertFalse(block.include_usage)
        self.assertTrue(block.verbose)
        self.assertEqual(sorted(block.tools), ["ask_operator", "other"])
        self.assertTrue(block.tools["ask_operator"].is_local)

    def test_a_child_cannot_be_written_before_its_parent(self):
        # `save` inserts the parent id as a foreign key, so blocks go in
        # parent-first order — the server always writes a fork after its parent
        root = build("root", created_at=1.0)
        child = build("child", parent=root, created_at=2.0)
        store_ = self.open_store()
        with self.assertRaises(sqlite3.IntegrityError):
            store_.save(child)
        self.assertEqual(store_.stats()["blocks"], 0)

    def test_children_are_linked_and_ordered(self):
        root = build("root", created_at=1.0)
        first = build("first", parent=root, created_at=2.0)
        second = build("second", parent=first, created_at=3.0)
        store_ = self.open_store()
        for block in (root, first, second):
            store_.save(block)

        blocks, warnings = store_.load("k")
        self.assertEqual(warnings, [])
        self.assertEqual([block.id for block in blocks], ["root", "first", "second"])
        by_id = {block.id: block for block in blocks}
        self.assertIsNone(by_id["root"].parent)
        self.assertIs(by_id["first"].parent, by_id["root"])
        self.assertIs(by_id["second"].parent, by_id["first"])

    def test_local_tools_are_stored_whole(self):
        ask = tools.ToolEntry(
            name="ask_operator",
            description="Ask the operator.",
            params=[tools.ToolParam("question", "string", "what to ask")],
            hook=None,
            remote_rollback=True,
            external_effects=True,
        )
        store = self.open_store(tools=[ask])
        store.save(build("a1", tools=[ask]))
        loaded, warnings = store.load("k")
        self.assertEqual(warnings, [])
        restored = loaded[0].tools["ask_operator"]
        self.assertTrue(restored.is_local)
        self.assertEqual(restored.description, "Ask the operator.")
        self.assertEqual(restored.params, [tools.ToolParam("question", "string", "what to ask")])
        self.assertTrue(restored.remote_rollback)
        self.assertTrue(restored.has_rollback)
        self.assertTrue(restored.external_effects)

    def test_a_server_tool_that_no_longer_exists_is_reported(self):
        gone = tools.ToolEntry("gone", "d", [], hook=lambda context: "x")
        store_ = self.open_store(tools=[])
        store_.save(build("a1", tools=[gone]))
        blocks, warnings = store_.load("k")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tools, {})
        self.assertEqual(warnings, ["a1: missing tools ['gone']"])

    def test_a_server_tool_is_resolved_by_name_against_the_catalogue(self):
        def hook(context):
            return "x"

        remembered = tools.ToolEntry("remember", "d", [], hook=hook)
        store_ = self.open_store(tools=[remembered])
        store_.save(build("a1", tools=[remembered]))
        blocks, warnings = store_.load("k")
        self.assertEqual(warnings, [])
        self.assertIs(blocks[0].tools["remember"], remembered)

    def test_the_python_key_never_reaches_the_file(self):
        store = self.open_store()
        store.save(build("a1"))
        connection = self.raw()
        rows = connection.execute("SELECT * FROM blocks").fetchall()
        connection.close()
        self.assertEqual(len(rows), 1)
        self.assertNotIn("secret-key", str(rows[0]))


# --- damaged rows --------------------------------------------------------


class DamageTests(StoreTestCase):
    def test_unreadable_messages_are_emptied_but_the_block_is_kept(self):
        store = self.open_store()
        store.save(build("a1"))
        self.close_store(store)
        connection = self.raw()
        with connection:
            connection.execute("UPDATE blocks SET messages = ? WHERE id = ?", ("{oops", "a1"))
        connection.close()

        reopened = self.open_store()
        blocks, warnings = reopened.load("k")
        self.assertEqual([block.id for block in blocks], ["a1"])
        self.assertEqual(blocks[0].messages, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("messages unreadable", warnings[0])

    def test_unreadable_state_is_emptied_but_the_block_is_kept(self):
        store = self.open_store()
        store.save(build("a1"))
        self.close_store(store)
        connection = self.raw()
        with connection:
            connection.execute("UPDATE blocks SET state_deltas = ? WHERE id = ?", (b"junk", "a1"))
        connection.close()

        reopened = self.open_store()
        blocks, warnings = reopened.load("k")
        self.assertEqual(blocks[0].state_deltas, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("state unreadable", warnings[0])

    def test_state_that_is_not_a_dict_is_reported(self):
        store = self.open_store()
        store.save(build("a1"))
        self.close_store(store)
        connection = self.raw()
        with connection:
            connection.execute(
                "UPDATE blocks SET state_deltas = ? WHERE id = ?",
                (pickle.dumps(["not", "a", "dict"]), "a1"),
            )
        connection.close()

        reopened = self.open_store()
        blocks, warnings = reopened.load("k")
        self.assertEqual(blocks[0].state_deltas, {})
        self.assertIn("expected a dict, got list", warnings[0])

    def test_a_row_whose_messages_are_broken_still_holds_its_children(self):
        # the loader keeps a broken row rather than dropping it: dropping it
        # would orphan its descendants, and an orphan loads as a root, which
        # would rewrite that chain's context
        root = build("root", created_at=1.0)
        child = build("child", parent=root, created_at=2.0)
        grandchild = build("grand", parent=child, created_at=3.0)
        store_ = self.open_store()
        for block in (root, child, grandchild):
            store_.save(block)
        self.close_store(store_)

        connection = self.raw()
        with connection:
            connection.execute("UPDATE blocks SET messages = ? WHERE id = ?", ("{oops", "child"))
        connection.close()

        reopened = self.open_store()
        blocks, warnings = reopened.load("k")
        self.assertEqual([block.id for block in blocks], ["root", "child", "grand"])
        self.assertIs(blocks[2].parent, blocks[1])  # the chain survives intact
        self.assertEqual(blocks[1].messages, [])
        self.assertIn("child: messages unreadable", warnings[0])

    def test_a_dangling_parent_makes_the_block_a_root(self):
        root = build("root", created_at=1.0)
        child = build("child", parent=root, created_at=2.0)
        grandchild = build("grand", parent=child, created_at=3.0)
        store_ = self.open_store()
        for block in (root, child, grandchild):
            store_.save(block)
        self.close_store(store_)

        connection = self.raw()
        with connection:
            connection.execute("DELETE FROM blocks WHERE id = ?", ("child",))
        connection.close()

        reopened = self.open_store()
        blocks, warnings = reopened.load("k")
        self.assertEqual([block.id for block in blocks], ["root", "grand"])
        self.assertIsNone(blocks[1].parent)
        self.assertEqual(warnings, ["grand: parent child is gone, treated as root"])


# --- state deltas encoding ----------------------------------------------


class DeltaEncodingTests(StoreTestCase):
    def test_a_namespace_that_cannot_be_pickled_is_dropped_alone(self):
        good = agent.StateDelta(changed={"ok": 1})
        bad = agent.StateDelta(changed={"fn": lambda: None})
        blob, dropped = store.HHStore._encode_deltas({"good": good, "bad": bad})
        self.assertEqual(dropped, ["bad"])
        self.assertEqual(pickle.loads(blob), {"good": good})

    def test_encoded_deltas_decode_unchanged(self):
        deltas = {
            "a": agent.StateDelta(changed={"x": [1, 2]}, removed=("y",)),
            "b": agent.StateDelta(),
        }
        blob, dropped = store.HHStore._encode_deltas(deltas)
        self.assertEqual(dropped, [])
        decoded, error = store.HHStore._decode_deltas(blob)
        self.assertIsNone(error)
        self.assertEqual(decoded, deltas)

    def test_a_save_that_drops_state_reports_it(self):
        store_ = self.open_store()
        block = build("a1")
        block.state_deltas = {
            "keepme": agent.StateDelta(changed={"n": 1}),
            "dropme": agent.StateDelta(changed={"fn": lambda: None}),
        }
        self.assertEqual(store_.save(block), ["dropme"])
        loaded, _ = store_.load("k")
        self.assertEqual(
            loaded[0].state_deltas, {"keepme": agent.StateDelta(changed={"n": 1})}
        )

    def test_decode_refuses_a_non_dict(self):
        deltas, error = store.HHStore._decode_deltas(pickle.dumps(7))
        self.assertEqual(deltas, {})
        self.assertEqual(error, "expected a dict, got int")

    def test_decode_reports_a_broken_blob(self):
        deltas, error = store.HHStore._decode_deltas(b"not a pickle")
        self.assertEqual(deltas, {})
        self.assertTrue(error)


# --- deletion, subtrees, candidates -------------------------------------


class SubtreeTests(StoreTestCase):
    def _chain(self, store_):
        """root → a → b, and root → c."""
        root = build("root", created_at=1.0)
        a = build("a", parent=root, created_at=2.0)
        b = build("b", parent=a, created_at=3.0)
        c = build("c", parent=root, created_at=4.0)
        for block in (root, a, b, c):
            store_.save(block)
        return root, a, b, c

    def test_subtree_ids_include_the_root(self):
        store_ = self.open_store()
        self._chain(store_)
        self.assertEqual(sorted(store_.subtree_ids("a")), ["a", "b"])
        self.assertEqual(sorted(store_.subtree_ids("c")), ["c"])
        self.assertEqual(
            sorted(store_.subtree_ids("root")), ["a", "b", "c", "root"]
        )

    def test_delete_cascades_to_children(self):
        store_ = self.open_store()
        self._chain(store_)
        self.assertEqual(store_.delete("a"), 1)  # the row itself
        self.assertEqual(sorted(store_.subtree_ids("root")), ["c", "root"])

    def test_deleting_a_missing_block_removes_nothing(self):
        store_ = self.open_store()
        self.assertEqual(store_.delete("nobody"), 0)

    def test_candidates_of_a_plain_chain_are_all_or_nothing(self):
        store_ = self.open_store()
        root = build("root", created_at=1.0)
        a = build("a", parent=root, created_at=2.0)
        b = build("b", parent=a, created_at=3.0)
        for block in (root, a, b):
            store_.save(block)
        candidates = store_.candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["root"], "root")
        self.assertEqual(candidates[0]["nodes"], 3)
        self.assertEqual(candidates[0]["newest"], 3.0)

    def test_a_fork_makes_both_the_root_and_the_branch_reclaimable(self):
        store_ = self.open_store()
        root, a, b, c = self._chain(store_)
        candidates = store_.candidates()
        by_root = {row["root"]: row for row in candidates}
        self.assertEqual(set(by_root), {"root", "a", "c"})
        self.assertEqual(by_root["root"]["nodes"], 4)  # the whole tree
        self.assertEqual(by_root["a"]["nodes"], 2)  # a and its child
        self.assertEqual(by_root["c"]["nodes"], 1)
        self.assertEqual(by_root["a"]["newest"], 3.0)

    def test_the_limit_is_recomputed_after_each_eviction(self):
        # once a branch is the last survivor under its parent, that parent has no
        # forks left, so the branch is only reclaimable together with its tree
        store_ = self.open_store()
        self._chain(store_)  # root → a → b, and root → c
        self.assertEqual(
            {row["root"] for row in store_.candidates()}, {"root", "a", "c"}
        )
        store_.delete("c")
        candidates = store_.candidates()
        self.assertEqual({row["root"] for row in candidates}, {"root"})
        self.assertEqual(candidates[0]["nodes"], 3)

    def test_candidates_are_oldest_first_by_their_newest_block(self):
        store_ = self.open_store()
        older = build("older", created_at=1.0)
        newer = build("newer", created_at=10.0)
        store_.save(older)
        store_.save(newer)
        self.assertEqual(
            [row["root"] for row in store_.candidates()], ["older", "newer"]
        )

    def test_a_busy_conversation_stays_young(self):
        # the age of a subtree is its newest block, so a chain that keeps
        # growing sorts after a quiet one and is evicted last
        store_ = self.open_store()
        store_.save(build("quiet", created_at=1.0))
        root = build("root", created_at=2.0)
        store_.save(root)
        newest = root
        for index in range(3):
            newest = build(f"block-{index}", parent=newest, created_at=3.0 + index)
            store_.save(newest)
        candidates = store_.candidates()
        self.assertEqual([row["root"] for row in candidates], ["quiet", "root"])
        self.assertEqual(candidates[1]["nodes"], 4)
        self.assertEqual(candidates[1]["newest"], 5.0)


# --- size and stats ------------------------------------------------------


class SizeTests(StoreTestCase):
    def test_size_is_pages_times_page_size(self):
        store_ = self.open_store()
        with store_._lock:
            pages = store_._db.execute("PRAGMA page_count").fetchone()[0]
            page_size = store_._db.execute("PRAGMA page_size").fetchone()[0]
        self.assertEqual(store_.size_bytes(), pages * page_size)
        self.assertGreater(store_.size_bytes(), 0)

    def test_eviction_can_shrink_the_file(self):
        store_ = self.open_store()
        store_.save(build("a", prompt="x" * 100_000))
        big = store_.size_bytes()
        store_.delete("a")
        self.assertLess(store_.size_bytes(), big)

    def test_stats_describes_the_store(self):
        store_ = self.open_store(max_bytes=4096)
        root = build("root", created_at=1.0)
        store_.save(root)
        store_.save(build("child", parent=root, created_at=2.0))
        stats = store_.stats()
        self.assertEqual(stats["path"], str(self.path))
        self.assertEqual(stats["blocks"], 2)
        self.assertEqual(stats["trees"], 1)  # one chain: one reclaimable subtree
        self.assertEqual(stats["max_bytes"], 4096)
        self.assertEqual(stats["bytes"], store_.size_bytes())

    def test_stats_counts_a_fork_as_a_second_tree(self):
        store_ = self.open_store()
        root = build("root", created_at=1.0)
        store_.save(root)
        store_.save(build("left", parent=root, created_at=2.0))
        store_.save(build("right", parent=root, created_at=3.0))
        self.assertEqual(store_.stats()["trees"], 3)  # the root, and each branch
