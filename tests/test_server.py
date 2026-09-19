"""Tests for `src/server.py`: the wire protocol, the registry, persistence.

These run a real `HHServer` in-process on an ephemeral port with a real
(temporary) SQLite store, and drive it through `tests.support.Client`. The
provider is the scripted fake from `tests/support.py`, so turns complete without
a network or a key.
"""

import contextlib
import io
import json
import pathlib
import pickle
import socket
import sqlite3
import struct
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from tests.support import (
    HHTestCase,
    Response,
    TEST_MODEL,
    agent,
    local_tool,
    pick,
    sandbox_tools,
    server,
    store,
    tools,
)


def seed_database(path, blocks):
    """Write blocks straight into a store file, then release the lock."""
    opened = store.HHStore(path)
    try:
        for block in blocks:
            opened.save(block)
    finally:
        opened.close()


def block_with(agent_id, **fields):
    """A block for seeding a database, with only the fields given set."""
    block = agent.HHAgent.root(
        fields.pop("endpoint", "http://provider.test"),
        fields.pop("key", "seeded-key"),
        id=agent_id,
        tools=fields.pop("tools", []),
    )
    for name, value in fields.items():
        setattr(block, name, value)
    return block


class RecordingConnection:
    """A stand-in for `server.Connection` that keeps what was sent to it."""

    def __init__(self):
        self.events = []
        self.closed = False

    def send(self, event, **fields):
        self.events.append({"event": event, **fields})
        return True


class ServerTestCase(HHTestCase):
    """Helpers for driving commands and reading their events."""

    with_store = False
    max_db_bytes = 0

    def create(self, client, agent_id, **fields):
        mark = client.mark()
        client.command("create_agent", id=agent_id, **fields)
        return client.wait_event("agent_created", since=mark, agent_id=agent_id)

    def fork(self, client, parent, prompt, new_id=None, **fields):
        mark = client.mark()
        payload = {"id": parent, "prompt": prompt, **fields}
        if new_id is not None:
            payload["new_id"] = new_id
        client.command("fork", **payload)
        return client.wait_event("agent_forked", since=mark, parent=parent)["agent_id"]

    def run_block(self, client, agent_id, timeout=20.0):
        """Send `run` and wait for the turn's own `command_finished`."""
        mark = client.mark()
        rid = f"run-{agent_id}-{mark}"
        client.send("run", rid=rid, id=agent_id)
        accepted = client.wait_event("run_accepted", since=mark, rid=rid, timeout=timeout)
        finished = client.wait_event(
            "command_finished", since=mark, rid=rid, timeout=timeout
        )
        return accepted, finished

    def fetch_block(self, fixture, agent_id):
        """The live block, for asserting on internals."""
        with fixture.server._registry_lock:
            return fixture.server._agents[agent_id]


# --- framing, session, lifecycle ----------------------------------------


class SessionTests(ServerTestCase):
    def test_session_hello_describes_the_server(self):
        fixture = self.start_server()
        client = fixture.client()
        hello = client.wait_event("session_hello")
        self.assertEqual(hello["protocol"], server.PROTOCOL_VERSION)
        self.assertEqual(hello["server"], "headless-harness")
        self.assertEqual(hello["peer"][0], "127.0.0.1")
        self.assertIsInstance(hello["peer"][1], int)
        self.assertGreater(hello["started_at"], 0)
        self.assertEqual(hello["store"], None)
        self.assertEqual(
            hello["defaults"],
            {"endpoint": self.provider.endpoint, "model": TEST_MODEL, "has_key": True},
        )
        self.assertNotIn("key", hello["defaults"])
        self.assertEqual(
            [entry["command"] for entry in hello["commands"]],
            [
                "create_agent", "fork", "run", "cancel", "get_context",
                "list_agents", "destroy_agent", "get_state", "set_state",
                "resolve_tool", "ping",
            ],
        )
        self.assertEqual(
            [tool["name"] for tool in hello["tools"]],
            [
                "get_system_info",
                "get_current_time",
                "set_magic_number",
                "get_magic_number",
                "web_fetch",
                "web_search",
                "nix_spawn_sandbox",
                "nix_sandbox_status",
                "nix_add_dependency",
                "nix_remove_dependency",
                "nix_exec",
                "nix_add_file",
                "nix_cat_file",
                "nix_destroy_sandbox",
            ],
        )
        setter = hello["tools"][2]
        self.assertEqual(setter["state_namespace"], "magic")
        self.assertFalse(setter["rollback"])
        self.assertEqual(
            setter["params"],
            [{"name": "magic", "type": "string", "description": "the new magic number"}],
        )

    def test_session_hello_carries_store_stats_and_warnings(self):
        fixture = self.start_server(with_store=True)
        hello = fixture.client().wait_event("session_hello")
        info = hello["store"]
        self.assertEqual(info["path"], str(fixture.db_path))
        self.assertEqual(info["blocks"], 0)
        self.assertEqual(info["trees"], 0)
        self.assertGreater(info["bytes"], 0)
        self.assertEqual(info["max_bytes"], 0)
        self.assertEqual(info["warnings"], [])

    def test_defaults_hide_the_key_when_there_is_none(self):
        fixture = self.start_server(defaults={})
        hello = fixture.client().wait_event("session_hello")
        self.assertEqual(
            hello["defaults"], {"endpoint": None, "model": None, "has_key": False}
        )

    def test_ping_reports_liveness_and_counts(self):
        fixture = self.start_server()
        client = fixture.client()
        client.command("create_agent", id="root", endpoint=self.provider.endpoint, key="k")
        client.command("fork", id="root", prompt="hi", new_id="child")
        before = time.time()
        reply = client.command("ping", echo={"any": "thing"})
        self.assertEqual(reply["status"], "ok")
        pong = client.wait_event("pong")
        self.assertEqual(pong["echo"], {"any": "thing"})
        self.assertGreaterEqual(pong["server_time"], before)
        self.assertGreaterEqual(pong["uptime_ms"], 0)
        self.assertEqual(pong["agents"], 2)
        self.assertEqual(pong["dirty"], 1)  # the fork has not run yet
        self.assertEqual(pong["running"], 0)
        self.assertGreater(pong["threads"], 0)
        self.assertIsNone(pong["store"])

    def test_every_command_is_bracketed(self):
        fixture = self.start_server()
        client = fixture.client()
        rid = "r-1"
        client.send("ping", rid=rid)
        received = client.wait_event("command_received", rid=rid)
        self.assertEqual(received["command"], "ping")
        self.assertEqual(received["keys"], ["command", "rid"])
        self.assertIsNone(received["target"])
        finished = client.wait_event("command_finished", rid=rid)
        self.assertEqual(finished["command"], "ping")
        self.assertEqual(finished["status"], "ok")
        self.assertIsNone(finished["target"])
        self.assertGreaterEqual(finished["elapsed_ms"], 0)

    def test_a_command_names_its_target(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.send("get_context", rid="ctx", id="root")
        self.assertEqual(client.wait_event("command_received", rid="ctx")["target"], "root")

    def test_rids_are_echoed_so_turns_can_be_told_apart(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.text("one")
        _, finished = self.run_block(client, "a1")
        self.assertTrue(finished["rid"].startswith("run-a1-"))
        self.assertEqual(finished["agent_id"], "a1")

    def test_seq_is_monotonic_and_ts_is_present(self):
        fixture = self.start_server()
        client = fixture.client()
        client.command("ping")
        client.command("ping")
        seqs = [event["seq"] for event in client.events]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))
        self.assertTrue(all(event["ts"] > 0 for event in client.events))

    def test_each_connection_has_its_own_seq(self):
        fixture = self.start_server()
        first = fixture.client()
        first.command("ping")
        second = fixture.client()
        hello = second.wait_event("session_hello")
        self.assertEqual(hello["seq"], 1)

    def test_blank_lines_are_ignored(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send_line("")
        client.send_line("   ")
        client.command("ping")
        self.assertEqual(pick(client.events, "error"), [])

    def test_bad_json_is_reported_and_the_connection_survives(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send_line("{not json")
        error = client.wait_event("error")
        self.assertEqual(error["code"], "bad_json")
        self.assertIn("line", error)
        client.command("ping")  # still usable

    def test_a_line_that_is_not_an_object(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send_line("[1, 2, 3]")
        error = client.wait_event("error")
        self.assertEqual(error["code"], "bad_command")
        self.assertEqual(error["got"], "list")

    def test_command_must_be_a_string(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send_line(json.dumps({"rid": "r1"}))
        error = client.wait_event("error", rid="r1")
        self.assertEqual(error["code"], "bad_command")
        self.assertIn("create_agent", error["commands"])

    def test_unknown_command_lists_the_valid_ones(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send("levitate", rid="r1")
        error = client.wait_error("r1", code="unknown_command")
        self.assertIn("levitate", error["message"])
        self.assertIn("ping", error["commands"])

    def test_one_command_line_fits_the_largest_nix_add_file_payload(self):
        # a client pipes a file in by answering resolve_tool with a call, so the
        # biggest file nix_add_file takes has to fit in one line: base64 is
        # ceil(n/3)*4 bytes, plus the small JSON envelope around it
        encoded = (sandbox_tools.ADD_FILE_MAX_BYTES + 2) // 3 * 4
        self.assertGreater(server.MAX_COMMAND_BYTES - encoded, 4096)

    def test_an_oversized_command_is_refused_and_closes_the_connection(self):
        fixture = self.start_server()
        client = fixture.client()
        with mock.patch.object(server, "MAX_COMMAND_BYTES", 64):
            client.send_line(json.dumps({"command": "ping", "echo": "x" * 200}))
            error = client.wait_event("error")
            self.assertEqual(error["code"], "command_too_large")
            self.assertGreater(error["bytes"], 64)
            client.wait_event("session_closing")
        self.wait_until(lambda: client.closed, description="the socket to close")

    def test_a_broken_property_produces_an_internal_error(self):
        fixture = self.start_server()
        client = fixture.client()

        def explode(conn, command, rid):
            raise RuntimeError("a bug in the handler")

        fixture.server.cmd_ping = explode
        client.send("ping", rid="r1")
        error = client.wait_error("r1", code="internal_error")
        self.assertIn("RuntimeError: a bug in the handler", error["message"])
        self.assertIn("traceback", error)

    def test_a_peer_that_resets_the_connection_does_not_stop_the_server(self):
        # a peer that resets rather than closes politely is not a bug to report:
        # the handler's own guard swallows it and other connections carry on
        fixture = self.start_server()
        client = fixture.client()
        client.wait_event("session_hello")
        # close with a reset instead of a polite FIN, so the handler's next read
        # raises rather than being told the peer is finished
        client.sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        client.sock.close()
        other = fixture.client()
        self.assertEqual(other.command("ping")["status"], "ok")

    def test_session_closing_reports_what_the_connection_did(self):
        fixture = self.start_server()
        client = fixture.client()
        client.command("ping")
        client.half_close()
        closing = client.wait_event("session_closing")
        self.assertEqual(closing["commands"], 1)
        self.assertGreaterEqual(closing["events_sent"], 3)
        self.wait_until(lambda: client.closed, description="the socket to close")

    def test_a_rejected_command_sends_no_command_finished(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send("get_context", rid="r1", id="nobody")
        client.wait_error("r1", code="unknown_agent")
        client.settle()
        self.assertEqual(client.named("command_finished"), [])


# --- create_agent --------------------------------------------------------


class CreateAgentTests(ServerTestCase):
    def test_defaults_come_from_the_server(self):
        fixture = self.start_server()
        client = fixture.client()
        created = self.create(client, "root")
        self.assertEqual(created["parent"], None)
        self.assertEqual(created["depth"], 0)
        self.assertFalse(created["dirty"])
        self.assertEqual(created["model"], TEST_MODEL)
        self.assertEqual(created["endpoint"], self.provider.endpoint)
        self.assertEqual(created["timeout"], agent.DEFAULT_TIMEOUT)
        self.assertEqual(created["summary_model"], "")
        self.assertEqual(created["max_tokens"], agent.DEFAULT_MAX_TOKENS)
        self.assertEqual(created["context_window"], agent.DEFAULT_CONTEXT_WINDOW)
        self.assertTrue(created["include_usage"])
        self.assertFalse(created["verbose"])
        self.assertEqual(created["local_tools"], [])
        self.assertEqual(
            sorted(created["tools"]),
            [
                "get_current_time",
                "get_magic_number",
                "get_system_info",
                "nix_add_dependency",
                "nix_add_file",
                "nix_cat_file",
                "nix_destroy_sandbox",
                "nix_exec",
                "nix_remove_dependency",
                "nix_sandbox_status",
                "nix_spawn_sandbox",
                "set_magic_number",
                "web_fetch",
                "web_search",
            ],
        )
        self.assertEqual(len(created["tool_schemas"]), 14)

    def test_an_id_is_generated_when_omitted(self):
        fixture = self.start_server()
        client = fixture.client()
        client.command("create_agent")
        created = client.wait_event("agent_created")
        self.assertTrue(created["agent_id"].startswith("agent-"))

    def test_bad_ids_are_refused(self):
        fixture = self.start_server()
        client = fixture.client()
        for bad in ("", 5, {"a": 1}):
            client.send("create_agent", rid="r", id=bad)
            client.wait_error("r", code="bad_id")

    def test_missing_credentials_are_refused_with_a_hint(self):
        fixture = self.start_server(defaults={})
        client = fixture.client()
        client.send("create_agent", rid="r1")
        error = client.wait_error("r1", code="missing_credentials")
        self.assertIn("--endpoint", error["message"])
        # supplying both on the command works without server defaults
        created = self.create(
            client, "root", endpoint=self.provider.endpoint, key="local-key"
        )
        self.assertEqual(created["endpoint"], self.provider.endpoint)

    def test_a_duplicate_id_is_refused(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.send("create_agent", rid="r1", id="root")
        error = client.wait_error("r1", code="duplicate_agent")
        self.assertEqual(error["detail"]["agents"], ["root"])

    def test_a_tool_subset_can_be_requested(self):
        fixture = self.start_server()
        client = fixture.client()
        created = self.create(client, "root", tools=["get_magic_number"])
        self.assertEqual(created["tools"], ["get_magic_number"])
        self.assertEqual(len(created["tool_schemas"]), 1)

    def test_an_empty_tool_list_is_allowed(self):
        fixture = self.start_server()
        client = fixture.client()
        created = self.create(client, "root", tools=[])
        self.assertEqual(created["tools"], [])
        self.assertEqual(created["tool_schemas"], [])

    def test_an_unknown_tool_lists_the_catalogue(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send("create_agent", rid="r1", id="root", tools=["nope"])
        error = client.wait_error("r1", code="unknown_tool")
        self.assertIn("get_current_time", error["detail"]["available"])

    def test_tools_must_be_a_list(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send("create_agent", rid="r1", id="root", tools="get_current_time")
        client.wait_error("r1", code="bad_tools")

    def test_local_tools_are_declared_and_offered(self):
        fixture = self.start_server()
        client = fixture.client()
        created = self.create(
            client,
            "root",
            tools=[],
            local_tools=[
                {
                    "name": "ask_operator",
                    "description": "Ask the operator.",
                    "params": [{"name": "question", "description": "what to ask"}],
                    "rollback": True,
                    "external_effects": True,
                }
            ],
        )
        self.assertEqual(created["local_tools"], ["ask_operator"])
        self.assertEqual(created["tools"], ["ask_operator"])
        schema = created["tool_schemas"][0]["function"]
        self.assertEqual(schema["name"], "ask_operator")
        self.assertEqual(
            schema["parameters"]["properties"],
            {"question": {"type": "string", "description": "what to ask"}},
        )
        self.assertEqual(schema["parameters"]["required"], ["question"])

        listed = client.command("list_agents")
        self.assertEqual(listed["status"], "ok")
        entry = client.wait_event("agents_listed")["agents"][0]
        self.assertEqual(entry["local_tools"], ["ask_operator"])
        self.assertEqual(entry["rollback_tools"], ["ask_operator"])

    def test_local_tool_definitions_are_validated(self):
        fixture = self.start_server()
        client = fixture.client()
        cases = [
            {"local_tools": "nope"},
            {"local_tools": ["nope"]},
            {"local_tools": [{"description": "no name"}]},
            {"local_tools": [{"name": ""}]},
            {"local_tools": [{"name": "t", "description": 5}]},
            {"local_tools": [{"name": "t", "params": "nope"}]},
            {"local_tools": [{"name": "t", "params": ["nope"]}]},
            {"local_tools": [{"name": "t", "params": [{"description": "no name"}]}]},
            {"local_tools": [{"name": "t", "rollback": "yes"}]},
            {"local_tools": [{"name": "t", "external_effects": "yes"}]},
        ]
        for index, fields in enumerate(cases):
            rid = f"r{index}"
            client.send("create_agent", rid=rid, id="root", **fields)
            error = client.wait_error(rid, code="bad_tools")
            self.assertIn("local_tools", error["message"])

    def test_a_local_tool_cannot_shadow_a_builtin(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send(
            "create_agent",
            rid="r1",
            id="root",
            local_tools=[{"name": "get_current_time"}],
        )
        error = client.wait_error("r1", code="bad_tools")
        self.assertIn("declared twice", error["message"])

    def test_timeouts_are_validated_and_zero_means_default(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root", timeout=30, local_timeout="2.5")
        block = self.fetch_block(fixture, "root")
        self.assertEqual(block.timeout, 30.0)
        self.assertEqual(block.local_timeout, 2.5)

        # a zero is falsy, so it falls back to the default rather than erroring
        self.create(client, "root2", timeout=0, local_timeout=0)
        block = self.fetch_block(fixture, "root2")
        self.assertEqual(block.timeout, agent.DEFAULT_TIMEOUT)
        self.assertEqual(block.local_timeout, agent.DEFAULT_LOCAL_TIMEOUT)

        cases = [
            (field, bad)
            for field in ("timeout", "local_timeout")
            for bad in (-1, "abc")
        ]
        for index, (field, bad) in enumerate(cases):
            rid = f"r{index}"
            client.send("create_agent", rid=rid, id=f"bad{index}", **{field: bad})
            client.wait_error(rid, code="bad_field")

    def test_include_usage_and_verbose_are_coerced_to_booleans(self):
        fixture = self.start_server()
        client = fixture.client()
        created = self.create(client, "root", include_usage=0, verbose=1)
        self.assertFalse(created["include_usage"])
        self.assertTrue(created["verbose"])

    def test_max_tokens_is_coerced_and_may_be_zero(self):
        fixture = self.start_server()
        client = fixture.client()
        created = self.create(client, "root", max_tokens="4096")
        self.assertEqual(created["max_tokens"], 4096)
        # 0 is meaningful here: the cap is dropped from the request entirely
        created = self.create(client, "root0", max_tokens=0)
        self.assertEqual(created["max_tokens"], 0)
        for index, bad in enumerate((-1, 1.5, "soon", True)):
            rid = f"r{index}"
            client.send("create_agent", rid=rid, id=f"bad{index}", max_tokens=bad)
            client.wait_error(rid, code="bad_field")

    def test_context_window_is_coerced_and_may_be_zero(self):
        fixture = self.start_server()
        client = fixture.client()
        created = self.create(client, "root", context_window="4096")
        self.assertEqual(created["context_window"], 4096)
        # 0 is meaningful here: the whole chain goes out, untruncated
        created = self.create(client, "root0", context_window=0)
        self.assertEqual(created["context_window"], 0)
        for index, bad in enumerate((-1, 1.5, "soon", True)):
            rid = f"w{index}"
            client.send("create_agent", rid=rid, id=f"bad{index}", context_window=bad)
            client.wait_error(rid, code="bad_field")

    def test_a_summary_model_can_be_set(self):
        fixture = self.start_server()
        created = self.create(fixture.client(), "root", summary_model="cheap")
        self.assertEqual(created["summary_model"], "cheap")

    def test_unknown_fields_are_ignored(self):
        fixture = self.start_server()
        self.create(fixture.client(), "root", unrelated="ignored")


# --- fork ----------------------------------------------------------------


class ForkTests(ServerTestCase):
    def test_fork_links_and_inherits(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        mark = client.mark()
        client.command("fork", id="root", prompt="hello", new_id="a1")
        forked = client.wait_event("agent_forked", since=mark)
        self.assertEqual(forked["agent_id"], "a1")
        self.assertEqual(forked["parent"], "root")
        self.assertEqual(forked["depth"], 1)
        self.assertTrue(forked["dirty"])
        self.assertEqual(forked["prompt"], "hello")
        self.assertEqual(forked["prompt_chars"], 5)
        self.assertEqual(forked["image_count"], 0)
        self.assertEqual(forked["path"], ["root", "a1"])
        self.assertEqual(forked["context_len"], 1)
        self.assertEqual(forked["model"], TEST_MODEL)
        self.assertEqual(forked["timeout"], agent.DEFAULT_TIMEOUT)
        self.assertEqual(forked["max_tokens"], agent.DEFAULT_MAX_TOKENS)
        self.assertEqual(forked["context_window"], agent.DEFAULT_CONTEXT_WINDOW)
        self.assertTrue(forked["include_usage"])
        self.assertFalse(forked["verbose"])
        self.assertEqual(
            sorted(forked["tools"]),
            [
                "get_current_time",
                "get_magic_number",
                "get_system_info",
                "nix_add_dependency",
                "nix_add_file",
                "nix_cat_file",
                "nix_destroy_sandbox",
                "nix_exec",
                "nix_remove_dependency",
                "nix_sandbox_status",
                "nix_spawn_sandbox",
                "set_magic_number",
                "web_fetch",
                "web_search",
            ],
        )
        self.assertEqual(forked["local_tools"], [])
        self.assertEqual(forked["state_namespaces"], [])

    def test_fork_carries_images_over_the_wire(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        data_uri = "data:image/png;base64,AAAA"
        mark = client.mark()
        client.command(
            "fork",
            id="root",
            prompt="what is this?",
            new_id="a1",
            images=[
                data_uri,
                {"url": "https://example.test/cat.webp", "detail": "low"},
            ],
        )
        forked = client.wait_event("agent_forked", since=mark)
        self.assertEqual(forked["prompt"], "what is this?")
        self.assertEqual(forked["prompt_chars"], 13)
        self.assertEqual(forked["image_count"], 2)

        client.command("get_context", id="a1")
        context = client.wait_event("context")
        self.assertEqual(
            context["messages"],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.test/cat.webp",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        )
        client.command("list_agents")
        listed = client.wait_event("agents_listed")
        entry = [a for a in listed["agents"] if a["agent_id"] == "a1"][0]
        self.assertEqual(entry["image_count"], 2)

    def test_bad_images_are_refused_and_leave_no_block(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        cases = [
            5,
            "",
            "/etc/passwd",
            "file:///etc/passwd",
            "data:text/html;base64,AAAA",
            [{"url": 5}],
            [{"url": "x", "detail": 5}],
            [["nested"]],
        ]
        for index, bad in enumerate(cases):
            rid = f"r{index}"
            client.send("fork", rid=rid, id="root", prompt="hi", images=bad)
            client.wait_error(rid, code="bad_image")
        client.command("list_agents")
        self.assertEqual(client.wait_event("agents_listed")["count"], 1)

    def test_long_prompts_are_reported_in_full_but_previewed_short(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        mark = client.mark()
        self.fork(client, "root", "p" * 500, new_id="a1")
        forked = client.wait_event("agent_forked", since=mark)
        self.assertEqual(forked["prompt"], "p" * 500)
        self.assertEqual(forked["prompt_chars"], 500)
        client.command("list_agents")
        listed = client.wait_event("agents_listed")
        entry = [
            candidate for candidate in listed["agents"] if candidate["agent_id"] == "a1"
        ][0]
        self.assertEqual(entry["prompt_chars"], 500)
        self.assertEqual(entry["prompt_preview"], "p" * 200)

    def test_a_prompt_is_required(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        for bad in ("", None, 5):
            client.send("fork", rid="r", id="root", prompt=bad)
            client.wait_error("r", code="bad_prompt")

    def test_the_parent_must_exist(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send("fork", rid="r1", id="ghost", prompt="hi")
        error = client.wait_error("r1", code="unknown_agent")
        self.assertEqual(error["detail"]["agents"], [])

    def test_fork_rejects_a_bad_new_id(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        for bad in ("", 5, {"a": 1}):
            client.send("fork", rid="r", id="root", prompt="hi", new_id=bad)
            client.wait_error("r", code="bad_id")

    def test_forking_from_a_dirty_block_is_refused(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "first", new_id="a1")
        client.send("fork", rid="r1", id="a1", prompt="second")
        error = client.wait_error("r1", code="parent_dirty")
        self.assertEqual(error["detail"], {"agent_id": "a1", "dirty": True})

    def test_a_duplicate_new_id_is_refused(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "first", new_id="a1")
        client.send("fork", rid="r1", id="root", prompt="second", new_id="a1")
        client.wait_error("r1", code="duplicate_agent")

    def test_a_fork_may_override_settings(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        mark = client.mark()
        client.command(
            "fork",
            id="root",
            prompt="hi",
            new_id="a1",
            model="other",
            timeout=9,
            local_timeout="2.5",
            max_tokens=8,
            context_window=1000,
            include_usage=False,
            verbose=True,
            summary_model="summariser",
        )
        forked = client.wait_event("agent_forked", since=mark)
        self.assertEqual(forked["model"], "other")
        self.assertEqual(forked["timeout"], 9.0)
        self.assertEqual(forked["max_tokens"], 8)
        self.assertEqual(forked["context_window"], 1000)
        self.assertFalse(forked["include_usage"])
        self.assertTrue(forked["verbose"])
        block = self.fetch_block(fixture, "a1")
        self.assertEqual(block.summary_model, "summariser")
        self.assertEqual(block.local_timeout, 2.5)

    def test_overrides_are_validated(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        cases = [
            ({"verbose": "yes"}, "bad_field"),
            ({"include_usage": 1}, "bad_field"),
            ({"model": ""}, "bad_field"),
            ({"model": 5}, "bad_field"),
            ({"timeout": "soon"}, "bad_field"),
            ({"timeout": -3}, "bad_field"),
            ({"max_tokens": -1}, "bad_field"),
            ({"max_tokens": 1.5}, "bad_field"),
            ({"max_tokens": True}, "bad_field"),
            ({"context_window": -1}, "bad_field"),
            ({"context_window": 1.5}, "bad_field"),
            ({"context_window": True}, "bad_field"),
            ({"summary_model": ""}, "bad_field"),
            ({"tools": ["nope"]}, "unknown_tool"),
            ({"local_tools": "nope"}, "bad_tools"),
        ]
        for index, (fields, expected) in enumerate(cases):
            rid = f"r{index}"
            client.send("fork", rid=rid, id="root", prompt="hi", **fields)
            client.wait_error(rid, code=expected)
        # none of the failed forks left a block behind
        client.command("list_agents")
        self.assertEqual(client.wait_event("agents_listed")["count"], 1)

    def test_the_tool_axes_are_replaced_independently(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root", local_tools=[{"name": "ask_operator"}])
        self.fork(client, "root", "first", new_id="a1", tools=["get_magic_number"])
        a1 = self.fetch_block(fixture, "a1")
        self.assertEqual(sorted(a1.tools), ["ask_operator", "get_magic_number"])

        self.provider.text("ok")
        self.run_block(client, "a1")
        self.fork(client, "a1", "second", new_id="a2", local_tools=[{"name": "other"}])
        a2 = self.fetch_block(fixture, "a2")
        self.assertEqual(sorted(a2.tools), ["get_magic_number", "other"])

        self.provider.text("ok")
        self.run_block(client, "a2")
        self.fork(client, "a2", "third", new_id="a3", tools=[])
        a3 = self.fetch_block(fixture, "a3")
        self.assertEqual(sorted(a3.tools), ["other"])

    def test_fork_and_run_can_be_pipelined(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.provider.text("pipelined")
        mark = client.mark()
        client.send("fork", rid="f1", id="root", prompt="hi", new_id="a1")
        client.send("run", rid="r1", id="a1")
        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "ok")
        self.assertEqual(
            [message["content"] for message in self.fetch_block(fixture, "a1").messages],
            ["hi", "pipelined"],
        )


# --- run and cancel ------------------------------------------------------


class RunTests(ServerTestCase):
    def test_a_turn_streams_and_finishes(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hello", new_id="a1")
        self.provider.text("hi there", chunk_size=3)
        mark = client.mark()
        accepted, finished = self.run_block(client, "a1")

        self.assertEqual(accepted["agent_id"], "a1")
        self.assertEqual(accepted["depth"], 1)
        self.assertEqual(accepted["context_len"], 1)

        self.assertEqual(finished["command"], "run")
        self.assertEqual(finished["agent_id"], "a1")
        self.assertEqual(finished["status"], "ok")
        self.assertIsNone(finished["error"])
        self.assertIsNone(finished["error_type"])
        self.assertFalse(finished["dirty"])
        self.assertEqual(finished["messages"], 2)
        self.assertEqual(finished["context_len"], 2)
        self.assertEqual(finished["state_committed"], [])
        self.assertGreaterEqual(finished["elapsed_ms"], 0)

        names = client.names(since=mark)
        for expected in (
            "turn_started", "request_started", "request_payload", "response_received",
            "content_delta", "request_finished", "assistant_message", "turn_finished",
        ):
            self.assertIn(expected, names)
        # every turn event names its block; the lifecycle ones need not
        for event in client.events[mark:]:
            if event.get("agent_id") is not None:
                self.assertEqual(event["agent_id"], "a1")
        self.assertTrue(any("agent_id" in event for event in client.events[mark:]))
        self.assertEqual(
            "".join(event["text"] for event in pick(client.events[mark:], "content_delta")),
            "hi there",
        )

    def test_a_root_cannot_run(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.send("run", rid="r1", id="root")
        error = client.wait_error("r1", code="root_agent")
        self.assertIn("fork from it", error["message"])

    def test_an_unknown_agent_cannot_run(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send("run", rid="r1", id="ghost")
        client.wait_error("r1", code="unknown_agent")

    def test_a_block_runs_once(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.text("ok")
        self.run_block(client, "a1")
        client.send("run", rid="r2", id="a1")
        error = client.wait_error("r2", code="agent_finished")
        self.assertIn("already finished", error["message"])
        self.assertIn("detail", error)
        self.assertEqual(self.provider.count, 1)

    def test_a_running_block_cannot_be_run_again(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.push(Response.steady(pieces=8, gap=0.05))
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("content_delta", since=mark)
        client.send("run", rid="r2", id="a1")
        client.wait_error("r2", code="agent_running")
        self.assertEqual(client.wait_event("command_finished", since=mark, rid="r1")["status"], "ok")

    def test_a_failed_turn_reports_its_error(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.error(500, b"provider is unhappy")
        _, finished = self.run_block(client, "a1")
        self.assertEqual(finished["status"], "error")
        self.assertIn("provider is unhappy", finished["error"])
        self.assertEqual(finished["error_type"], "HHAgentError")
        self.assertFalse(finished["dirty"])

        listed = client.command("list_agents")
        self.assertEqual(listed["status"], "ok")
        entry = [
            candidate
            for candidate in client.wait_event("agents_listed")["agents"]
            if candidate["agent_id"] == "a1"
        ][0]
        self.assertEqual(entry["outcome"], "failed")
        self.assertIn("provider is unhappy", entry["error"])

    def test_a_turn_commits_tool_state(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "set the magic number", new_id="a1")
        self.provider.script(
            Response.tool_call("set_magic_number", {"magic": "11"}),
            Response.text("set"),
        )
        _, finished = self.run_block(client, "a1")
        self.assertEqual(finished["state_committed"], ["magic"])
        state = client.command("get_state", id="a1")
        self.assertEqual(state["status"], "ok")
        event = client.wait_event("state")
        self.assertEqual(event["state"], {"magic": {"magic": "11"}})

    def test_cancelling_an_idle_block_is_a_no_op(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        reply = client.command("cancel", id="a1")
        self.assertEqual(reply["status"], "ok")
        result = client.wait_event("cancel_result")
        self.assertFalse(result["cancelled"])
        self.assertFalse(result["running"])
        self.assertTrue(result["dirty"])

    def test_cancelling_a_running_turn(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.push(Response.steady(pieces=10, gap=0.05))
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("content_delta", since=mark)
        reply = client.command("cancel", id="a1")
        self.assertEqual(reply["status"], "ok")
        self.assertTrue(client.wait_event("cancel_result")["cancelled"])
        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "cancelled")
        self.assertIn("cancelled", finished["error"])
        self.assertEqual(finished["error_type"], "HHAgentCancelled")

    def test_cancelling_an_unknown_agent(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send("cancel", rid="r1", id="ghost")
        client.wait_error("r1", code="unknown_agent")

    def test_two_turns_run_at_once_on_one_connection(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "first", new_id="a1")
        self.fork(client, "root", "second", new_id="a2")
        # routed by payload, not by arrival order: the two requests race
        self.provider.match(
            "first", Response.steady(pieces=4, chars=700, gap=0.02, fill="a")
        )
        self.provider.match(
            "second", Response.steady(pieces=4, chars=700, gap=0.02, fill="b")
        )
        client.send("run", rid="r1", id="a1")
        client.send("run", rid="r2", id="a2")
        first_done = client.wait_event("command_finished", rid="r1")
        second_done = client.wait_event("command_finished", rid="r2")
        self.assertEqual(first_done["status"], "ok")
        self.assertEqual(second_done["status"], "ok")
        self.assertEqual(self.fetch_block(fixture, "a1").text, "a" * 2800)
        self.assertEqual(self.fetch_block(fixture, "a2").text, "b" * 2800)
        # the two turns' events are told apart by agent_id
        for event in client.events:
            if event["event"] == "turn_finished":
                self.assertIn(event["agent_id"], {"a1", "a2"})

    def test_many_turns_run_at_once(self):
        fixture = self.start_server()
        first = fixture.client()
        second = fixture.client()
        clients = [first, second]
        for index in range(6):
            client = clients[index % 2]
            self.create(client, f"root{index}")
            self.fork(client, f"root{index}", f"prompt {index}", new_id=f"a{index}")
            self.provider.text("one and the same answer")
        for index in range(6):
            clients[index % 2].send("run", rid=f"r{index}", id=f"a{index}")
        for index in range(6):
            finished = clients[index % 2].wait_event(
                "command_finished", rid=f"r{index}", timeout=30
            )
            self.assertEqual(finished["status"], "ok")
            self.assertEqual(
                self.fetch_block(fixture, f"a{index}").text, "one and the same answer"
            )

    def test_an_unexpected_turn_error_is_still_reported(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        with mock.patch.object(
            agent.HHAgent, "stream", side_effect=ValueError("a harness bug")
        ):
            _, finished = self.run_block(client, "a1")
        self.assertEqual(finished["status"], "error")
        self.assertEqual(finished["error_type"], "ValueError")
        self.assertIn("a harness bug", finished["error"])

    def test_a_disconnect_cancels_the_turn(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.push(Response.steady(pieces=20, chars=700, gap=0.05))
        client.send("run", rid="r1", id="a1")
        client.wait_event("content_delta")
        client.close()
        block = self.fetch_block(fixture, "a1")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and block.outcome is None:
            time.sleep(0.02)
        self.assertEqual(block.outcome, "cancelled")
        self.assertFalse(block.dirty)


# --- context and state ---------------------------------------------------


class ContextAndStateTests(ServerTestCase):
    def test_get_context_flattens_the_chain(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "first", new_id="a1")
        self.provider.text("answer one")
        self.run_block(client, "a1")
        self.fork(client, "a1", "second", new_id="a2")
        self.provider.text("answer two")
        self.run_block(client, "a2")

        reply = client.command("get_context", id="a2")
        self.assertEqual(reply["status"], "ok")
        event = client.wait_event("context")
        self.assertEqual(event["agent_id"], "a2")
        self.assertEqual(event["depth"], 2)
        self.assertEqual(event["path"], ["root", "a1", "a2"])
        self.assertEqual(event["context_len"], 4)
        self.assertEqual(event["local_len"], 2)
        self.assertEqual(event["request_len"], 4)
        self.assertEqual(event["context_window"], agent.DEFAULT_CONTEXT_WINDOW)
        self.assertEqual(
            event["budget_tokens"],
            agent.DEFAULT_CONTEXT_WINDOW - agent.DEFAULT_MAX_TOKENS,
        )
        self.assertEqual(event["dropped_blocks"], 0)
        self.assertEqual(
            [message["content"] for message in event["context"]],
            ["first", "answer one", "second", "answer two"],
        )
        self.assertEqual(
            event["messages"],
            [
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "answer two"},
            ],
        )
        self.assertEqual(event["context"][0], {"role": "user", "content": "first"})

    def test_get_context_reports_what_a_request_would_carry(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root", tools=[], max_tokens=0, context_window=1)
        self.fork(client, "root", "first", new_id="a1")
        self.provider.text("answer one")
        self.run_block(client, "a1")
        self.fork(client, "a1", "second", new_id="a2")

        client.command("get_context", id="a2")
        event = client.wait_event("context")
        # the whole chain is still reported for inspection...
        self.assertEqual(event["context_len"], 3)
        self.assertEqual(event["path"], ["root", "a1", "a2"])
        # ...but a request would carry only the newest block
        self.assertEqual(event["request_len"], 1)
        self.assertEqual(event["dropped_blocks"], 1)
        self.assertEqual(event["context_window"], 1)
        self.assertEqual(event["budget_tokens"], 1)

    def test_get_state_returns_every_touched_namespace(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "go", new_id="a1")
        self.provider.script(
            Response.tool_call("set_magic_number", {"magic": "3"}),
            Response.tool_call("get_current_time", {}),
            Response.text("done"),
        )
        self.run_block(client, "a1")
        client.command("get_state", id="a1")
        event = client.wait_event("state")
        self.assertEqual(event["keys"], ["magic"])
        self.assertEqual(event["state"], {"magic": {"magic": "3"}})
        self.assertEqual(event["depth"], 1)
        self.assertEqual(event["path"], ["root", "a1"])

    def test_get_state_for_one_tool_or_namespace(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "go", new_id="a1")
        self.provider.script(
            Response.tool_call("set_magic_number", {"magic": "3"}),
            Response.text("done"),
        )
        self.run_block(client, "a1")

        mark = client.mark()
        client.command("get_state", id="a1", tool="set_magic_number")
        self.assertEqual(client.wait_event("state", since=mark)["state"], {"magic": {"magic": "3"}})

        mark = client.mark()
        client.command("get_state", id="a1", tool="magic")
        self.assertEqual(client.wait_event("state", since=mark)["state"], {"magic": {"magic": "3"}})

        # an unknown name is a namespace in its own right and holds nothing
        mark = client.mark()
        client.command("get_state", id="a1", tool="unheard_of")
        event = client.wait_event("state", since=mark)
        self.assertEqual(event["keys"], ["unheard_of"])
        self.assertEqual(event["state"], {"unheard_of": {}})

    def test_get_state_rejects_a_bad_tool_field(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        for bad in ("", 5):
            client.send("get_state", rid="r", id="root", tool=bad)
            client.wait_error("r", code="bad_field")

    def test_set_state_seeds_a_root_for_the_whole_subtree(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        reply = client.command("set_state", id="root", tool="set_magic_number", key="magic", value="9")
        self.assertEqual(reply["status"], "ok")
        seeded = client.wait_event("state_seeded")
        self.assertEqual(seeded["tool"], "set_magic_number")
        self.assertEqual(seeded["state_namespace"], "magic")
        self.assertEqual(seeded["key"], "magic")
        self.assertEqual(seeded["value"], "9")
        self.assertFalse(seeded["deleted"])
        self.assertFalse(seeded["replaced"])
        self.assertEqual(seeded["keys"], ["magic"])

        self.fork(client, "root", "what is the magic number?", new_id="a1")
        self.provider.script(
            Response.tool_call("get_magic_number", {}),
            Response.text("it is 9"),
        )
        self.run_block(client, "a1")
        self.assertEqual(self.fetch_block(fixture, "a1").merged_state("magic"), {"magic": "9"})
        # the reading turn commits nothing of its own
        self.assertEqual(self.fetch_block(fixture, "a1").state_deltas, {})

    def test_set_state_reports_a_replaced_value(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.command("set_state", id="root", tool="set_magic_number", key="magic", value="1")
        self.assertFalse(client.wait_event("state_seeded")["replaced"])
        client.command("set_state", id="root", tool="set_magic_number", key="magic", value="2")
        replaced = client.wait_event("state_seeded", replaced=True)
        self.assertEqual(replaced["value"], "2")

    def test_set_state_can_delete_and_drop_a_seed(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.command("set_state", id="root", tool="set_magic_number", key="magic", value="1")
        client.command("set_state", id="root", tool="set_magic_number", key="magic", delete=True)
        deleted = client.wait_event("state_seeded", deleted=True)
        self.assertEqual(deleted["keys"], [])
        self.assertEqual(self.fetch_block(fixture, "root").merged_state("magic"), {})

        # deleting a key that was never there is harmless
        client.command("set_state", id="root", tool="set_magic_number", key="nope", delete=True)
        self.assertEqual(client.wait_event("state_seeded", deleted=True)["keys"], [])

    def test_set_state_drops_a_key_inherited_from_an_ancestor(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.command("set_state", id="root", tool="set_magic_number", key="magic", value="1")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.text("ok")
        self.run_block(client, "a1")

        client.command("set_state", id="a1", tool="set_magic_number", key="magic", delete=True)
        self.assertEqual(self.fetch_block(fixture, "a1").merged_state("magic"), {})
        self.assertEqual(self.fetch_block(fixture, "root").merged_state("magic"), {"magic": "1"})

    def test_set_state_can_put_a_key_back_after_deleting_it(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.command("set_state", id="root", tool="set_magic_number", key="magic", value="1")
        client.command("set_state", id="root", tool="set_magic_number", key="magic", delete=True)
        client.command("set_state", id="root", tool="set_magic_number", key="magic", value="2")
        block = self.fetch_block(fixture, "root")
        self.assertEqual(block.merged_state("magic"), {"magic": "2"})
        # the delta no longer claims to remove what it now sets
        self.assertEqual(block.state_deltas["magic"].removed, ())
        mark = client.mark()
        client.command("get_state", id="root", tool="magic")
        self.assertEqual(
            client.wait_event("state", since=mark)["state"], {"magic": {"magic": "2"}}
        )

    def test_set_state_takes_any_json_value(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.command("set_state", id="root", tool="removed_tool", key="k", value={"a": [1, 2]})
        seeded = client.wait_event("state_seeded")
        self.assertEqual(seeded["state_namespace"], "removed_tool")
        self.assertEqual(self.fetch_block(fixture, "root").merged_state("removed_tool"), {"k": {"a": [1, 2]}})

        client.command("set_state", id="root", tool="removed_tool", key="nothing", value=None)
        self.assertEqual(self.fetch_block(fixture, "root").merged_state("removed_tool")["nothing"], None)

    def test_set_state_refuses_a_dirty_or_running_block(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        client.send("set_state", rid="r1", id="a1", tool="set_magic_number", key="magic", value="1")
        client.wait_error("r1", code="agent_dirty")

        self.provider.push(Response.steady(pieces=8, gap=0.05))
        mark = client.mark()
        client.send("run", rid="r2", id="a1")
        client.wait_event("content_delta", since=mark)
        client.send("set_state", rid="r3", id="a1", tool="set_magic_number", key="magic", value="1")
        client.wait_error("r3", code="agent_running")
        client.wait_event("command_finished", since=mark, rid="r2")

    def test_set_state_validates_its_fields(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        cases = [
            {"tool": "set_magic_number", "key": "magic"},  # no value, no delete
            {"key": "magic", "value": "1"},  # no tool
            {"tool": "", "key": "magic", "value": "1"},
            {"tool": 5, "key": "magic", "value": "1"},
            {"tool": "set_magic_number", "value": "1"},  # no key
            {"tool": "set_magic_number", "key": "", "value": "1"},
            {"tool": "set_magic_number", "key": 5, "value": "1"},
        ]
        for index, fields in enumerate(cases):
            rid = f"r{index}"
            client.send("set_state", rid=rid, id="root", **fields)
            client.wait_error(rid, code="bad_field")

    def test_set_state_is_written_through_to_the_store(self):
        fixture = self.start_server(with_store=True)
        client = fixture.client()
        self.create(client, "root")
        client.command("set_state", id="root", tool="set_magic_number", key="magic", value="5")
        connection = sqlite3.connect(fixture.db_path)
        row = connection.execute("SELECT state_deltas FROM blocks WHERE id = 'root'").fetchone()
        connection.close()
        self.assertIsNotNone(row)
        deltas = pickle.loads(row[0])
        self.assertEqual(deltas["magic"], agent.StateDelta(changed={"magic": "5"}))


# --- local tools over the protocol ---------------------------------------


class LocalToolProtocolTests(ServerTestCase):
    def test_a_local_call_parks_one_turn_and_nothing_else(self):
        fixture = self.start_server(with_store=True)
        client = fixture.client()
        self.create(client, "root", local_tools=[{"name": "ask_operator"}])
        self.fork(client, "root", "ask", new_id="a1")
        self.provider.script(
            Response.tool_calls([("ask_operator", {"question": "lunch?"}, "c1")]),
            Response.text("thanks"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        called = client.wait_event("local_tool_called", since=mark)
        self.assertEqual(called["agent_id"], "a1")
        self.assertEqual(called["call_id"], "c1")
        self.assertEqual(called["name"], "ask_operator")
        self.assertEqual(called["kind"], "call")
        self.assertEqual(called["arguments"], {"question": "lunch?"})
        self.assertEqual(called["timeout_ms"], agent.DEFAULT_LOCAL_TIMEOUT * 1000)

        # every other connection keeps working while the turn is parked, writes
        # included — the wait holds no lock and no transaction
        self.assertFalse(fixture.store._db.in_transaction)
        self.assertTrue(fixture.server._registry_lock.acquire(blocking=False))
        fixture.server._registry_lock.release()
        other = fixture.client()
        self.create(other, "second-root")
        self.assertIsNotNone(fixture.agent("second-root"))
        self.assertEqual(fixture.store.stats()["blocks"], 3)  # root, a1 and the new root
        listed = other.command("list_agents")
        self.assertEqual(listed["status"], "ok")
        entry = [
            candidate
            for candidate in other.wait_event("agents_listed")["agents"]
            if candidate["agent_id"] == "a1"
        ][0]
        self.assertEqual(entry["waiting_on"], ["c1"])
        self.assertTrue(entry["running"])

        # the parked turn is finished by answering it
        client.command("resolve_tool", id="a1", call_id="c1", result="red braised pork")
        answered = client.wait_event("local_tool_answered")
        self.assertEqual(answered["call_id"], "c1")
        self.assertTrue(answered["ok"])
        self.assertEqual(answered["result_chars"], len("red braised pork"))
        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "ok")
        self.assertEqual(self.fetch_block(fixture, "a1").messages[2]["content"], "red braised pork")

    def test_a_local_call_can_be_reported_as_failed(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root", local_tools=[{"name": "ask_operator"}])
        self.fork(client, "root", "ask", new_id="a1")
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.text("ok then"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("local_tool_called", since=mark)
        client.command("resolve_tool", id="a1", call_id="c1", error="operator_away")
        self.assertFalse(client.wait_event("local_tool_answered")["ok"])
        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "ok")  # the turn survives the failure
        block = self.fetch_block(fixture, "a1")
        self.assertIn("operator_away", block.messages[2]["content"])

    def test_a_non_string_result_is_json_encoded(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root", local_tools=[{"name": "ask_operator"}])
        self.fork(client, "root", "ask", new_id="a1")
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.text("ok"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("local_tool_called", since=mark)
        client.command("resolve_tool", id="a1", call_id="c1", result={"answer": 5})
        client.wait_event("local_tool_answered")
        client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(
            self.fetch_block(fixture, "a1").messages[2]["content"], '{"answer": 5}'
        )

    def test_a_local_answer_may_carry_images_instead_of_text(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root", local_tools=[{"name": "ask_operator"}])
        self.fork(client, "root", "ask", new_id="a1")
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.text("nice"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("local_tool_called", since=mark)
        data_uri = "data:image/png;base64,AAAA"
        client.command("resolve_tool", id="a1", call_id="c1", images=[data_uri])
        answered = client.wait_event("local_tool_answered")
        self.assertTrue(answered["ok"])
        self.assertEqual(answered["result_chars"], 0)
        self.assertEqual(answered["image_count"], 1)
        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "ok")
        # the model sees the image alone as the tool message
        self.assertEqual(
            self.provider.last_payload()["messages"][-1],
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
            },
        )
        self.assertEqual(
            self.fetch_block(fixture, "a1").messages[2]["content"],
            [{"type": "image_url", "image_url": {"url": data_uri}}],
        )

    def test_resolve_tool_rejects_bad_images(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root", local_tools=[{"name": "ask_operator"}])
        self.fork(client, "root", "ask", new_id="a1")
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.text("ok"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("local_tool_called", since=mark)
        client.send(
            "resolve_tool", rid="r2", id="a1", call_id="c1", images=[{"url": 5}]
        )
        client.wait_error("r2", code="bad_image")
        # the call is still parked, so it can be answered properly
        self.assertEqual(self.fetch_block(fixture, "a1").pending_calls(), ["c1"])
        client.command("resolve_tool", id="a1", call_id="c1", result="fine")
        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "ok")

    def test_resolving_a_call_nothing_waits_on(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.send("resolve_tool", rid="r1", id="root", call_id="ghost", result="x")
        error = client.wait_error("r1", code="unknown_call")
        self.assertEqual(error["detail"]["pending"], [])

    def test_resolve_tool_validates_its_fields(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        cases = [
            {"call_id": "", "result": "x"},
            {"call_id": 5, "result": "x"},
            {"call_id": "c1"},  # neither result nor error
            {"call_id": "c1", "result": "x", "error": ""},
            {"call_id": "c1", "error": 5},
            {"call_id": "c1", "error": ""},
        ]
        for index, fields in enumerate(cases):
            rid = f"r{index}"
            client.send("resolve_tool", rid=rid, id="root", **fields)
            client.wait_error(rid, code="bad_field")

    def test_a_local_call_that_times_out_is_reported_to_the_model(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(
            client, "root", local_timeout=0.15, local_tools=[{"name": "ask_operator"}]
        )
        self.fork(client, "root", "ask", new_id="a1")
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.text("the operator never answered"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        unresolved = client.wait_event("local_tool_unresolved", since=mark)
        self.assertEqual(unresolved["reason"], "timeout")
        self.assertEqual(unresolved["kind"], "call")
        self.assertGreaterEqual(unresolved["waited_ms"], 100)
        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "ok")
        block = self.fetch_block(fixture, "a1")
        self.assertIn("was not answered (timeout)", block.messages[2]["content"])

    def test_a_cancelled_turn_releases_a_parked_call(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root", local_tools=[{"name": "ask_operator"}])
        self.fork(client, "root", "ask", new_id="a1")
        self.provider.script(Response.tool_calls([("ask_operator", {}, "c1")]))
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("local_tool_called", since=mark)
        client.command("cancel", id="a1")
        unresolved = client.wait_event("local_tool_unresolved", since=mark)
        self.assertEqual(unresolved["reason"], "cancelled")
        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "cancelled")

    def test_a_local_undo_is_asked_for_over_the_protocol(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(
            client,
            "root",
            local_tools=[{"name": "ask_operator", "rollback": True}],
        )
        self.fork(client, "root", "ask", new_id="a1")
        self.provider.script(
            Response.tool_calls([("ask_operator", {"q": "hi"}, "c1")]),
            Response.error(500),
            Response.text("summary"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("local_tool_called", since=mark)
        client.command("resolve_tool", id="a1", call_id="c1", result="the answer")

        started = client.wait_event("rollback_started", since=mark)
        self.assertEqual(started["tool"], "ask_operator")
        self.assertEqual(started["call_id"], "c1")
        self.assertEqual((started["index"], started["total"]), (1, 1))
        asked = client.wait_event("local_tool_rollback", since=mark)
        self.assertEqual(asked["call_id"], "c1:rollback")
        self.assertEqual(asked["rollback_of"], "c1")
        self.assertEqual(asked["arguments"], {"q": "hi"})
        self.assertEqual(asked["result"], "the answer")
        self.assertTrue(asked["call_ok"])
        client.command("resolve_tool", id="a1", call_id="c1:rollback", result="undone")
        finished = client.wait_event("rollback_finished", since=mark)
        self.assertTrue(finished["ok"])
        self.assertIsNone(finished["error"])
        self.assertEqual(finished["result"], "undone")
        run = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(run["status"], "error")
        note = self.fetch_block(fixture, "a1").messages[-1]["content"]
        self.assertIn("Undone: ask_operator.", note)

    def test_a_local_undo_that_fails_is_reported_in_the_note(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(
            client, "root", local_tools=[{"name": "ask_operator", "rollback": True}]
        )
        self.fork(client, "root", "ask", new_id="a1")
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.error(500),
            Response.text("summary"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("local_tool_called", since=mark)
        client.command("resolve_tool", id="a1", call_id="c1", result="the answer")
        client.wait_event("local_tool_rollback", since=mark)
        client.command("resolve_tool", id="a1", call_id="c1:rollback", error="cannot undo")
        finished = client.wait_event("rollback_finished", since=mark)
        self.assertFalse(finished["ok"])
        self.assertEqual(finished["error"], "cannot undo")
        client.wait_event("command_finished", since=mark, rid="r1")
        note = self.fetch_block(fixture, "a1").messages[-1]["content"]
        self.assertIn("Could not be undone: ask_operator.", note)

    def test_a_local_tool_without_an_undo_is_only_named_when_it_declares_effects(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(
            client,
            "root",
            local_tools=[{"name": "send_mail", "external_effects": True}],
        )
        self.fork(client, "root", "mail", new_id="a1")
        self.provider.script(
            Response.tool_calls([("send_mail", {}, "c1")]),
            Response.error(500),
            Response.text("summary"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("local_tool_called", since=mark)
        client.command("resolve_tool", id="a1", call_id="c1", result="sent")
        unavailable = client.wait_event("rollback_unavailable", since=mark)
        self.assertEqual(unavailable["tool"], "send_mail")
        client.wait_event("command_finished", since=mark, rid="r1")
        note = self.fetch_block(fixture, "a1").messages[-1]["content"]
        self.assertIn("May still be in effect: send_mail.", note)


# --- tool pipes over the protocol ----------------------------------------


class ToolPipeProtocolTests(ServerTestCase):
    """A local tool can answer with a call; the server runs it before the model."""

    def _pipe_turn(self, fixture, client):
        """Run a turn whose only call is a local one, up to its answer."""
        self.create(client, "root", local_tools=[{"name": "fetch_file"}])
        self.fork(client, "root", "fetch", new_id="a1")
        self.provider.script(
            Response.tool_calls([("fetch_file", {"url": "https://x.test/f"}, "c1")]),
            Response.text("thanks"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        called = client.wait_event("local_tool_called", since=mark)
        self.assertEqual(called["call_id"], "c1")
        return mark

    def test_a_local_answer_can_pipe_into_a_server_tool(self):
        fixture = self.start_server()
        client = fixture.client()
        mark = self._pipe_turn(fixture, client)

        # the client fetched the bytes itself and hands them to a builtin, so
        # neither the bytes nor the note cross the provider connection
        client.command(
            "resolve_tool",
            id="a1",
            call_id="c1",
            result="downloaded 32 bytes",
            call={"name": "set_magic_number", "arguments": {"magic": "piped"}},
        )
        answered = client.wait_event("local_tool_answered", since=mark)
        self.assertTrue(answered["ok"])
        self.assertEqual(answered["next"], "set_magic_number")

        resolved = client.wait_event("local_tool_resolved", since=mark)
        self.assertEqual(resolved["next"], "set_magic_number")
        step = client.wait_event("pipe_step_started", since=mark)
        self.assertEqual(step["name"], "set_magic_number")
        self.assertEqual(step["call_id"], "c1:pipe:1")
        self.assertEqual(step["parent_call_id"], "c1")
        self.assertEqual(step["step"], 1)
        self.assertEqual(step["chain"], ["fetch_file", "set_magic_number"])
        self.assertEqual(step["via"], "server")
        finished = client.wait_event("pipe_step_finished", since=mark)
        self.assertTrue(finished["ok"])
        self.assertEqual(finished["text"], "Magic is set to piped")
        outer = client.wait_event("tool_call_finished", since=mark)
        self.assertEqual(outer["chain"], ["fetch_file", "set_magic_number"])
        self.assertEqual(outer["steps"], 1)

        done = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(done["status"], "ok")
        block = self.fetch_block(fixture, "a1")
        self.assertEqual(
            block.messages[2]["content"],
            "[tool pipe] fetch_file -> set_magic_number\nMagic is set to piped",
        )
        # the piped call was a real call: its state committed with the turn
        self.assertEqual(block.merged_state("magic"), {"magic": "piped"})
        self.assertEqual(
            block.pipe_traces[0]["steps"][0]["text"], "downloaded 32 bytes"
        )
        self.assertNotIn(
            "downloaded 32 bytes", json.dumps(self.provider.last_payload()["messages"])
        )

    def test_resolve_tool_validates_a_piped_call(self):
        fixture = self.start_server()
        client = fixture.client()
        mark = self._pipe_turn(fixture, client)
        cases = [
            {"call": 5},
            {"call": {"name": ""}},
            {"call": {"name": "set_magic_number", "arguments": []}},
            {"call": {"name": "set_magic_number"}, "error": "nope"},
            {
                "call": {"name": "set_magic_number"},
                "images": ["data:image/png;base64,AAAA"],
            },
        ]
        for index, fields in enumerate(cases):
            rid = f"r{index}"
            client.send("resolve_tool", rid=rid, id="a1", call_id="c1", **fields)
            client.wait_error(rid, code="bad_call")
            # a rejected answer leaves the call parked, so it can be retried
            self.assertEqual(self.fetch_block(fixture, "a1").pending_calls(), ["c1"])
        client.command("resolve_tool", id="a1", call_id="c1", result="fine")
        done = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(done["status"], "ok")
        self.assertEqual(self.fetch_block(fixture, "a1").messages[2]["content"], "fine")

    def test_get_context_and_list_agents_report_pipe_traces(self):
        fixture = self.start_server(with_store=True)
        client = fixture.client()
        mark = self._pipe_turn(fixture, client)
        client.command(
            "resolve_tool",
            id="a1",
            call_id="c1",
            call={"name": "set_magic_number", "arguments": {"magic": "piped"}},
        )
        client.wait_event("command_finished", since=mark, rid="r1")

        client.command("get_context", id="a1")
        context = client.wait_event("context", since=mark)
        self.assertEqual(context["messages"][2]["content"].splitlines()[0],
                         "[tool pipe] fetch_file -> set_magic_number")
        self.assertEqual(len(context["pipe_traces"]), 1)
        trace = context["pipe_traces"][0]
        self.assertEqual(trace["call_id"], "c1")
        self.assertEqual(trace["chain"], ["fetch_file", "set_magic_number"])
        self.assertTrue(trace["ok"])
        self.assertEqual(
            [step["name"] for step in trace["steps"]],
            ["fetch_file", "set_magic_number"],
        )
        self.assertEqual(trace["steps"][0]["via"], "client")
        self.assertEqual(trace["steps"][0]["next"], "set_magic_number")
        self.assertEqual(
            trace["steps"][1]["arguments"], {"magic": "piped"}
        )

        client.command("list_agents")
        listed = client.wait_event("agents_listed", since=mark)
        entry = [
            candidate
            for candidate in listed["agents"]
            if candidate["agent_id"] == "a1"
        ][0]
        self.assertEqual(entry["pipe_traces"], 1)


# --- the registry --------------------------------------------------------


class RegistryTests(ServerTestCase):
    def test_list_agents_describes_every_block(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hello", new_id="a1")
        self.provider.text("hi")
        self.run_block(client, "a1")

        reply = client.command("list_agents")
        self.assertEqual(reply["status"], "ok")
        listed = client.wait_event("agents_listed")
        self.assertEqual(listed["count"], 2)
        self.assertEqual(listed["roots"], ["root"])
        self.assertEqual(listed["dirty"], [])
        entries = {entry["agent_id"]: entry for entry in listed["agents"]}
        root = entries["root"]
        self.assertIsNone(root["parent"])
        self.assertEqual(root["depth"], 0)
        self.assertFalse(root["dirty"])
        self.assertIsNone(root["outcome"])
        self.assertEqual(root["prompt_chars"], 0)
        self.assertEqual(root["prompt_preview"], "")
        self.assertEqual(root["text_chars"], 0)
        self.assertEqual(root["local_len"], 0)
        self.assertEqual(root["context_len"], 0)
        self.assertEqual(root["model"], TEST_MODEL)
        self.assertEqual(root["summary_model"], "")
        self.assertEqual(root["state_namespaces"], [])
        self.assertEqual(root["waiting_on"], [])
        self.assertEqual(root["max_tokens"], agent.DEFAULT_MAX_TOKENS)
        self.assertEqual(root["context_window"], agent.DEFAULT_CONTEXT_WINDOW)
        self.assertEqual(
            root["budget_tokens"],
            agent.DEFAULT_CONTEXT_WINDOW - agent.DEFAULT_MAX_TOKENS,
        )
        self.assertTrue(root["include_usage"])
        self.assertFalse(root["verbose"])
        self.assertGreater(root["created_at"], 0)
        self.assertGreaterEqual(root["age_ms"], 0)
        child = entries["a1"]
        self.assertEqual(child["parent"], "root")
        self.assertEqual(child["depth"], 1)
        self.assertFalse(child["dirty"])
        self.assertEqual(child["outcome"], "ok")
        self.assertEqual(child["prompt_chars"], 5)
        self.assertEqual(child["prompt_preview"], "hello")
        self.assertEqual(child["text_chars"], 2)
        self.assertEqual(child["local_len"], 2)
        self.assertEqual(child["context_len"], 2)
        self.assertEqual(child["tools"], sorted(child["tools"]))

    def test_destroy_agent_drops_the_whole_subtree(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "first", new_id="a1")
        self.provider.text("one")
        self.run_block(client, "a1")
        self.fork(client, "a1", "second", new_id="a2")
        self.fork(client, "root", "other branch", new_id="b1")

        reply = client.command("destroy_agent", id="a1")
        self.assertEqual(reply["status"], "ok")
        destroyed = client.wait_event("agent_destroyed")
        self.assertEqual(destroyed["agent_id"], "a1")
        self.assertEqual(sorted(destroyed["dropped"]), ["a1", "a2"])
        self.assertEqual(destroyed["count"], 2)
        self.assertEqual(destroyed["cancelled"], [])
        self.assertGreaterEqual(destroyed["age_ms"], 0)
        self.assertEqual(destroyed["remaining"], 2)

        client.command("list_agents")
        remaining = client.wait_event("agents_listed")
        self.assertEqual(sorted(entry["agent_id"] for entry in remaining["agents"]), ["b1", "root"])
        # the dropped blocks are really gone, not just hidden
        client.send("get_context", rid="r1", id="a2")
        client.wait_error("r1", code="unknown_agent")

    def test_destroy_agent_cancels_a_running_turn(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.push(Response.steady(pieces=20, chars=700, gap=0.05))
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("content_delta", since=mark)
        client.command("destroy_agent", id="a1")
        destroyed = client.wait_event("agent_destroyed", since=mark)
        self.assertEqual(destroyed["cancelled"], ["a1"])
        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "cancelled")
        with fixture.server._registry_lock:
            self.assertEqual(sorted(fixture.server._agents), ["root"])

    def test_destroy_agent_refuses_an_unknown_id(self):
        fixture = self.start_server()
        client = fixture.client()
        client.send("destroy_agent", rid="r1", id="ghost")
        client.wait_error("r1", code="unknown_agent")

    def test_destroy_agent_removes_the_store_rows(self):
        fixture = self.start_server(with_store=True)
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.assertEqual(fixture.store.stats()["blocks"], 2)
        client.command("destroy_agent", id="a1")
        self.assertEqual(fixture.store.stats()["blocks"], 1)

    def test_a_block_destroyed_mid_turn_is_not_written_back(self):
        fixture = self.start_server(with_store=True)
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.push(Response.steady(pieces=20, chars=700, gap=0.03))
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("content_delta", since=mark)
        client.command("destroy_agent", id="a1")
        client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(fixture.store.stats()["blocks"], 1)  # only the root


# --- persistence ---------------------------------------------------------


class PersistenceTests(ServerTestCase):
    with_store = True

    def test_blocks_are_written_as_they_are_created(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.assertEqual(fixture.store.stats()["blocks"], 1)
        self.fork(client, "root", "hi", new_id="a1")
        self.assertEqual(fixture.store.stats()["blocks"], 2)

    def test_the_turn_is_on_disk_before_its_completion_event(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.text("stored before you knew")
        _, finished = self.run_block(client, "a1")
        self.assertEqual(finished["status"], "ok")

        connection = sqlite3.connect(fixture.db_path)
        try:
            row = connection.execute(
                "SELECT text, outcome, dirty, messages FROM blocks WHERE id = 'a1'"
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(row[0], "stored before you knew")
        self.assertEqual(row[1], "ok")
        self.assertEqual(row[2], 0)
        self.assertEqual(
            [message["role"] for message in json.loads(row[3])], ["user", "assistant"]
        )

    def test_a_restart_restores_the_chain(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "hi", new_id="a1")
        self.provider.text("first answer")
        self.run_block(client, "a1")
        self.stop_server(fixture)

        second = self.start_server()
        client = second.client()
        hello = client.wait_event("session_hello")
        self.assertEqual(hello["store"]["blocks"], 2)
        client.command("list_agents")
        listed = client.wait_event("agents_listed")
        self.assertEqual(sorted(entry["agent_id"] for entry in listed["agents"]), ["a1", "root"])
        child = [entry for entry in listed["agents"] if entry["agent_id"] == "a1"][0]
        self.assertEqual(child["parent"], "root")
        self.assertEqual(child["outcome"], "ok")

        client.command("get_context", id="a1")
        context = client.wait_event("context")
        self.assertEqual(
            [message["content"] for message in context["context"]],
            ["hi", "first answer"],
        )

        # the restored chain can be continued, with the server's own key
        self.fork(client, "a1", "and again", new_id="a2")
        self.provider.text("second answer")
        _, finished = self.run_block(client, "a2")
        self.assertEqual(finished["status"], "ok")
        self.assertEqual(
            [message["content"] for message in self.fetch_block(second, "a2").messages],
            ["and again", "second answer"],
        )

    def test_an_image_message_survives_a_restart(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        data_uri = "data:image/png;base64,AAAA"
        self.fork(client, "root", "look", new_id="a1", images=[data_uri])
        self.stop_server(fixture)

        second = self.start_server()
        client = second.client()
        client.wait_event("session_hello")
        client.command("get_context", id="a1")
        context = client.wait_event("context")
        self.assertEqual(
            context["context"][0]["content"],
            [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        )
        client.command("list_agents")
        listed = client.wait_event("agents_listed")
        entry = [a for a in listed["agents"] if a["agent_id"] == "a1"][0]
        self.assertEqual(entry["image_count"], 1)

    def test_pipe_traces_do_not_survive_a_restart(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root", local_tools=[{"name": "fetch_file"}])
        self.fork(client, "root", "fetch", new_id="a1")
        self.provider.script(
            Response.tool_calls([("fetch_file", {"url": "https://x.test/f"}, "c1")]),
            Response.text("thanks"),
        )
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("local_tool_called", since=mark)
        client.command(
            "resolve_tool",
            id="a1",
            call_id="c1",
            call={"name": "set_magic_number", "arguments": {"magic": "kept"}},
        )
        client.wait_event("command_finished", since=mark, rid="r1")

        # the process that ran the pipe can still show it
        client.command("get_context", id="a1")
        self.assertEqual(len(client.wait_event("context")["pipe_traces"]), 1)
        self.stop_server(fixture)

        second = self.start_server()
        client = second.client()
        client.command("get_context", id="a1")
        event = client.wait_event("context")
        # a trace is inspection data, not part of the conversation, and the
        # store never keeps it: a restarted block reports none
        self.assertEqual(event["pipe_traces"], [])
        client.command("list_agents")
        listed = client.wait_event("agents_listed")
        entry = [a for a in listed["agents"] if a["agent_id"] == "a1"][0]
        self.assertEqual(entry["pipe_traces"], 0)
        # what the turn really owns survives: the transcript and the pipe note
        self.assertEqual(
            self.fetch_block(second, "a1").messages[2]["content"],
            "[tool pipe] fetch_file -> set_magic_number\nMagic is set to kept",
        )

    def test_a_restart_keeps_local_tool_definitions(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(
            client,
            "root",
            local_tools=[{"name": "ask_operator", "rollback": True, "external_effects": True}],
        )
        self.stop_server(fixture)

        second = self.start_server()
        client = second.client()
        client.command("list_agents")
        entry = client.wait_event("agents_listed")["agents"][0]
        self.assertEqual(entry["local_tools"], ["ask_operator"])
        self.assertEqual(entry["rollback_tools"], ["ask_operator"])
        block = self.fetch_block(second, "root")
        self.assertTrue(block.tools["ask_operator"].is_local)
        self.assertTrue(block.tools["ask_operator"].remote_rollback)
        self.assertTrue(block.tools["ask_operator"].external_effects)

    def test_restore_warnings_are_reported_to_the_client(self):
        fixture = self.start_server()
        self.stop_server(fixture)
        seed_database(
            fixture.db_path,
            [block_with("ghostly", tools=[tools.ToolEntry("ghost", "d", [], hook=lambda c: "")])],
        )
        second = self.start_server()
        hello = second.client().wait_event("session_hello")
        self.assertEqual(hello["store"]["warnings"], ["ghostly: missing tools ['ghost']"])

    def test_seeded_state_survives_a_restart(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        client.command("set_state", id="root", tool="set_magic_number", key="magic", value="42")
        self.stop_server(fixture)

        second = self.start_server()
        client = second.client()
        client.command("get_state", id="root")
        self.assertEqual(
            client.wait_event("state")["state"], {"magic": {"magic": "42"}}
        )

    def test_unpicklable_state_produces_a_persist_warning(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        block = self.fetch_block(fixture, "root")
        block.state_deltas["broken"] = agent.StateDelta(changed={"fn": lambda: None})

        stub = RecordingConnection()
        fixture.server.persist(stub, block)
        warning = [event for event in stub.events if event["event"] == "persist_warning"][0]
        self.assertEqual(warning["agent_id"], "root")
        self.assertEqual(warning["dropped_tools"], ["broken"])
        self.assertIn("could not be stored", warning["message"])
        # the rest of the block is still written
        self.assertEqual(fixture.store.stats()["blocks"], 1)

    def test_persist_skips_a_block_that_is_no_longer_registered(self):
        fixture = self.start_server()
        client = fixture.client()
        self.create(client, "root")
        block = self.fetch_block(fixture, "root")
        with fixture.server._registry_lock:
            del fixture.server._agents["root"]
        stub = RecordingConnection()
        fixture.server.persist(stub, block)
        self.assertEqual(stub.events, [])
        self.assertEqual(fixture.store.stats()["blocks"], 1)  # the old row stays


# --- eviction ------------------------------------------------------------


class EvictionTests(ServerTestCase):
    with_store = True
    max_db_bytes = 8192

    def test_a_whole_tree_is_evicted_when_the_file_outgrows_its_budget(self):
        fixture = self.start_server(max_db_bytes=0)
        client = fixture.client()
        self.create(client, "root")
        # allow exactly one more page beyond the root, so a big prompt busts it
        fixture.store.max_bytes = fixture.store.size_bytes() + 4096
        mark = client.mark()
        self.fork(client, "root", "x" * 60_000, new_id="a1")
        evicted = client.wait_event("evicted", since=mark)
        self.assertEqual(evicted["agent_id"], "root")
        self.assertEqual(sorted(evicted["dropped"]), ["a1", "root"])
        self.assertEqual(evicted["nodes"], 2)
        self.assertGreater(evicted["newest"], 0)
        self.assertLessEqual(evicted["bytes"], evicted["max_bytes"])
        self.assertEqual(evicted["max_bytes"], fixture.store.max_bytes)
        self.assertEqual(fixture.store.stats()["blocks"], 0)
        client.send("get_context", rid="r1", id="a1")
        client.wait_error("r1", code="unknown_agent")

    def test_the_oldest_tree_is_evicted_first(self):
        fixture = self.start_server(max_db_bytes=0)
        client = fixture.client()
        self.create(client, "quiet")
        self.fork(client, "quiet", "small", new_id="quiet-child")
        self.create(client, "busy")
        self.fork(client, "busy", "y" * 60_000, new_id="busy-child")
        self.provider.text("done")
        self.run_block(client, "busy-child")
        fixture.store.max_bytes = 8192

        # any write triggers the check; seeding state on the quiet root will do
        client.command("set_state", id="quiet", tool="set_magic_number", key="magic", value="1")
        evicted = client.wait_events("evicted", count=2)
        self.assertEqual([event["agent_id"] for event in evicted], ["quiet", "busy"])
        self.assertEqual(sorted(evicted[0]["dropped"]), ["quiet", "quiet-child"])
        self.assertEqual(fixture.store.stats()["blocks"], 0)

    def test_a_running_tree_is_protected_until_it_finishes(self):
        fixture = self.start_server(max_db_bytes=0)
        client = fixture.client()
        self.create(client, "root")
        self.fork(client, "root", "x" * 60_000, new_id="a1")
        fixture.store.max_bytes = 8192  # over budget from here on

        self.provider.push(Response.steady(pieces=10, gap=0.05))
        mark = client.mark()
        client.send("run", rid="r1", id="a1")
        client.wait_event("content_delta", since=mark)

        # a second connection writes: the running tree must be skipped, so the
        # newer, idle one is evicted instead
        other = fixture.client()
        self.create(other, "other-root")
        evicted = other.wait_event("evicted")
        self.assertEqual(evicted["agent_id"], "other-root")
        self.assertIsNotNone(fixture.agent("a1"))

        finished = client.wait_event("command_finished", since=mark, rid="r1")
        self.assertEqual(finished["status"], "ok")
        # with the turn over, the next write can reclaim it
        client.wait_event("evicted", agent_id="root", timeout=10)
        self.assertIsNone(fixture.agent("a1"))

    def test_the_limit_is_off_at_zero(self):
        fixture = self.start_server(max_db_bytes=0)
        client = fixture.client()
        self.create(client, "root")
        mark = client.mark()
        self.fork(client, "root", "z" * 60_000, new_id="a1")
        client.assert_no_event("evicted", timeout=0.25, since=mark)
        self.assertIsNotNone(fixture.agent("a1"))
        self.assertGreater(fixture.store.size_bytes(), 8192)

    def test_a_budget_that_can_never_be_met_still_stops(self):
        # SQLite's own schema is larger than this budget, so the store is emptied
        # and the loop stops rather than spinning
        fixture = self.start_server(max_db_bytes=1)
        client = fixture.client()
        self.create(client, "root")
        evicted = client.wait_event("evicted")
        self.assertEqual(evicted["agent_id"], "root")
        self.assertEqual(fixture.store.stats()["blocks"], 0)
        self.assertEqual(client.command("ping")["status"], "ok")

    def test_a_write_and_an_eviction_never_interleave(self):
        # `persist` checks the registry and writes under one lock hold, and
        # `enforce_limit` picks a victim, deletes it and drops it under one hold
        # as well. Splitting either lets an eviction land between another turn's
        # check and its insert, and the insert fails its foreign key.
        fixture = self.start_server(max_db_bytes=0)
        client = fixture.client()
        self.create(client, "root")
        child_id = self.fork(client, "root", "hi", new_id="a1")
        child = self.fetch_block(fixture, child_id)

        real_save = fixture.store.save
        inside_save = threading.Event()
        release_save = threading.Event()
        failures = []

        def slow_save(block):
            inside_save.set()
            release_save.wait(5.0)
            return real_save(block)

        def evict():
            try:
                fixture.server.enforce_limit(RecordingConnection())
            except BaseException as exc:  # the bug this test guards against
                failures.append(exc)

        with mock.patch.object(fixture.store, "save", side_effect=slow_save):
            writer = threading.Thread(
                target=fixture.server.persist,
                args=(RecordingConnection(), child),
            )
            writer.start()
            self.assertTrue(inside_save.wait(5.0), "the writer never reached save")

            fixture.store.max_bytes = 1  # anything at all must now be evicted
            evictor = threading.Thread(target=evict)
            evictor.start()
            time.sleep(0.3)
            self.assertTrue(evictor.is_alive(), "the eviction ran inside the write")

            release_save.set()
            writer.join(5.0)
            evictor.join(5.0)

        self.assertEqual(failures, [])
        self.assertFalse(writer.is_alive())
        self.assertFalse(evictor.is_alive())
        # the write landed, then the eviction reclaimed the tree; nothing dangles
        self.assertEqual(fixture.store.stats()["blocks"], 0)
        with fixture.server._registry_lock:
            self.assertEqual(fixture.server._agents, {})

        # and the store is still healthy afterwards
        fixture.store.max_bytes = 0
        self.create(client, "fresh-root")
        self.assertEqual(fixture.store.stats()["blocks"], 1)


# --- module-level helpers ------------------------------------------------


class HelperTests(ServerTestCase):
    def test_require_id(self):
        self.assertEqual(server.require_id({"id": "x"}), "x")
        for bad in ({}, {"id": ""}, {"id": 5}, {"id": None}):
            with self.assertRaises(server.HHTcpError) as caught:
                server.require_id(bad)
            self.assertEqual(caught.exception.code, "bad_id")

    def test_hhtcp_error_carries_code_and_detail(self):
        error = server.HHTcpError("weird", "a message", {"k": 1})
        self.assertEqual(str(error), "a message")
        self.assertEqual((error.code, error.detail), ("weird", {"k": 1}))

    def test_as_timeout(self):
        self.assertEqual(server.as_timeout("2.5", "timeout"), 2.5)
        self.assertEqual(server.as_timeout(3, "timeout"), 3.0)
        for bad in (0, -1, "soon", None, [1]):
            with self.assertRaises(server.HHTcpError) as caught:
                server.as_timeout(bad, "local_timeout")
            self.assertEqual(caught.exception.code, "bad_field")
            self.assertIn("local_timeout", str(caught.exception))

    def test_namespace_for(self):
        self.assertEqual(server.namespace_for("set_magic_number"), "magic")
        self.assertEqual(server.namespace_for("get_magic_number"), "magic")
        self.assertEqual(server.namespace_for("get_current_time"), "get_current_time")
        self.assertEqual(server.namespace_for("a_tool_long_gone"), "a_tool_long_gone")

    def test_select_tools(self):
        self.assertEqual(server.select_tools(None), list(tools.builtin_tools))
        self.assertEqual(server.select_tools([]), [])
        self.assertEqual(
            server.select_tools(["get_current_time"]), [tools.get_current_time_tool]
        )
        with self.assertRaises(server.HHTcpError) as caught:
            server.select_tools("get_current_time")
        self.assertEqual(caught.exception.code, "bad_tools")
        with self.assertRaises(server.HHTcpError) as caught:
            server.select_tools(["nope"])
        self.assertEqual(caught.exception.code, "unknown_tool")
        self.assertIn("get_magic_number", caught.exception.detail["available"])

    def test_build_tools_merges_and_refuses_collisions(self):
        local = tools.ToolEntry("ask_operator", "", [])
        merged = server.build_tools([tools.get_current_time_tool], [local])
        self.assertEqual(sorted(merged), ["ask_operator", "get_current_time"])
        with self.assertRaises(server.HHTcpError) as caught:
            server.build_tools([tools.get_current_time_tool], [tools.get_current_time_tool])
        self.assertEqual(caught.exception.code, "bad_tools")
        self.assertIn("declared twice", str(caught.exception))

    def test_parse_local_tools(self):
        self.assertEqual(server.parse_local_tools(None), [])
        parsed = server.parse_local_tools(
            [
                {
                    "name": "ask_operator",
                    "description": "Ask the operator.",
                    "params": [
                        {"name": "question", "type": "string", "description": "what"},
                        {"name": "urgent"},  # defaults fill in the rest
                    ],
                    "rollback": True,
                    "external_effects": True,
                }
            ]
        )[0]
        self.assertEqual(parsed.name, "ask_operator")
        self.assertTrue(parsed.is_local)
        self.assertTrue(parsed.remote_rollback)
        self.assertTrue(parsed.external_effects)
        self.assertEqual(
            parsed.params,
            [
                tools.ToolParam("question", "string", "what"),
                tools.ToolParam("urgent", "string", ""),
            ],
        )

    def test_tool_catalogue_describes_the_builtins(self):
        catalogue = {entry["name"]: entry for entry in server.tool_catalogue()}
        self.assertEqual(catalogue["set_magic_number"]["state_namespace"], "magic")
        self.assertFalse(catalogue["set_magic_number"]["rollback"])
        self.assertFalse(catalogue["set_magic_number"]["external_effects"])
        self.assertEqual(
            catalogue["set_magic_number"]["params"],
            [{"name": "magic", "type": "string", "description": "the new magic number"}],
        )
        self.assertEqual(catalogue["get_current_time"]["params"], [])

    def test_every_documented_command_has_a_handler(self):
        names = [entry["command"] for entry in server.COMMANDS]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertTrue(
                callable(getattr(server.HHServer, f"cmd_{name}", None)),
                f"no handler for {name}",
            )
            entry = next(item for item in server.COMMANDS if item["command"] == name)
            self.assertTrue(entry["summary"])
            self.assertIsInstance(entry["fields"], dict)

    def test_apply_overrides_replaces_only_the_tool_axis_given(self):
        block = agent.HHAgent.root(
            "http://x", "k", tools=[tools.get_current_time_tool, local_tool("ask")]
        )
        server.apply_overrides(block, {"tools": ["get_magic_number"]})
        self.assertEqual(sorted(block.tools), ["ask", "get_magic_number"])
        server.apply_overrides(block, {"local_tools": [{"name": "other"}]})
        self.assertEqual(sorted(block.tools), ["get_magic_number", "other"])

    def test_apply_overrides_rejects_bad_values(self):
        block = agent.HHAgent.root("http://x", "k", tools=[])
        for fields in (
            {"verbose": 1},
            {"include_usage": "yes"},
            {"model": ""},
            {"timeout": "soon"},
            {"max_tokens": -1},
            {"max_tokens": 1.5},
            {"summary_model": 5},
        ):
            with self.assertRaises(server.HHTcpError) as caught:
                server.apply_overrides(block, fields)
            self.assertEqual(caught.exception.code, "bad_field")

    def test_connection_helpers(self):
        sent = []

        class Sock:
            def sendall(self, data):
                sent.append(json.loads(data.decode()))

        connection = server.Connection(Sock(), ("127.0.0.1", 1234))
        self.assertTrue(connection.send("hello", n=1))
        self.assertEqual(sent[0]["event"], "hello")
        self.assertEqual(sent[0]["n"], 1)
        self.assertEqual(sent[0]["seq"], 1)
        self.assertGreater(sent[0]["ts"], 0)
        self.assertEqual(connection.events, 1)

        connection.forward({"event": "turn_started", "agent_id": "a1"})
        self.assertEqual(sent[1]["event"], "turn_started")
        self.assertEqual(sent[1]["agent_id"], "a1")
        self.assertEqual(sent[1]["seq"], 2)

        connection.close()
        self.assertFalse(connection.send("after-close"))
        self.assertEqual(len(sent), 2)


class LocalCredentialTests(unittest.TestCase):
    """`local_credentials` is tested through a stub `pathlib`, never `test.py`.

    The shipped `test.py` is gitignored and may do anything at import, so the
    real module is never executed by this suite; the stub below stands in for
    whatever path `local_credentials` decides to read.
    """

    class StubPath:
        """A `pathlib.Path` that reads from wherever `target` points."""

        target = None

        def __init__(self, value):
            self.value = pathlib.Path(value)

        def resolve(self):
            self.value = self.value.resolve()
            return self

        @property
        def parent(self):
            return LocalCredentialTests.StubPath(self.value.parent)

        def __truediv__(self, other):
            return LocalCredentialTests.StubPath(self.value / other)

        def is_file(self):
            return LocalCredentialTests.StubPath.target.is_file()

        def __fspath__(self):
            return str(LocalCredentialTests.StubPath.target)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hh-cred-")
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)

    def write_module(self, text):
        path = self.dir / "test.py"
        path.write_text(text, encoding="utf-8")
        LocalCredentialTests.StubPath.target = path
        return path

    def call(self):
        stub = self.StubPath
        with mock.patch.object(server, "pathlib", types.SimpleNamespace(Path=stub)):
            return server.local_credentials()

    def test_a_missing_file_yields_nothing(self):
        LocalCredentialTests.StubPath.target = self.dir / "absent.py"
        self.assertEqual(self.call(), {})

    def test_a_file_that_will_not_import_yields_nothing(self):
        self.write_module("raise RuntimeError('nope')\n")
        self.assertEqual(self.call(), {})

    def test_endpoint_and_key_are_picked_up(self):
        self.write_module("endpoint = 'http://provider.test'\nkey = 'a-key'\n")
        self.assertEqual(
            self.call(), {"endpoint": "http://provider.test", "key": "a-key"}
        )

    def test_values_that_are_not_strings_are_ignored(self):
        self.write_module("endpoint = 5\nkey = None\nother = 'ignored'\n")
        self.assertEqual(self.call(), {})

    def test_one_field_alone_is_enough(self):
        self.write_module("key = 'a-key'\n")
        self.assertEqual(self.call(), {"key": "a-key"})

    def test_a_path_with_no_module_spec_yields_nothing(self):
        self.write_module("endpoint = 'http://provider.test'\nkey = 'a-key'\n")
        with mock.patch.object(
            server.importlib.util, "spec_from_file_location", return_value=None
        ):
            self.assertEqual(self.call(), {})


class MainTests(unittest.TestCase):
    """`main` is driven with the store and the server replaced by stubs."""

    def setUp(self):
        self.constructed = {}
        state = self.constructed

        class StubStore:
            def __init__(self, path, max_bytes=0):
                state["db"] = path
                state["max_bytes"] = max_bytes

            def size_bytes(self):
                return 123

            def close(self):
                state["store_closed"] = True

        class StubServer:
            def __init__(self, address, defaults=None, store=None):
                state["address"] = address
                state["defaults"] = defaults
                state["store"] = store

            def serve_forever(self):
                state["served"] = True
                raise KeyboardInterrupt

            def server_close(self):
                state["server_closed"] = True

        self.stub_store = StubStore
        self.stub_server = StubServer

    def run_main(self, argv, credentials=None):
        output = io.StringIO()
        with mock.patch.object(server, "HHStore", self.stub_store), mock.patch.object(
            server, "HHServer", self.stub_server
        ), mock.patch.object(server, "local_credentials", return_value=credentials or {}):
            with contextlib.redirect_stdout(output):
                server.main(argv)
        return output.getvalue()

    def test_flags_reach_the_store_and_the_server(self):
        text = self.run_main(
            [
                "--host", "127.0.0.1", "--port", "0",
                "--endpoint", "http://provider.test", "--key", "a-key",
                "--model", "a-model", "--db", "/tmp/x.db", "--max-db-bytes", "2048",
            ]
        )
        self.assertIn("listening on 127.0.0.1:0", text)
        self.assertIn("model default a-model", text)
        self.assertIn(f"protocol {server.PROTOCOL_VERSION}", text)
        self.assertIn("credentials: flags", text)
        self.assertIn("store: /tmp/x.db (123 bytes, limit 2048)", text)
        self.assertIn("shutting down", text)
        self.assertEqual(self.constructed["db"], "/tmp/x.db")
        self.assertEqual(self.constructed["max_bytes"], 2048)
        self.assertEqual(
            self.constructed["defaults"],
            {"endpoint": "http://provider.test", "key": "a-key", "model": "a-model"},
        )
        self.assertEqual(self.constructed["address"], ("127.0.0.1", 0))
        self.assertTrue(self.constructed["served"])
        self.assertTrue(self.constructed["server_closed"])
        self.assertTrue(self.constructed["store_closed"])

    def test_credentials_fall_back_to_the_client_file(self):
        text = self.run_main(
            ["--port", "0"],
            credentials={"endpoint": "http://from-test", "key": "from-test"},
        )
        self.assertIn("credentials: test.py", text)
        self.assertEqual(self.constructed["defaults"]["endpoint"], "http://from-test")
        self.assertEqual(self.constructed["defaults"]["key"], "from-test")

    def test_flags_win_over_the_client_file(self):
        text = self.run_main(
            ["--port", "0", "--endpoint", "http://gave", "--key", "flag-key"],
            credentials={"endpoint": "http://from-test", "key": "from-test"},
        )
        self.assertIn("credentials: flags", text)
        self.assertEqual(self.constructed["defaults"]["endpoint"], "http://gave")
        self.assertEqual(self.constructed["defaults"]["key"], "flag-key")

    def test_no_credentials_still_starts_and_warns(self):
        text = self.run_main(["--port", "0"])
        self.assertIn("warning: no credentials", text)
        self.assertIsNone(self.constructed["defaults"]["endpoint"])
        self.assertIsNone(self.constructed["defaults"]["key"])
        self.assertTrue(self.constructed["served"])

    def test_a_store_that_cannot_open_exits(self):
        def explode(path, max_bytes=0):
            raise store.HHStoreError(f"{path} is already open in another process")

        output = io.StringIO()
        with mock.patch.object(server, "HHStore", explode), mock.patch.object(
            server, "HHServer", self.stub_server
        ), mock.patch.object(server, "local_credentials", return_value={}), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as caught:
                server.main(["--port", "0", "--db", "/tmp/taken.db"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("already open", output.getvalue())

    def test_the_default_database_lives_beside_the_sources(self):
        self.run_main(["--port", "0"])
        self.assertTrue(
            str(self.constructed["db"]).endswith("harness.db")
        )
        self.assertEqual(
            self.constructed["max_bytes"], server.DEFAULT_MAX_DB_BYTES
        )


if __name__ == "__main__":
    unittest.main()
