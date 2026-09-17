"""Tests for `src/agent.py`: the block/chain model, turns, tools, state, rollback.

Every test drives a real `HHAgent` against the scripted fake provider in
`tests/support.py`, so HTTP, SSE parsing and the whole turn machinery run for
real; only the model itself is fake.
"""

import json
import threading
import time
import unittest
from unittest import mock

import requests

from tests.support import (
    HHTestCase,
    Response,
    agent,
    delta,
    finish,
    inner,
    local_tool,
    param,
    pick,
    server_tool,
    tools,
    usage,
)

SUMMARY_PROMPT = agent.SUMMARY_PROMPT


# --- tools several tests share ------------------------------------------


def remember_hook(context, key, value):
    context.state[key] = value
    return f"remembered {key}={value}"


def forget_hook(context, key):
    context.state.pop(key, None)
    return f"forgot {key}"


def append_hook(context, item):
    context.state.setdefault("items", []).append(item)
    return f"added {item}"


def boom_hook(context):
    raise ValueError("tool exploded")


REMEMBER = server_tool(
    "remember", remember_hook, params=[param("key"), param("value")], namespace="mem"
)
FORGET = server_tool("forget", forget_hook, params=[param("key")], namespace="mem")
APPEND = server_tool("append", append_hook, params=[param("item")], namespace="mem")
BOOM = server_tool("boom", boom_hook)


# --- ids, arguments, stream reassembly ----------------------------------


class IdAndParsingTests(unittest.TestCase):
    def test_new_id_is_prefixed_and_unique(self):
        first = agent.new_id()
        self.assertTrue(first.startswith("agent-"))
        self.assertEqual(len(first), len("agent-") + 12)
        int(first[len("agent-") :], 16)  # must be hex
        self.assertNotEqual(first, agent.new_id())

    def test_parse_arguments_json_string(self):
        self.assertEqual(agent._parse_arguments("t", '{"a": 1}'), ({"a": 1}, None))

    def test_parse_arguments_empty_strings_are_an_empty_call(self):
        self.assertEqual(agent._parse_arguments("t", ""), ({}, None))
        self.assertEqual(agent._parse_arguments("t", "   "), ({}, None))

    def test_parse_arguments_bad_json_explains_itself(self):
        arguments, error = agent._parse_arguments("t", "{oops")
        self.assertIsNone(arguments)
        self.assertIn("not valid JSON", error)
        self.assertIn("'t'", error)

    def test_parse_arguments_dict_passes_through(self):
        self.assertEqual(agent._parse_arguments("t", {"a": 1}), ({"a": 1}, None))

    def test_parse_arguments_other_scalars_become_empty(self):
        # anything that is neither a string nor a dict is dropped: only a
        # JSON string that parses to a scalar can reach the "must be a JSON
        # object" branch in `_run_tools`
        self.assertEqual(agent._parse_arguments("t", 5), ({}, None))
        self.assertEqual(agent._parse_arguments("t", None), ({}, None))
        self.assertEqual(agent._parse_arguments("t", ["a"]), ({}, None))

    def test_same_value_never_raises_and_never_lies(self):
        self.assertTrue(agent._same_value(1, 1))
        self.assertFalse(agent._same_value(1, 2))

        class Uncomparable:
            def __eq__(self, other):
                raise RuntimeError("cannot compare")

        class NoTruth:
            def __bool__(self):
                raise ValueError("truth value is ambiguous")

        class Weird:
            def __eq__(self, other):
                return NoTruth()

        # both failures mean "assume changed": a delta superset is harmless
        self.assertFalse(agent._same_value(Uncomparable(), Uncomparable()))
        self.assertFalse(agent._same_value(Weird(), Weird()))

    def test_run_hook_reports_a_missing_hook(self):
        tool = tools.ToolEntry(name="local", description="", params=[])
        context = tools.ToolContext(
            tool=tool, agent=None, call_id="c", arguments={}, raw_arguments=None, state={}
        )
        self.assertEqual(
            agent._run_hook(tool, context),
            ("Error: tool 'local' has no hook", "no_hook"),
        )

    def test_run_hook_bad_signature_is_bad_arguments(self):
        context = tools.ToolContext(
            tool=REMEMBER,
            agent=None,
            call_id="c",
            arguments={"nonsense": 1},
            raw_arguments="{}",
            state={},
        )
        text, error = agent._run_hook(REMEMBER, context)
        self.assertEqual(error, "bad_arguments")
        self.assertIn("bad arguments for 'remember'", text)

    def test_run_hook_exception_is_contained(self):
        context = tools.ToolContext(
            tool=BOOM, agent=None, call_id="c", arguments={}, raw_arguments="{}", state={}
        )
        text, error = agent._run_hook(BOOM, context)
        self.assertEqual(error, "tool_raised")
        self.assertEqual(text, "Error: tool 'boom' raised ValueError: tool exploded")

    def test_not_run_codes_are_just_the_two(self):
        self.assertEqual(set(agent.NOT_RUN), {"bad_arguments", "no_hook"})

    def test_turn_absorbs_fragmented_tool_calls(self):
        turn = agent._Turn()
        turn.absorb(inner(delta(content="thinking")))
        turn.absorb(
            inner(delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "ge"}}]))
        )
        turn.absorb(
            inner(
                delta(
                    tool_calls=[{"index": 0, "function": {"name": "t_time", "arguments": '{"a"'}}]
                )
            )
        )
        turn.absorb(
            inner(delta(tool_calls=[{"index": 0, "function": {"arguments": ": 1}"}}]))
        )
        self.assertEqual(turn.content, "thinking")
        self.assertEqual(
            turn.message(),
            {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": '{"a": 1}'},
                    }
                ],
            },
        )

    def test_turn_index_defaults_to_zero(self):
        turn = agent._Turn()
        turn.absorb(inner(delta(tool_calls=[{"id": "c1", "function": {"name": "x"}}])))
        self.assertEqual(turn.message()["tool_calls"][0]["function"]["name"], "x")

    def test_turn_message_without_calls_has_content_only(self):
        turn = agent._Turn()
        turn.absorb(inner(delta(content="hi")))
        self.assertEqual(turn.message(), {"role": "assistant", "content": "hi"})

    def test_turn_message_defaults_empty_arguments_and_omits_empty_content(self):
        turn = agent._Turn()
        turn.absorb(
            inner(delta(tool_calls=[{"index": 0, "id": "c", "function": {"name": "x"}}]))
        )
        message = turn.message()
        self.assertNotIn("content", message)
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], "{}")

    def test_turn_sorts_calls_by_index(self):
        turn = agent._Turn()
        turn.absorb(
            inner(delta(tool_calls=[{"index": 1, "id": "b", "function": {"name": "second"}}]))
        )
        turn.absorb(
            inner(delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "first"}}]))
        )
        names = [call["function"]["name"] for call in turn.message()["tool_calls"]]
        self.assertEqual(names, ["first", "second"])

    def test_message_summary_shape(self):
        summary = agent.HHAgent._message_summary(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "t", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ]
        )
        self.assertEqual(summary[0], {"role": "user", "chars": 5})
        self.assertEqual(summary[1], {"role": "assistant", "chars": 0, "tool_calls": ["t"]})
        self.assertEqual(summary[2], {"role": "tool", "chars": 1, "tool_call_id": "c1"})

    def test_sse_payloads_unwrap_only_data_lines(self):
        class FakeResponse:
            def iter_lines(self):
                return iter(
                    [
                        b"",
                        b": keep-alive",
                        b"event: message",
                        b'data: {"a": 1}',
                        b"data: [DONE]",
                        "data: 你好".encode("utf-8"),
                    ]
                )

        self.assertEqual(
            list(agent.HHAgent._sse_payloads(FakeResponse())),
            ['{"a": 1}', "[DONE]", "你好"],
        )


# --- roots, forks, lineage ----------------------------------------------


class RootAndForkTests(HHTestCase):
    def test_root_defaults(self):
        root = agent.HHAgent.root("http://example.test/provider/", "k")
        self.assertTrue(root.id.startswith("agent-"))
        self.assertIsNone(root.parent)
        self.assertEqual(root.prompt, "")
        self.assertEqual(root.messages, [])
        self.assertEqual(root.text, "")
        self.assertIsNone(root.error)
        self.assertIsNone(root.outcome)
        self.assertFalse(root.dirty)
        self.assertFalse(root.running)
        self.assertEqual(root.endpoint, "http://example.test/provider")
        self.assertEqual(root.key, "k")
        self.assertEqual(root.model, agent.DEFAULT_MODEL)
        self.assertEqual(root.timeout, agent.DEFAULT_TIMEOUT)
        self.assertEqual(root.local_timeout, agent.DEFAULT_LOCAL_TIMEOUT)
        self.assertEqual(root.summary_model, "")
        self.assertTrue(root.include_usage)
        self.assertFalse(root.verbose)
        self.assertEqual(root.state_deltas, {})
        self.assertEqual(
            sorted(root.tools),
            [
                "get_current_time",
                "get_magic_number",
                "get_system_info",
                "set_magic_number",
                "web_fetch",
            ],
        )

    def test_root_keeps_the_id_it_is_given(self):
        self.assertEqual(self.root(id="mine").id, "mine")

    def test_root_tools_are_keyed_by_name(self):
        root = self.root(tools=[BOOM, REMEMBER])
        self.assertEqual(sorted(root.tools), ["boom", "remember"])
        self.assertIs(root.tools["boom"], BOOM)

    def test_fork_needs_a_non_empty_string_prompt(self):
        root = self.root()
        for bad in ("", None, 5, {"a": 1}):
            with self.assertRaises(agent.HHAgentError):
                root.fork(bad)

    def test_fork_refuses_a_dirty_parent(self):
        child = self.root().fork("one")
        self.assertTrue(child.dirty)
        with self.assertRaisesRegex(agent.HHAgentError, "has not finished"):
            child.fork("two")

    def test_fork_child_shape(self):
        root = self.root()
        child = root.fork("hello", id="child")
        self.assertEqual(child.id, "child")
        self.assertIs(child.parent, root)
        self.assertEqual(child.prompt, "hello")
        self.assertEqual(child.messages, [{"role": "user", "content": "hello"}])
        self.assertTrue(child.dirty)
        self.assertEqual(child.context(), [{"role": "user", "content": "hello"}])
        self.assertEqual(child.state_deltas, {})
        self.assertFalse(root.dirty)

    def test_fork_inherits_settings_and_can_override_them(self):
        def hook(event):
            pass

        root = self.root(
            model="parent-model",
            timeout=7.0,
            local_timeout=3.0,
            summary_model="summariser",
            include_usage=False,
            verbose=True,
            on_event=hook,
            tools=[REMEMBER],
        )
        child = root.fork("hi")
        self.assertEqual(child.endpoint, root.endpoint)
        self.assertEqual(child.key, root.key)
        self.assertEqual(child.model, "parent-model")
        self.assertEqual(child.timeout, 7.0)
        self.assertEqual(child.local_timeout, 3.0)
        self.assertEqual(child.summary_model, "summariser")
        self.assertFalse(child.include_usage)
        self.assertTrue(child.verbose)
        self.assertIs(child.on_event, hook)
        self.assertEqual(sorted(child.tools), ["remember"])
        self.assertIs(root.fork("hi", on_event=print).on_event, print)

    def test_fork_copies_the_tool_table(self):
        root = self.root(tools=[REMEMBER])
        child = root.fork("hi")
        child.tools.pop("remember")
        self.assertIn("remember", root.tools)

    def test_lineage_path_and_depth(self):
        self.provider.text("a")
        root = self.root(tools=[])
        first = root.fork("one")
        self.run_turn(first)
        self.provider.text("b")
        second = first.fork("two")
        self.assertEqual([node.id for node in second.lineage()], [root.id, first.id, second.id])
        self.assertEqual(second.path(), [root.id, first.id, second.id])
        self.assertEqual((root.depth, first.depth, second.depth), (0, 1, 2))

    def test_context_concatenates_the_chain(self):
        self.provider.text("a")
        root = self.root(tools=[])
        first = root.fork("one")
        self.run_turn(first)
        second = first.fork("two")
        self.assertEqual(
            second.context(),
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "two"},
            ],
        )

    def test_state_namespaces_are_in_first_seen_order(self):
        root = self.root(tools=[])
        child = root.fork("hi")
        self.assertEqual(child.state_namespaces(), [])
        root.state_deltas["zeta"] = agent.StateDelta(changed={"k": 1})
        root.state_deltas["alpha"] = agent.StateDelta(changed={"k": 2})
        self.assertEqual(child.state_namespaces(), ["zeta", "alpha"])

    def test_fork_from_a_failed_block_is_allowed(self):
        root = self.root(tools=[])
        block = root.fork("hi")
        self.provider.error(500)
        thread = self.turn(block)
        thread.join()
        self.assertEqual(block.outcome, "failed")
        self.assertFalse(block.dirty)
        self.assertEqual(block.fork("again").parent, block)


# --- a plain turn --------------------------------------------------------


class TurnLifecycleTests(HHTestCase):
    def test_simple_text_turn(self):
        self.provider.text("hello world", chunk_size=4)
        root = self.root(tools=[])
        block = root.fork("hi there")
        events = []
        reply = self.run_turn(block, on_event=events.append)

        self.assertEqual(reply, "hello world")
        self.assertEqual(block.text, "hello world")
        self.assertEqual(block.outcome, "ok")
        self.assertFalse(block.dirty)
        self.assertIsNone(block.error)
        self.assertFalse(block.running)
        self.assertEqual(
            block.messages,
            [
                {"role": "user", "content": "hi there"},
                {"role": "assistant", "content": "hello world"},
            ],
        )

        started = pick(events, "turn_started")[0]
        self.assertEqual(started["prompt"], "hi there")
        self.assertEqual(started["prompt_chars"], 8)
        self.assertEqual(started["depth"], 1)
        self.assertEqual(started["path"], [root.id, block.id])
        self.assertEqual(started["model"], "test-model")
        self.assertEqual(started["tools"], [])
        self.assertEqual(started["context_len"], 1)
        self.assertTrue(started["include_usage"])
        self.assertFalse(started["verbose"])
        for event in events:
            self.assertEqual(event["agent_id"], block.id)

        self.assertEqual(
            [event["source"] for event in pick(events, "history_appended")],
            ["fork", "assistant"],
        )
        appended = pick(events, "history_appended")
        self.assertEqual(appended[0]["role"], "user")
        self.assertEqual(appended[0]["preview"], "hi there")
        self.assertEqual(appended[0]["chars"], 8)
        self.assertEqual(appended[0]["messages"], 1)
        self.assertEqual(appended[0]["context_len"], 1)
        self.assertEqual(appended[1]["role"], "assistant")
        self.assertEqual(appended[1]["preview"], "hello world")
        self.assertEqual(appended[1]["messages"], 2)
        self.assertEqual(appended[1]["context_len"], 2)

        finished = pick(events, "turn_finished")[0]
        self.assertEqual(finished["text"], "hello world")
        self.assertEqual(finished["text_chars"], 11)
        self.assertEqual(finished["rounds"], 1)
        self.assertEqual(finished["tool_calls"], 0)
        self.assertEqual(finished["messages"], 2)
        self.assertEqual(finished["context_len"], 2)
        self.assertFalse(finished["dirty"])

        assistant = pick(events, "assistant_message")[0]
        self.assertEqual(assistant["round"], 1)
        self.assertEqual(assistant["content"], "hello world")
        self.assertEqual(assistant["tool_calls"], [])

        deltas = pick(events, "content_delta")
        self.assertEqual("".join(event["text"] for event in deltas), "hello world")
        self.assertEqual(deltas[0]["round"], 1)
        for event in deltas:
            self.assertEqual(event["chars"], len(event["text"]))

        request = pick(events, "request_started")[0]
        self.assertEqual(request["round"], 1)
        self.assertEqual(request["url"], f"{self.provider.endpoint}/v1/chat/completions")
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["messages"], 1)
        self.assertEqual(request["tools"], 0)
        self.assertEqual(request["depth"], 1)
        self.assertGreater(request["request_bytes"], 0)

        payload = pick(events, "request_payload")[0]
        self.assertEqual(payload["round"], 1)
        self.assertEqual(payload["messages"], [{"role": "user", "chars": 8}])

        self.assertEqual(pick(events, "response_received")[0]["status"], 200)
        done = pick(events, "request_finished")[0]
        self.assertEqual(done["round"], 1)
        self.assertEqual(done["status"], 200)
        self.assertGreater(done["chunks"], 1)
        self.assertEqual(done["finish_reason"], "stop")
        self.assertFalse(done["cancelled"])
        self.assertIsNotNone(done["first_chunk_ms"])

    def test_request_headers_and_path(self):
        self.provider.text("ok")
        self.run_turn(self.root(tools=[]).fork("hi"))
        request = self.provider.requests[0]
        self.assertEqual(request.path, "/v1/chat/completions")
        self.assertEqual(
            request.headers["Authorization"], "Bearer test-key-that-is-not-a-secret"
        )
        self.assertEqual(request.headers["Content-Type"], "application/json")

    def test_request_payload_shape_with_tools_and_usage(self):
        self.provider.text("ok")
        self.run_turn(self.root(tools=[REMEMBER]).fork("hi"))
        payload = self.provider.last_payload()
        self.assertEqual(payload["model"], "test-model")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(
            payload["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "remember",
                        "description": "remember test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string", "description": "a parameter"},
                                "value": {"type": "string", "description": "a parameter"},
                            },
                            "required": ["key", "value"],
                        },
                    },
                }
            ],
        )

    def test_include_usage_false_drops_stream_options(self):
        self.provider.text("ok")
        self.run_turn(self.root(tools=[], include_usage=False).fork("hi"))
        self.assertNotIn("stream_options", self.provider.last_payload())

    def test_no_tools_key_when_the_block_has_none(self):
        self.provider.text("ok")
        self.run_turn(self.root(tools=[]).fork("hi"))
        self.assertNotIn("tools", self.provider.last_payload())

    def test_usage_event_and_finish_reason(self):
        self.provider.push(
            Response(chunks=[delta(content="hi"), finish("length"), usage(11, 22), "[DONE]"])
        )
        thread = self.turn(self.root(tools=[]).fork("hi"))
        thread.join()
        self.assertEqual(thread.text, "hi")
        self.assertEqual(
            thread.last("usage")["usage"],
            {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
        )
        self.assertEqual(thread.last("request_finished")["finish_reason"], "length")

    def test_reasoning_deltas_are_reported_but_not_part_of_the_reply(self):
        self.provider.push(
            Response(
                chunks=[
                    delta(reasoning="step one; "),
                    delta(reasoning_content="step two"),
                    delta(content="answer"),
                    finish(),
                    "[DONE]",
                ]
            )
        )
        thread = self.turn(self.root(tools=[]).fork("hi"))
        thread.join()
        self.assertEqual(thread.text, "answer")
        reasoning = pick(thread.events, "reasoning_delta")
        self.assertEqual([event["text"] for event in reasoning], ["step one; ", "step two"])
        self.assertEqual([event["chars"] for event in reasoning], [10, 8])

    def test_verbose_emits_raw_chunks(self):
        self.provider.text("hi")
        thread = self.turn(self.root(tools=[], verbose=True).fork("hi"))
        thread.join()
        chunks = pick(thread.events, "sse_chunk")
        self.assertEqual(len(chunks), 2)  # the content delta and the finish chunk
        self.assertEqual(chunks[0]["index"], 1)
        self.assertEqual(chunks[0]["chunk"]["choices"][0]["delta"], {"content": "hi"})

    def test_unparsed_stream_lines_are_reported(self):
        self.provider.push(
            Response(
                chunks=[
                    "x" * 600,  # the reported line is capped at 500 chars
                    "this is not json",
                    delta(content="fine"),
                    finish(),
                    "[DONE]",
                ]
            )
        )
        thread = self.turn(self.root(tools=[]).fork("hi"))
        thread.join()
        unparsed = pick(thread.events, "sse_unparsed")
        self.assertEqual([event["line"] for event in unparsed], ["x" * 500, "this is not json"])
        self.assertEqual(thread.text, "fine")

    def test_comment_lines_are_ignored(self):
        self.provider.push(
            Response(chunks=[": keep-alive", delta(content="hi"), finish(), "[DONE]"])
        )
        thread = self.turn(self.root(tools=[]).fork("hi"))
        thread.join()
        self.assertEqual(thread.text, "hi")
        self.assertEqual(pick(thread.events, "sse_unparsed"), [])

    def test_missing_done_marker_still_finishes(self):
        self.provider.push(Response(chunks=[delta(content="hi"), finish()]))
        thread = self.turn(self.root(tools=[]).fork("hi"))
        thread.join()
        self.assertIsNone(thread.error)
        self.assertEqual(thread.text, "hi")
        self.assertEqual(thread.last("request_finished")["finish_reason"], "stop")

    def test_unicode_survives_the_round_trip(self):
        self.provider.text("你好，世界 🌍")
        block = self.root(tools=[]).fork("说你好")
        self.assertEqual(self.run_turn(block), "你好，世界 🌍")
        self.assertIn("说你好".encode("utf-8"), self.provider.requests[0].raw)

    def test_empty_reply_is_a_successful_turn(self):
        self.provider.push(Response(chunks=[finish(), "[DONE]"]))
        block = self.root(tools=[]).fork("hi")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(thread.text, "")
        self.assertEqual(block.outcome, "ok")
        self.assertEqual(block.messages[-1], {"role": "assistant", "content": ""})

    def test_root_refuses_to_run_and_records_the_failure(self):
        root = self.root(tools=[])
        events = []
        with self.assertRaisesRegex(agent.HHAgentError, "is a root; fork from it first"):
            list(root.stream(events.append))
        self.assertEqual(self.provider.count, 0)
        self.assertFalse(root.dirty)
        self.assertEqual(root.outcome, "failed")
        self.assertIn("is a root; fork from it first", root.error)
        # the finally block leaves a note, as it does for any failed turn
        note = root.messages[-1]
        self.assertEqual(note["role"], "user")
        self.assertTrue(note["content"].startswith("[harness] the previous turn failed"))
        self.assertIn(root.error, note["content"])
        self.assertIn("No summary was available.", note["content"])
        self.assertEqual(pick(events, "failure_summary_started"), [])
        self.assertEqual(pick(events, "turn_failed")[0]["error_type"], "HHAgentError")

    def test_a_finished_block_refuses_to_run_again(self):
        self.provider.text("ok")
        block = self.root(tools=[]).fork("hi")
        self.run_turn(block)
        with self.assertRaisesRegex(agent.HHAgentError, "already finished its turn"):
            list(block.stream())
        self.assertEqual(self.provider.count, 1)  # nothing new was sent

    def test_a_running_block_refuses_a_second_turn(self):
        self.provider.push(Response.steady(pieces=6, gap=0.02))
        block = self.root(tools=[]).fork("hi")
        thread = self.turn(block)
        thread.wait_event("content_delta")
        self.assertTrue(block.running)
        with self.assertRaisesRegex(agent.HHAgentError, "is already running"):
            list(block.stream())
        thread.join()
        self.assertEqual(block.outcome, "ok")
        self.assertEqual(thread.text, "x" * 4200)

    def test_stream_is_lazy(self):
        self.provider.text("ok")
        block = self.root(tools=[]).fork("hi")
        stream = block.stream()
        self.assertEqual(self.provider.count, 0)
        self.assertFalse(block.running)
        self.assertEqual("".join(stream), "ok")

    def test_chat_returns_the_joined_stream(self):
        self.provider.text("chunky", chunk_size=2)
        self.assertEqual(self.root(tools=[]).fork("hi").chat(), "chunky")

    def test_a_broken_event_hook_cannot_kill_a_turn(self):
        def broken(event):
            raise RuntimeError("listener is broken")

        self.provider.text("ok")
        block = self.root(tools=[], on_event=broken).fork("hi")
        self.assertEqual(block.chat(), "ok")
        self.assertEqual(block.outcome, "ok")

    def test_the_per_call_event_hook_wins_over_the_block_one(self):
        self.provider.text("ok")
        collected = []
        block = self.root(tools=[], on_event=lambda event: None).fork("hi")
        self.assertEqual("".join(block.stream(collected.append)), "ok")
        self.assertIn("turn_finished", [event["event"] for event in collected])

    def test_abandoned_turn_is_recorded_and_noted(self):
        self.provider.push(Response.steady(pieces=6, gap=0.02))
        block = self.root(tools=[]).fork("hi")
        events = []
        stream = block.stream(events.append)
        first = next(stream)
        self.assertTrue(first)
        stream.close()

        self.assertFalse(block.dirty)
        self.assertEqual(block.outcome, "abandoned")
        self.assertEqual(block.text, first)  # partial text is kept on the block
        self.assertEqual(pick(events, "turn_cancelled"), [])
        self.assertEqual(pick(events, "turn_failed"), [])
        self.assertEqual(pick(events, "failure_summary_started"), [])
        note = block.messages[-1]
        self.assertEqual(note["role"], "user")
        self.assertIn("the previous turn was abandoned before it finished", note["content"])
        self.assertIn("the state it changed was discarded", note["content"])

    def test_messages_are_not_polluted_by_a_partial_reply(self):
        self.provider.push(Response.steady(pieces=6, gap=0.02))
        block = self.root(tools=[]).fork("hi")
        stream = block.stream()
        next(stream)
        stream.close()
        self.assertEqual(block.messages[0], {"role": "user", "content": "hi"})
        self.assertEqual(block.messages[1]["role"], "user")
        self.assertTrue(block.messages[1]["content"].startswith("[harness]"))
        self.assertEqual(len(block.messages), 2)

    def test_http_error_fails_the_turn(self):
        self.provider.error(500, b"no capacity")
        block = self.root(tools=[]).fork("hi")
        thread = self.turn(block)
        thread.join()
        self.assertIsInstance(thread.error, agent.HHAgentError)
        self.assertIn("500", str(thread.error))
        self.assertIn("no capacity", str(thread.error))
        self.assertEqual(block.outcome, "failed")
        self.assertEqual(block.error, str(thread.error))
        self.assertFalse(block.dirty)
        self.assertEqual(pick(thread.events, "request_finished"), [])
        self.assertEqual(pick(thread.events, "response_received")[0]["status"], 500)
        failed = thread.last("turn_failed")
        self.assertEqual(failed["error_type"], "HHAgentError")
        self.assertIn("no capacity", failed["error"])
        self.assertFalse(failed["dirty"])

    def test_transport_error_at_request_time(self):
        block = self.root(tools=[], endpoint="http://127.0.0.1:1").fork("hi")
        thread = self.turn(block)
        thread.join()
        self.assertIsInstance(thread.error, agent.HHAgentError)
        self.assertIn("request to http://127.0.0.1:1 failed", str(thread.error))
        failure = thread.last("request_failed")
        self.assertEqual(failure["round"], 1)
        self.assertIn("error_type", failure)
        self.assertEqual(thread.last("turn_failed")["error_type"], "HHAgentError")

    def test_transport_error_mid_stream(self):
        self.provider.push(Response.text("y" * 4000, chunk_size=500, truncate=True))
        block = self.root(tools=[]).fork("hi")
        thread = self.turn(block)
        thread.join()
        self.assertIsInstance(thread.error, agent.HHAgentError)
        self.assertIn("stream from", str(thread.error))
        failure = thread.last("request_failed")
        self.assertGreater(failure["chunks"], 0)
        self.assertIn(
            failure["error_type"],
            {"ChunkedEncodingError", "ConnectionError", "ProtocolError"},
        )
        self.assertEqual(block.outcome, "failed")
        self.assertTrue(block.text.startswith("y"))  # partial content was kept

    def test_cancel_idle_is_a_no_op(self):
        block = self.root(tools=[]).fork("hi")
        self.assertFalse(block.cancel())
        self.assertFalse(block.running)

    def test_cancel_stops_a_running_turn(self):
        self.provider.push(Response.steady(pieces=8, gap=0.01))
        block = self.root(tools=[]).fork("hi")
        thread = self.turn(block)
        thread.wait_event("content_delta")
        self.assertTrue(block.running)
        started = time.monotonic()
        self.assertTrue(block.cancel())
        thread.join(5.0)
        self.assertLess(time.monotonic() - started, 3.0)

        self.assertIsInstance(thread.error, agent.HHAgentCancelled)
        self.assertIsInstance(thread.error, agent.HHAgentError)
        self.assertEqual(block.outcome, "cancelled")
        self.assertIn("cancelled by caller", block.error)
        self.assertFalse(block.dirty)
        self.assertFalse(block.running)
        cancelled = thread.last("turn_cancelled")
        self.assertIn("cancelled by caller", cancelled["error"])
        self.assertTrue(cancelled["text"].startswith("x"))
        self.assertFalse(cancelled["dirty"])
        self.assertIn(
            "the previous turn was cancelled by the caller", block.messages[-1]["content"]
        )
        self.assertEqual(pick(thread.events, "turn_failed"), [])
        self.assertEqual(pick(thread.events, "failure_summary_started"), [])

    def test_a_stale_cancel_does_not_poison_the_next_turn(self):
        self.provider.text("ok")
        block = self.root(tools=[]).fork("hi")
        block._cancel.set()
        self.assertEqual(block.chat(), "ok")
        self.assertEqual(block.outcome, "ok")

    def test_cancel_after_finishing_returns_false(self):
        self.provider.text("ok")
        block = self.root(tools=[]).fork("hi")
        self.run_turn(block)
        self.assertFalse(block.cancel())


# --- tool rounds ---------------------------------------------------------


class ToolRoundTests(HHTestCase):
    def test_one_tool_round(self):
        self.provider.script(
            Response.tool_call("set_magic_number", {"magic": "7"}),
            Response.text("all set"),
        )
        root = self.root()
        block = root.fork("set the magic number to 7")
        events = []
        thread = self.turn(block, on_event=events.append)
        thread.join()
        self.assertIsNone(thread.error)
        self.assertEqual(thread.text, "all set")

        self.assertEqual(
            [message["role"] for message in block.messages],
            ["user", "assistant", "tool", "assistant"],
        )
        call = block.messages[1]["tool_calls"][0]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["function"]["name"], "set_magic_number")
        self.assertEqual(call["function"]["arguments"], '{"magic": "7"}')
        self.assertEqual(block.messages[2]["tool_call_id"], "call_1")
        self.assertEqual(block.messages[2]["content"], "Magic is set to 7")
        self.assertEqual(block.messages[3]["content"], "all set")

        requested = thread.last("tool_call_requested")
        self.assertEqual(requested["round"], 1)
        self.assertEqual(requested["call_id"], "call_1")
        self.assertEqual(requested["name"], "set_magic_number")
        self.assertEqual(requested["raw_arguments"], '{"magic": "7"}')
        self.assertTrue(requested["known"])
        self.assertEqual(thread.last("tool_call_started")["name"], "set_magic_number")

        finished = thread.last("tool_call_finished")
        self.assertTrue(finished["ok"])
        self.assertIsNone(finished["error"])
        self.assertEqual(finished["result"], "Magic is set to 7")
        self.assertEqual(finished["result_chars"], len("Magic is set to 7"))

        loaded = thread.last("state_loaded")
        self.assertEqual(loaded["tool"], "set_magic_number")
        self.assertEqual(loaded["state_namespace"], "magic")
        self.assertEqual(loaded["keys"], [])
        self.assertEqual(loaded["inherited"], 0)

        # the second request carries the assistant call and its result
        second = self.provider.payloads()[1]
        self.assertEqual(
            [message["role"] for message in second["messages"]],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(second["messages"][1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(second["messages"][2]["content"], "Magic is set to 7")

        done = thread.last("turn_finished")
        self.assertEqual(done["rounds"], 2)
        self.assertEqual(done["tool_calls"], 1)
        self.assertEqual(len(pick(events, "request_started")), 2)

    def test_two_calls_in_one_round(self):
        self.provider.script(
            Response.tool_calls(
                [
                    ("remember", {"key": "a", "value": "1"}),
                    ("remember", {"key": "b", "value": "2"}),
                ]
            ),
            Response.text("remembered both"),
        )
        block = self.root(tools=[REMEMBER]).fork("remember two things")
        thread = self.turn(block)
        thread.join()
        self.assertIsNone(thread.error)
        self.assertEqual(
            [message["role"] for message in block.messages],
            ["user", "assistant", "tool", "tool", "assistant"],
        )
        self.assertEqual(len(block.messages[1]["tool_calls"]), 2)
        self.assertEqual(
            [message["content"] for message in block.messages[2:4]],
            ["remembered a=1", "remembered b=2"],
        )
        self.assertEqual(thread.last("turn_finished")["tool_calls"], 2)
        self.assertEqual(
            [event["call_id"] for event in pick(thread.events, "tool_call_finished")],
            ["call_1", "call_2"],
        )
        self.assertEqual(block.merged_state("mem"), {"a": "1", "b": "2"})

    def test_fragmented_arguments_are_reassembled(self):
        arguments = {"key": "fragment", "value": "piece-by-piece"}
        self.provider.script(
            Response.tool_calls([("remember", arguments)], arguments_split=4),
            Response.text("ok"),
        )
        block = self.root(tools=[REMEMBER]).fork("go")
        thread = self.turn(block)
        thread.join()
        self.assertIsNone(thread.error)
        # `remember` stores by argument, so the state is keyed the other way round
        self.assertEqual(block.merged_state("mem"), {"fragment": "piece-by-piece"})
        self.assertEqual(
            block.messages[1]["tool_calls"][0]["function"]["arguments"],
            json.dumps(arguments),
        )

    def test_tool_name_split_across_deltas(self):
        self.provider.script(
            Response.tool_calls([("set_magic_number", {"magic": "9"})], name_split=True),
            Response.text("ok"),
        )
        block = self.root().fork("go")
        thread = self.turn(block)
        thread.join()
        self.assertIsNone(thread.error)
        self.assertEqual(block.merged_state("magic"), {"magic": "9"})

    def test_an_empty_argument_string_becomes_an_empty_object(self):
        self.provider.script(
            Response.tool_calls([("set_magic_number", "")]),
            Response.text("ok"),
        )
        block = self.root().fork("go")
        thread = self.turn(block)
        thread.join()
        # the provider never sent an argument string, so the assistant message
        # it is remembered with carries "{}", which parses back to no arguments
        self.assertEqual(
            block.messages[1]["tool_calls"][0]["function"]["arguments"], "{}"
        )
        self.assertEqual(thread.last("tool_call_requested")["raw_arguments"], "{}")
        # the builtin indexes its argument, so a missing one is a raised error
        # rather than a bad binding — the failure mode is still contained
        finished = thread.last("tool_call_finished")
        self.assertFalse(finished["ok"])
        self.assertEqual(finished["error"], "tool_raised")
        self.assertIn("KeyError", finished["result"])

    def test_a_hook_that_cannot_bind_its_arguments(self):
        self.provider.script(
            Response.tool_calls([("remember", "")]),
            Response.text("ok"),
        )
        block = self.root(tools=[REMEMBER]).fork("go")
        thread = self.turn(block)
        thread.join()
        finished = thread.last("tool_call_finished")
        self.assertEqual(finished["error"], "bad_arguments")
        self.assertIn("bad arguments for 'remember'", finished["result"])

    def test_malformed_arguments_do_not_run_the_tool(self):
        self.provider.script(
            Response.tool_calls([("set_magic_number", "{not json")]),
            Response.text("ok"),
        )
        block = self.root().fork("go")
        thread = self.turn(block)
        thread.join()
        finished = thread.last("tool_call_finished")
        self.assertFalse(finished["ok"])
        self.assertEqual(finished["error"], "bad_arguments")
        self.assertIn("are not valid JSON", finished["result"])
        # the error goes back to the model like any other tool result
        self.assertEqual(block.messages[2]["role"], "tool")
        self.assertIn("are not valid JSON", block.messages[2]["content"])
        self.assertEqual(block.state_deltas, {})

    def test_arguments_that_are_not_an_object(self):
        self.provider.script(
            Response.tool_calls([("remember", '["key", "value"]')]),
            Response.text("ok"),
        )
        block = self.root(tools=[REMEMBER]).fork("go")
        thread = self.turn(block)
        thread.join()
        finished = thread.last("tool_call_finished")
        self.assertEqual(finished["error"], "bad_arguments")
        self.assertEqual(
            finished["result"], "Error: arguments for 'remember' must be a JSON object"
        )

    def test_unknown_tool(self):
        self.provider.script(
            Response.tool_calls([("no_such_tool", {"a": 1})]),
            Response.text("ok"),
        )
        block = self.root().fork("go")
        thread = self.turn(block)
        thread.join()
        self.assertFalse(thread.last("tool_call_requested")["known"])
        finished = thread.last("tool_call_finished")
        self.assertEqual(finished["error"], "unknown_tool")
        self.assertEqual(finished["result"], "Error: unknown tool 'no_such_tool'")

    def test_tool_that_raises(self):
        self.provider.script(
            Response.tool_call("boom", {}),
            Response.text("ok"),
        )
        block = self.root(tools=[BOOM]).fork("go")
        thread = self.turn(block)
        thread.join()
        finished = thread.last("tool_call_finished")
        self.assertEqual(finished["error"], "tool_raised")
        self.assertEqual(
            finished["result"], "Error: tool 'boom' raised ValueError: tool exploded"
        )
        self.assertEqual(thread.last("turn_finished")["rounds"], 2)

    def test_max_tool_rounds(self):
        for _ in range(agent.MAX_TOOL_ROUNDS):
            self.provider.tool_call("get_current_time", {})
        self.provider.text("a summary of the loop")
        block = self.root().fork("loop forever")
        thread = self.turn(block)
        thread.join(30.0)
        self.assertIsInstance(thread.error, agent.HHAgentError)
        self.assertIn("more than 120 times in a row", str(thread.error))
        self.assertEqual(block.outcome, "failed")
        self.assertEqual(thread.last("turn_failed")["error_type"], "HHAgentError")

    def test_tool_schemas(self):
        root = self.root(
            tools=[REMEMBER, tools.ToolEntry(name="plain", description="d", params=[])]
        )
        self.assertEqual(
            root.tool_schemas(),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "remember",
                        "description": "remember test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string", "description": "a parameter"},
                                "value": {"type": "string", "description": "a parameter"},
                            },
                            "required": ["key", "value"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "plain",
                        "description": "d",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                },
            ],
        )

    def test_tool_schemas_empty_without_tools(self):
        self.assertEqual(self.root(tools=[]).tool_schemas(), [])


# --- tool state ----------------------------------------------------------


class StateTests(HHTestCase):
    def test_state_is_committed_on_success(self):
        self.provider.script(
            Response.tool_call("remember", {"key": "colour", "value": "blue"}),
            Response.text("done"),
        )
        block = self.root(tools=[REMEMBER]).fork("remember blue")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(
            block.state_deltas["mem"], agent.StateDelta(changed={"colour": "blue"})
        )
        self.assertEqual(block.merged_state("mem"), {"colour": "blue"})
        self.assertEqual(block.all_states(), {"mem": {"colour": "blue"}})
        self.assertEqual(block.state_namespaces(), ["mem"])

        delta_event = thread.last("state_delta")
        self.assertEqual(delta_event["tool"], "remember")
        self.assertEqual(delta_event["state_namespace"], "mem")
        self.assertEqual(delta_event["changed"], ["colour"])
        self.assertEqual(delta_event["removed"], [])
        self.assertEqual(delta_event["keys"], ["colour"])
        self.assertEqual(pick(thread.events, "state_discarded"), [])

    def test_a_child_inherits_and_reports_what_it_inherited(self):
        self.provider.script(
            Response.tool_call("remember", {"key": "colour", "value": "blue"}),
            Response.text("done"),
        )
        root = self.root(tools=[REMEMBER])
        first = root.fork("remember blue")
        self.run_turn(first)

        self.provider.script(
            Response.tool_call("remember", {"key": "shape", "value": "round"}),
            Response.text("ok"),
        )
        second = first.fork("what do you remember?")
        thread = self.turn(second)
        thread.join()
        self.assertIsNone(thread.error)
        self.assertEqual(second.merged_state("mem"), {"colour": "blue", "shape": "round"})
        loaded = thread.last("state_loaded")
        self.assertEqual(loaded["keys"], ["colour"])
        self.assertEqual(loaded["inherited"], 1)

    def test_untouched_state_leaves_no_delta(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}),
            Response.text("done"),
        )
        block = self.root().fork("what time is it?")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(block.state_deltas, {})
        self.assertEqual(pick(thread.events, "state_delta"), [])
        self.assertEqual(pick(thread.events, "state_discarded"), [])

    def test_a_value_equal_to_the_inherited_one_is_not_a_delta(self):
        self.provider.script(
            Response.tool_call("set_magic_number", {"magic": "7"}),
            Response.text("ok"),
        )
        root = self.root()
        first = root.fork("set 7")
        self.run_turn(first)

        self.provider.script(
            Response.tool_call("set_magic_number", {"magic": "7"}),
            Response.text("ok"),
        )
        second = first.fork("set 7 again")
        thread = self.turn(second)
        thread.join()
        self.assertEqual(second.state_deltas, {})
        self.assertEqual(pick(thread.events, "state_delta"), [])
        self.assertEqual(pick(thread.events, "state_discarded"), [])

    def test_removed_keys_are_recorded(self):
        self.provider.script(
            Response.tool_call("remember", {"key": "a", "value": "1"}),
            Response.text("ok"),
        )
        root = self.root(tools=[REMEMBER, FORGET])
        first = root.fork("remember a")
        self.run_turn(first)

        self.provider.script(
            Response.tool_call("forget", {"key": "a"}),
            Response.text("ok"),
        )
        second = first.fork("forget a")
        thread = self.turn(second)
        thread.join()
        self.assertEqual(second.state_deltas["mem"], agent.StateDelta(removed=("a",)))
        self.assertEqual(second.merged_state("mem"), {})
        delta_event = thread.last("state_delta")
        self.assertEqual(delta_event["changed"], [])
        self.assertEqual(delta_event["removed"], ["a"])
        self.assertEqual(delta_event["keys"], [])

    def test_nested_edits_are_captured_by_top_level_key(self):
        self.provider.script(
            Response.tool_call("append", {"item": "one"}),
            Response.tool_call("append", {"item": "two"}),
            Response.text("ok"),
        )
        block = self.root(tools=[APPEND]).fork("add two")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(block.merged_state("mem"), {"items": ["one", "two"]})
        self.assertEqual(thread.last("state_delta")["changed"], ["items"])

    def test_state_is_discarded_when_the_turn_fails(self):
        self.provider.script(
            Response.tool_call("remember", {"key": "colour", "value": "blue"}),
            Response.error(500, b"nope"),
            Response.text("it was remembering a colour"),
        )
        block = self.root(tools=[REMEMBER]).fork("remember blue")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(block.outcome, "failed")
        self.assertEqual(block.state_deltas, {})
        discarded = thread.last("state_discarded")
        self.assertEqual(discarded["tool"], "remember")
        self.assertEqual(discarded["state_namespace"], "mem")
        self.assertEqual(discarded["changed"], ["colour"])
        self.assertEqual(discarded["reason"], "failed")
        self.assertEqual(pick(thread.events, "state_delta"), [])

    def test_state_is_discarded_when_the_turn_is_cancelled(self):
        self.provider.script(
            Response.tool_call("remember", {"key": "k", "value": "v"}),
            Response.steady(pieces=8, gap=0.01),
        )
        block = self.root(tools=[REMEMBER]).fork("remember then stall")
        thread = self.turn(block)
        thread.wait_event("content_delta")
        block.cancel()
        thread.join()
        self.assertEqual(block.outcome, "cancelled")
        discarded = thread.last("state_discarded")
        self.assertEqual(discarded["changed"], ["k"])
        self.assertEqual(discarded["reason"], "cancelled")
        self.assertEqual(block.state_deltas, {})

    def test_state_is_discarded_when_the_turn_is_abandoned(self):
        self.provider.script(
            Response.tool_call("remember", {"key": "k", "value": "v"}),
            Response.steady(pieces=6, gap=0.02),
        )
        block = self.root(tools=[REMEMBER]).fork("remember then stall")
        events = []
        stream = block.stream(events.append)
        next(stream)  # the second round's first chunk
        stream.close()
        self.assertEqual(block.outcome, "abandoned")
        self.assertEqual(block.state_deltas, {})
        discarded = pick(events, "state_discarded")[0]
        self.assertEqual(discarded["reason"], "abandoned")
        self.assertEqual(discarded["changed"], ["k"])
        self.assertTrue(block.messages[-1]["content"].startswith("[harness]"))

    def test_forking_rewinds_state(self):
        self.provider.script(
            Response.tool_call("remember", {"key": "a", "value": "1"}),
            Response.text("ok"),
        )
        root = self.root(tools=[REMEMBER])
        branch_a = root.fork("remember a")
        self.run_turn(branch_a)

        self.provider.script(
            Response.tool_call("remember", {"key": "b", "value": "2"}),
            Response.text("ok"),
        )
        branch_b = branch_a.fork("remember b")
        self.run_turn(branch_b)
        self.assertEqual(branch_b.merged_state("mem"), {"a": "1", "b": "2"})

        self.provider.script(
            Response.tool_call("remember", {"key": "c", "value": "3"}),
            Response.text("ok"),
        )
        branch_c = branch_a.fork("remember c")
        self.run_turn(branch_c)
        # the sibling's write is invisible here
        self.assertEqual(branch_c.merged_state("mem"), {"a": "1", "c": "3"})

    def test_merged_state_is_a_view_and_tool_state_is_a_copy(self):
        root = self.root(tools=[REMEMBER])
        root.state_deltas["mem"] = agent.StateDelta(changed={"items": ["x"]})
        root.merged_state("mem")["items"].append("y")  # documented as read-only
        self.assertEqual(root.state_deltas["mem"].changed["items"], ["x", "y"])

        root.state_deltas["mem"] = agent.StateDelta(changed={"items": ["x"]})
        root.tool_state("mem")["items"].append("y")
        self.assertEqual(root.state_deltas["mem"].changed["items"], ["x"])

    def test_all_states_covers_every_touched_namespace(self):
        root = self.root()
        root.state_deltas["magic"] = agent.StateDelta(changed={"magic": "7"})
        root.state_deltas["mem"] = agent.StateDelta(changed={"a": 1}, removed=("b",))
        self.assertEqual(root.all_states(), {"magic": {"magic": "7"}, "mem": {"a": 1}})

    def test_tool_for_resolves_a_namespace_to_a_tool(self):
        root = self.root()
        self.assertEqual(root.tool_for("magic"), "set_magic_number")
        self.assertEqual(root.tool_for("nobody"), "nobody")

    def test_live_state_is_released_after_a_turn(self):
        self.provider.script(
            Response.tool_call("remember", {"key": "a", "value": "1"}),
            Response.text("ok"),
        )
        block = self.root(tools=[REMEMBER]).fork("hi")
        self.run_turn(block)
        # the turn really did load a namespace...
        self.assertEqual(block.merged_state("mem"), {"a": "1"})
        # ...and the live copy was dropped once it was committed
        self.assertEqual(block._live_states, {})
        self.assertEqual(block._state_bases, {})

    def test_the_magic_pair_shares_one_namespace(self):
        self.provider.script(
            Response.tool_call("set_magic_number", {"magic": "42"}),
            Response.tool_call("get_magic_number", {}),
            Response.text("the magic number is 42"),
        )
        block = self.root().fork("set it, then read it")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(block.merged_state("magic"), {"magic": "42"})
        self.assertEqual(block.messages[4]["content"], "42")


# --- local (client-run) tools -------------------------------------------


class LocalToolTests(HHTestCase):
    def test_a_local_call_is_announced_then_answered(self):
        self.provider.script(
            Response.tool_calls([("ask_operator", {"question": "lunch?"}, "c1")]),
            Response.text("thanks"),
        )
        root = self.root(tools=[local_tool("ask_operator", params=[param("question")])])
        block = root.fork("ask the operator")
        events = []
        thread = self.turn(block, on_event=events.append)
        called = thread.wait_event("local_tool_called")
        self.assertEqual(called["round"], 1)
        self.assertEqual(called["call_id"], "c1")
        self.assertEqual(called["name"], "ask_operator")
        self.assertEqual(called["kind"], "call")
        self.assertEqual(called["arguments"], {"question": "lunch?"})
        self.assertEqual(called["raw_arguments"], '{"question": "lunch?"}')
        self.assertEqual(called["timeout_ms"], agent.DEFAULT_LOCAL_TIMEOUT * 1000)
        self.assertEqual(block.pending_calls(), ["c1"])
        self.assertEqual(block.messages[-1]["role"], "assistant")

        self.assertTrue(block.resolve_local_call("c1", "red braised pork"))
        thread.join()
        self.assertIsNone(thread.error)
        self.assertEqual(thread.text, "thanks")
        self.assertEqual(
            block.messages[2],
            {"role": "tool", "tool_call_id": "c1", "content": "red braised pork"},
        )
        resolved = thread.last("local_tool_resolved")
        self.assertTrue(resolved["ok"])
        self.assertIsNone(resolved["error"])
        self.assertEqual(resolved["result"], "red braised pork")
        self.assertEqual(resolved["result_chars"], len("red braised pork"))
        self.assertEqual(resolved["kind"], "call")
        self.assertEqual(block.pending_calls(), [])
        finished = thread.last("tool_call_finished")
        self.assertTrue(finished["ok"])
        self.assertIsNone(finished["error"])
        self.assertEqual(finished["result"], "red braised pork")

    def test_resolving_an_unknown_call_is_refused(self):
        self.assertFalse(self.root(tools=[]).fork("hi").resolve_local_call("nope", "x"))

    def test_a_local_call_can_be_answered_with_an_error(self):
        self.provider.script(
            Response.tool_calls([("ask_operator", {"question": "lunch?"}, "c1")]),
            Response.text("ok then"),
        )
        block = self.root(tools=[local_tool("ask_operator")]).fork("ask")
        thread = self.turn(block)
        thread.wait_event("local_tool_called")
        block.resolve_local_call("c1", "the operator is away", error="operator_away")
        thread.join()
        resolved = thread.last("local_tool_resolved")
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["error"], "operator_away")
        self.assertEqual(resolved["result"], "the operator is away")
        finished = thread.last("tool_call_finished")
        self.assertFalse(finished["ok"])
        self.assertEqual(finished["error"], "operator_away")
        self.assertEqual(block.messages[2]["content"], "the operator is away")

    def test_an_error_answer_without_a_result_still_produces_a_message(self):
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.text("ok"),
        )
        block = self.root(tools=[local_tool("ask_operator")]).fork("ask")
        thread = self.turn(block)
        thread.wait_event("local_tool_called")
        block.resolve_local_call("c1", "", error="boom")
        thread.join()
        self.assertEqual(
            block.messages[2]["content"], "Error: local tool 'ask_operator': boom"
        )

    def test_a_local_call_that_is_never_answered_times_out(self):
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.text("the operator never answered"),
        )
        block = self.root(
            tools=[local_tool("ask_operator")], local_timeout=0.1
        ).fork("ask")
        thread = self.turn(block)
        unresolved = thread.wait_event("local_tool_unresolved")
        self.assertEqual(unresolved["reason"], "timeout")
        self.assertEqual(unresolved["kind"], "call")
        thread.join()
        self.assertIsNone(thread.error)
        self.assertEqual(thread.text, "the operator never answered")
        finished = thread.last("tool_call_finished")
        self.assertEqual(finished["error"], "timeout")
        self.assertIn("was not answered (timeout)", finished["result"])
        # a timeout is not a failure of the turn, and commits normally
        self.assertEqual(block.outcome, "ok")
        self.assertEqual(pick(thread.events, "state_discarded"), [])

    def test_cancelling_releases_a_parked_call(self):
        self.provider.script(Response.tool_calls([("ask_operator", {}, "c1")]))
        block = self.root(tools=[local_tool("ask_operator")]).fork("ask")
        thread = self.turn(block)
        thread.wait_event("local_tool_called")
        self.assertTrue(block.cancel())
        thread.join()
        self.assertIsInstance(thread.error, agent.HHAgentCancelled)
        self.assertEqual(thread.last("local_tool_unresolved")["reason"], "cancelled")
        self.assertEqual(block.outcome, "cancelled")
        self.assertEqual(thread.last("turn_cancelled")["error"], str(thread.error))

    def test_local_calls_are_asked_one_at_a_time(self):
        self.provider.script(
            Response.tool_calls(
                [
                    ("ask_operator", {"question": "first"}, "c1"),
                    ("ask_operator", {"question": "second"}, "c2"),
                ]
            ),
            Response.text("both answered"),
        )
        block = self.root(tools=[local_tool("ask_operator")]).fork("ask twice")
        thread = self.turn(block)
        self.assertEqual(
            thread.wait_event("local_tool_called", call_id="c1")["arguments"],
            {"question": "first"},
        )
        self.assertEqual(block.pending_calls(), ["c1"])
        block.resolve_local_call("c1", "one")
        self.assertEqual(
            thread.wait_event("local_tool_called", call_id="c2")["arguments"],
            {"question": "second"},
        )
        self.assertEqual(block.pending_calls(), ["c2"])
        block.resolve_local_call("c2", "two")
        thread.join()
        self.assertEqual(
            [message["content"] for message in block.messages if message["role"] == "tool"],
            ["one", "two"],
        )

    def test_local_tools_have_no_server_side_state(self):
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.text("ok"),
        )
        block = self.root(tools=[local_tool("ask_operator")]).fork("ask")
        thread = self.turn(block)
        thread.wait_event("local_tool_called")
        block.resolve_local_call("c1", "answer")
        thread.join()
        self.assertEqual(block.state_deltas, {})
        self.assertEqual(pick(thread.events, "state_loaded"), [])
        self.assertEqual(pick(thread.events, "state_delta"), [])

    def test_local_tool_definitions_survive_a_fork(self):
        root = self.root(tools=[local_tool("ask_operator", rollback=True)])
        child = root.fork("hi")
        self.assertTrue(child.tools["ask_operator"].is_local)
        self.assertTrue(child.tools["ask_operator"].has_rollback)
        self.assertEqual(child.tool_schemas()[0]["function"]["name"], "ask_operator")


# --- rollback ------------------------------------------------------------


class RollbackTests(HHTestCase):
    def _install(self, undo="ok", effects=True):
        """A tool whose hook and rollback leave a visible trace."""
        trace = []

        def hook(context, package):
            trace.append(("run", package))
            context.state["installed"] = package
            return f"installed {package}"

        def rollback(context, package):
            trace.append(("undo", package, context.result))
            if undo == "raise":
                raise RuntimeError("undo is broken")
            if undo == "copy":
                context.state["oops"] = True
            return f"removed {package}"

        tool = server_tool(
            "install",
            hook,
            params=[param("package")],
            rollback=rollback if undo is not None else None,
            external_effects=effects,
        )
        return tool, trace

    def test_rollbacks_run_newest_first_with_the_call_context(self):
        tool, trace = self._install()
        self.provider.script(
            Response.tool_calls(
                [
                    ("install", {"package": "one"}, "c1"),
                    ("install", {"package": "two"}, "c2"),
                    ("install", {"package": "three"}, "c3"),
                ]
            ),
            Response.error(500, b"the turn dies here"),
            Response.text("it was installing packages"),
        )
        block = self.root(tools=[tool]).fork("install three things")
        thread = self.turn(block)
        thread.join()

        self.assertEqual(
            trace,
            [
                ("run", "one"),
                ("run", "two"),
                ("run", "three"),
                ("undo", "three", "installed three"),
                ("undo", "two", "installed two"),
                ("undo", "one", "installed one"),
            ],
        )
        started = pick(thread.events, "rollback_started")
        self.assertEqual([event["tool"] for event in started], ["install"] * 3)
        self.assertEqual([event["index"] for event in started], [1, 2, 3])
        self.assertEqual([event["total"] for event in started], [3, 3, 3])
        self.assertEqual([event["call_id"] for event in started], ["c3", "c2", "c1"])
        finished = pick(thread.events, "rollback_finished")
        self.assertEqual(
            [event["result"] for event in finished],
            ["removed three", "removed two", "removed one"],
        )
        self.assertTrue(all(event["ok"] for event in finished))
        self.assertIn("Undone: install", block.messages[-1]["content"])

    def test_a_failing_rollback_is_contained_and_the_rest_still_run(self):
        good, good_trace = self._install()
        bad, _ = self._install(undo="raise")
        bad.name = "install_bad"
        self.provider.script(
            Response.tool_calls(
                [
                    ("install", {"package": "good"}, "c1"),
                    ("install_bad", {"package": "bad"}, "c2"),
                ]
            ),
            Response.error(500),
            Response.text("summary of a bad turn"),
        )
        block = self.root(tools=[good, bad]).fork("install two things")
        thread = self.turn(block)
        thread.join()

        by_call = {event["call_id"]: event for event in pick(thread.events, "rollback_finished")}
        self.assertTrue(by_call["c1"]["ok"])
        self.assertFalse(by_call["c2"]["ok"])
        self.assertEqual(by_call["c2"]["error"], "rollback_raised")
        self.assertIn("RuntimeError", by_call["c2"]["result"])
        # the good rollback still ran, even though it is older
        self.assertEqual(good_trace[-1], ("undo", "good", "installed good"))
        note = block.messages[-1]["content"]
        self.assertIn("Could not be undone: install_bad.", note)
        self.assertIn("Undone: install.", note)
        self.assertEqual(block.outcome, "failed")  # the turn keeps its own failure
        self.assertIn("500", block.error)

    def test_a_rollback_with_bad_arguments_is_reported(self):
        def hook(context, package):
            return f"installed {package}"

        def rollback(context, other_name):
            return "never runs"

        tool = server_tool("install", hook, params=[param("package")], rollback=rollback)
        self.provider.script(
            Response.tool_call("install", {"package": "x"}),
            Response.error(500),
            Response.text("summary"),
        )
        block = self.root(tools=[tool]).fork("install")
        thread = self.turn(block)
        thread.join()
        finished = thread.last("rollback_finished")
        self.assertFalse(finished["ok"])
        self.assertEqual(finished["error"], "bad_arguments")
        self.assertIn("bad arguments", finished["result"])

    def test_the_rollback_state_is_a_copy(self):
        tool, _ = self._install(undo="copy")
        self.provider.script(
            Response.tool_call("install", {"package": "x"}),
            Response.error(500),
            Response.text("summary"),
        )
        block = self.root(tools=[tool]).fork("install")
        thread = self.turn(block)
        thread.join()
        discarded = thread.last("state_discarded")
        self.assertEqual(discarded["changed"], ["installed"])
        self.assertEqual(discarded["removed"], [])
        self.assertEqual(block.state_deltas, {})

    def test_external_effects_without_an_undo_are_named(self):
        tool, _ = self._install(undo=None, effects=True)
        self.provider.script(
            Response.tool_call("install", {"package": "x"}),
            Response.error(500),
            Response.text("summary"),
        )
        block = self.root(tools=[tool]).fork("install")
        thread = self.turn(block)
        thread.join()
        unavailable = thread.last("rollback_unavailable")
        self.assertEqual(unavailable["tool"], "install")
        self.assertEqual(unavailable["call_id"], "call_1")
        self.assertEqual(unavailable["index"], 1)
        self.assertEqual(unavailable["total"], 1)
        self.assertEqual(pick(thread.events, "rollback_started"), [])
        self.assertIn("May still be in effect: install.", block.messages[-1]["content"])

    def test_a_tool_that_declares_nothing_is_not_mentioned(self):
        self.provider.script(
            Response.tool_call("remember", {"key": "a", "value": "1"}),
            Response.error(500),
            Response.text("summary"),
        )
        block = self.root(tools=[REMEMBER]).fork("remember")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(pick(thread.events, "rollback_started"), [])
        self.assertEqual(pick(thread.events, "rollback_unavailable"), [])
        note = block.messages[-1]["content"]
        for marker in ("Undone:", "Could not be undone:", "May still be in effect:"):
            self.assertNotIn(marker, note)

    def test_calls_that_never_ran_are_not_rolled_back(self):
        tool, trace = self._install()
        self.provider.script(
            Response.tool_calls(
                [
                    ("install", "{not json", "c1"),
                    ("no_such_tool", {}, "c2"),
                    ("install", '["not", "an", "object"]', "c3"),
                ]
            ),
            Response.error(500),
            Response.text("summary"),
        )
        block = self.root(tools=[tool]).fork("break things")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(trace, [])
        self.assertEqual(pick(thread.events, "rollback_started"), [])
        self.assertEqual(pick(thread.events, "rollback_unavailable"), [])
        self.assertEqual(block._called, [])

    def test_a_successful_turn_rolls_back_nothing(self):
        tool, trace = self._install()
        self.provider.script(
            Response.tool_call("install", {"package": "x"}),
            Response.text("done"),
        )
        block = self.root(tools=[tool]).fork("install")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(trace, [("run", "x")])
        self.assertEqual(pick(thread.events, "rollback_started"), [])
        self.assertEqual(block.merged_state("install"), {"installed": "x"})

    def test_a_local_undo_is_asked_for_but_not_waited_on_after_a_cancel(self):
        self.provider.script(
            Response.tool_calls([("ask_operator", {"q": "hi"}, "c1")]),
            Response.steady(pieces=8, gap=0.01),
        )
        block = self.root(tools=[local_tool("ask_operator", rollback=True)]).fork(
            "ask, then stall"
        )
        thread = self.turn(block)
        thread.wait_event("local_tool_called")
        block.resolve_local_call("c1", "the answer")
        thread.wait_event("content_delta")
        started = time.monotonic()
        block.cancel()
        thread.join(5.0)
        self.assertLess(time.monotonic() - started, 3.0)

        undo = thread.last("local_tool_rollback")
        self.assertEqual(undo["call_id"], "c1:rollback")
        self.assertEqual(undo["rollback_of"], "c1")
        self.assertEqual(undo["name"], "ask_operator")
        self.assertEqual(undo["kind"], "rollback")
        self.assertEqual(undo["arguments"], {"q": "hi"})
        self.assertEqual(undo["result"], "the answer")
        self.assertTrue(undo["call_ok"])
        self.assertEqual(undo["timeout_ms"], 0.0)
        unresolved = thread.last("local_tool_unresolved")
        self.assertEqual(unresolved["kind"], "rollback")
        self.assertEqual(unresolved["reason"], "cancelled")

    def test_a_local_undo_follows_an_abandoned_turn_too(self):
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.steady(pieces=6, gap=0.02),
        )
        block = self.root(tools=[local_tool("ask_operator", rollback=True)]).fork(
            "ask, then stall"
        )
        events = []
        # the local call has to be answered while the turn is parked, so the
        # answering happens on its own thread
        answerer = threading.Thread(
            target=lambda: (
                self.wait_until(lambda: block.pending_calls()),
                block.resolve_local_call("c1", "answer"),
            ),
            daemon=True,
        )
        answerer.start()
        stream = block.stream(events.append)
        next(stream)
        answerer.join(5.0)
        stream.close()
        self.assertEqual(block.outcome, "abandoned")
        undos = pick(events, "local_tool_rollback")
        self.assertEqual(len(undos), 1)
        self.assertEqual(undos[0]["rollback_of"], "c1")
        self.assertEqual(undos[0]["timeout_ms"], 0.0)

    def test_a_cancel_during_a_rollback_wait_keeps_the_original_failure(self):
        # the undo of a failed turn does wait for the client, but a cancel must
        # not be replaced by a rollback error: the undo is abandoned instead
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.error(500),
            Response.text("summary"),
        )
        block = self.root(tools=[local_tool("ask_operator", rollback=True)]).fork("ask")
        thread = self.turn(block)
        thread.wait_event("local_tool_called")
        block.resolve_local_call("c1", "answer")
        thread.wait_event("local_tool_rollback")
        block.cancel()
        thread.join(5.0)

        self.assertEqual(block.outcome, "failed")  # the failure stands
        self.assertIn("500", block.error)
        finished = thread.last("rollback_finished")
        self.assertFalse(finished["ok"])
        self.assertEqual(finished["error"], "cancelled")
        self.assertEqual(thread.last("local_tool_unresolved")["reason"], "cancelled")
        self.assertIn("Could not be undone: ask_operator.", block.messages[-1]["content"])

    def test_a_server_tool_that_only_promises_a_remote_rollback(self):
        # `remote_rollback` on a server-run tool is a promise with nothing to
        # call: the undo is reported as done without running anything
        def hook(context):
            return "did something"

        tool = tools.ToolEntry(
            "install",
            "d",
            [],
            hook=hook,
            remote_rollback=True,
            external_effects=True,
        )
        self.provider.script(
            Response.tool_call("install", {}),
            Response.error(500),
            Response.text("summary"),
        )
        block = self.root(tools=[tool]).fork("install")
        thread = self.turn(block)
        thread.join()
        finished = thread.last("rollback_finished")
        self.assertTrue(finished["ok"])
        self.assertEqual(finished["result"], "")
        self.assertIn("Undone: install.", block.messages[-1]["content"])

    def test_a_local_call_that_was_never_answered_is_still_offered_an_undo(self):
        # the client may have run the tool before falling silent, so the undo is
        # offered even though no result ever arrived
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.error(500),
            Response.text("summary"),
        )
        block = self.root(
            tools=[local_tool("ask_operator", rollback=True)], local_timeout=0.1
        ).fork("ask")
        thread = self.turn(block)
        thread.wait_event("local_tool_unresolved", kind="call")
        undo = thread.wait_event("local_tool_rollback")
        self.assertEqual(undo["rollback_of"], "c1")
        self.assertIn("was not answered", undo["result"])  # the error text
        self.assertFalse(undo["call_ok"])
        # the undo is not answered either, and gives up on its own timeout
        thread.wait_event("local_tool_unresolved", kind="rollback")
        thread.join(5.0)
        self.assertEqual(thread.last("local_tool_unresolved")["reason"], "timeout")
        finished = thread.last("rollback_finished")
        self.assertFalse(finished["ok"])
        self.assertEqual(finished["error"], "timeout")
        self.assertEqual(block.outcome, "failed")
        self.assertIn("Could not be undone: ask_operator.", block.messages[-1]["content"])

    def test_a_local_tool_without_a_promised_undo_is_not_asked(self):
        self.provider.script(
            Response.tool_calls([("ask_operator", {}, "c1")]),
            Response.error(500),
            Response.text("summary"),
        )
        block = self.root(tools=[local_tool("ask_operator", rollback=False)]).fork("ask")
        thread = self.turn(block)
        thread.wait_event("local_tool_called")
        block.resolve_local_call("c1", "answer")
        thread.join()
        self.assertEqual(pick(thread.events, "local_tool_rollback"), [])
        self.assertEqual(pick(thread.events, "rollback_unavailable"), [])

    def test_the_called_list_is_cleared_after_a_turn(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}),
            Response.text("ok"),
        )
        block = self.root().fork("time?")
        self.run_turn(block)
        self.assertEqual(block._called, [])


# --- failure notes and summaries ----------------------------------------


class FailureNoteTests(HHTestCase):
    def test_a_failed_turn_is_summarised_and_noted(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}, "c1"),
            Response.error(503, b"no capacity"),
            Response.text("the model was checking the clock when the provider refused"),
        )
        block = self.root().fork("what time is it?")
        events = []
        thread = self.turn(block, on_event=events.append)
        thread.join()

        started = pick(events, "failure_summary_started")[0]
        self.assertEqual(started["model"], "test-model")
        self.assertEqual(started["outcome"], "failed")
        self.assertIn("503", started["error"])
        self.assertGreater(started["request_bytes"], 0)
        self.assertEqual(started["timeout"], agent.DEFAULT_TIMEOUT)
        self.assertEqual(started["messages"], 3)  # before the note was appended

        finished = pick(events, "failure_summary_finished")[0]
        self.assertEqual(finished["status"], 200)
        self.assertEqual(
            finished["summary"],
            "the model was checking the clock when the provider refused",
        )
        self.assertEqual(finished["chars"], len(finished["summary"]))
        self.assertEqual(finished["finish_reason"], "stop")
        self.assertGreaterEqual(finished["chunks"], 1)
        self.assertIsNone(finished["usage"])

        note = block.messages[-1]
        self.assertEqual(note["role"], "user")
        self.assertTrue(
            note["content"].startswith(
                "[harness] the previous turn failed; the state it changed was discarded."
            )
        )
        self.assertIn("Reported error: 503", note["content"])
        self.assertIn(
            "In short: the model was checking the clock when the provider refused",
            note["content"],
        )

        # the note is announced as an ordinary message, marked as the failure's
        appended = pick(events, "history_appended")[-1]
        self.assertEqual(appended["source"], "failure")
        self.assertEqual(appended["role"], "user")
        self.assertEqual(appended["chars"], len(note["content"]))
        self.assertEqual(appended["messages"], len(block.messages))
        self.assertEqual(appended["preview"], note["content"][:200])

        # the summary is a plain request with no tools, carrying the transcript
        payload = self.provider.last_payload()
        self.assertEqual(payload["model"], "test-model")
        self.assertTrue(payload["stream"])
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], SUMMARY_PROMPT)
        prompt = payload["messages"][1]["content"]
        self.assertIn("The turn that failed, as it was recorded:", prompt)
        self.assertIn("get_current_time", prompt)
        self.assertIn("The reported error: 503", prompt)

    def test_the_summary_model_can_differ(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}),
            Response.error(500),
            Response.text("a summary"),
        )
        block = self.root(summary_model="cheap-model").fork("time?")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(thread.last("failure_summary_started")["model"], "cheap-model")
        self.assertEqual(self.provider.last_payload()["model"], "cheap-model")

    def test_an_empty_summary_falls_back_to_the_raw_error(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}),
            Response.error(500),
            Response.text(""),
        )
        block = self.root().fork("time?")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(thread.last("failure_summary_finished")["summary"], "")
        note = block.messages[-1]["content"]
        self.assertIn("No summary was available.", note)
        self.assertNotIn("In short:", note)
        self.assertIn("Reported error: 500", note)

    def test_the_summary_stream_tolerates_junk_and_reports_usage(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}),
            Response.error(500),
            Response(
                chunks=[
                    "this line is not json",
                    delta(content="the short "),
                    usage(7, 3),
                    delta(content="story"),
                    finish(),
                    "[DONE]",
                ]
            ),
        )
        block = self.root().fork("time?")
        thread = self.turn(block)
        thread.join()
        finished = thread.last("failure_summary_finished")
        self.assertEqual(finished["summary"], "the short story")
        self.assertEqual(
            finished["usage"],
            {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )
        self.assertIn("In short: the short story", block.messages[-1]["content"])

    def test_a_refused_summary_request_is_reported(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}),
            Response.error(500),
            Response.error(500, b"summariser is down"),
        )
        block = self.root().fork("time?")
        thread = self.turn(block)
        thread.join()
        failed = thread.last("failure_summary_failed")
        self.assertEqual(failed["status"], 500)
        self.assertEqual(failed["error_type"], "HTTPError")
        self.assertIn("summariser is down", failed["error"])
        self.assertIn("No summary was available.", block.messages[-1]["content"])

    def test_a_summary_that_breaks_mid_stream_is_reported(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}),
            Response.error(500),
            Response.text("z" * 4000, chunk_size=500, truncate=True),
        )
        block = self.root().fork("time?")
        thread = self.turn(block)
        thread.join()
        self.assertGreater(thread.last("failure_summary_failed")["chunks"], 0)
        self.assertIn("No summary was available.", block.messages[-1]["content"])

    def test_a_summary_request_that_cannot_be_sent_is_reported(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}),
            Response.error(500),
        )
        real_post = requests.post

        def explode(url, **kwargs):
            if b"You are the harness" in kwargs.get("data", b""):
                raise requests.ConnectionError("no route to the summariser")
            return real_post(url, **kwargs)

        block = self.root().fork("time?")
        events = []
        with mock.patch.object(agent.requests, "post", side_effect=explode):
            thread = self.turn(block, on_event=events.append)
            thread.join()
        failed = pick(events, "failure_summary_failed")[0]
        self.assertEqual(failed["error_type"], "ConnectionError")
        self.assertIn("no route to the summariser", failed["error"])
        self.assertIn("No summary was available.", block.messages[-1]["content"])

    def test_a_turn_with_nothing_to_summarise_makes_no_request(self):
        block = self.root(tools=[], endpoint="http://127.0.0.1:1").fork("hi")
        thread = self.turn(block)
        thread.join()
        self.assertIsInstance(thread.error, agent.HHAgentError)
        self.assertEqual(pick(thread.events, "failure_summary_started"), [])
        self.assertEqual(pick(thread.events, "failure_summary_failed"), [])
        note = block.messages[-1]["content"]
        self.assertIn("[harness] the previous turn failed", note)
        self.assertIn("No summary was available.", note)

    def test_a_failure_that_only_produced_text_is_still_summarised(self):
        # `_worth_summarizing` is true when the turn had said something, even
        # though no tool ever ran
        self.provider.script(
            Response.text("I was in the middle of an answer " * 40, truncate=True),
            Response.text("it was answering when the stream broke"),
        )
        block = self.root(tools=[]).fork("hi")
        thread = self.turn(block)
        thread.join()
        self.assertEqual(block._called, [])  # no tool was called
        self.assertTrue(block.text)  # but the partial answer is on the block
        self.assertEqual(
            thread.last("failure_summary_finished")["summary"],
            "it was answering when the stream broke",
        )
        self.assertIn(
            "In short: it was answering when the stream broke",
            block.messages[-1]["content"],
        )

    def test_a_cancelled_turn_uses_fixed_text_and_no_request(self):
        self.provider.script(
            Response.tool_call("get_current_time", {}),
            Response.steady(pieces=8, gap=0.01),
        )
        block = self.root().fork("time, then stall")
        thread = self.turn(block)
        thread.wait_event("content_delta")
        block.cancel()
        thread.join()
        self.assertEqual(pick(thread.events, "failure_summary_started"), [])
        note = block.messages[-1]["content"]
        self.assertTrue(
            note.startswith(
                "[harness] the previous turn was cancelled by the caller; "
                "the state it changed was discarded."
            )
        )
        self.assertIn("Reported error:", note)
        self.assertNotIn("In short:", note)
        self.assertNotIn("No summary was available.", note)

    def test_an_abandoned_turn_uses_fixed_text(self):
        self.provider.script(Response.steady(pieces=6, gap=0.02))
        block = self.root().fork("hi")
        stream = block.stream()
        next(stream)
        stream.close()
        self.assertTrue(
            block.messages[-1]["content"].startswith(
                "[harness] the previous turn was abandoned before it finished; "
                "the state it changed was discarded."
            )
        )

    def test_a_fork_carries_the_note_into_its_context(self):
        self.provider.script(Response.error(500))
        failed = self.root(tools=[]).fork("first")
        self.turn(failed).join()
        context = failed.fork("try again").context()
        self.assertEqual(context[-1], {"role": "user", "content": "try again"})
        self.assertTrue(
            context[-2]["content"].startswith("[harness] the previous turn failed")
        )
        self.assertEqual(len(context), 3)

    def test_the_note_reports_each_kind_of_effect(self):
        undone = agent._Undo("done_tool", "undone")
        stuck = agent._Undo("stuck_tool", "stuck", "rollback_raised")
        standing = agent._Undo("standing_tool", "standing")
        root = self.root(tools=[])
        root.error = "HHAgentError: it broke"
        note = root._failure_note("failed", "it did a thing", [undone, stuck, standing])
        self.assertIn("Undone: done_tool.", note)
        self.assertIn("Could not be undone: stuck_tool.", note)
        self.assertIn("May still be in effect: standing_tool.", note)
        self.assertIn("Reported error: HHAgentError: it broke", note)
        self.assertIn("In short: it did a thing", note)

    def test_the_note_without_an_error_or_summary(self):
        note = self.root(tools=[])._failure_note("failed", None, [])
        self.assertIn("[harness] the previous turn failed", note)
        self.assertIn("No summary was available.", note)
        self.assertNotIn("Reported error:", note)

    def test_the_summary_prompt_describes_calls_and_truncates(self):
        root = self.root(tools=[])
        root.messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "remember", "arguments": '{"key": "a"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 9000},
        ]
        root.text = "partial answer"
        root.outcome = "failed"
        root.error = "boom"
        prompt = root._summary_prompt([agent._Undo("remember", "undone")])
        self.assertIn('assistant: [tool calls: remember({"key": "a"})]', prompt)
        self.assertIn("tool: " + "x" * 4000, prompt)
        self.assertNotIn("x" * 4001, prompt)
        self.assertIn("assistant, cut off mid-answer: partial answer", prompt)
        self.assertIn("The turn ended as: failed", prompt)
        self.assertIn("The reported error: boom", prompt)
        self.assertIn("Effects that were undone: remember", prompt)


if __name__ == "__main__":
    unittest.main()
