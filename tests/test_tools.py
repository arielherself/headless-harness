"""Tests for `src/tools.py`: the tool schema, its state rules and the builtins."""

import os
import re
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import requests

from tests.support import tools


class ToolEntryTests(unittest.TestCase):
    def test_a_tool_with_a_hook_runs_on_the_server(self):
        tool = tools.ToolEntry("t", "d", [], hook=lambda context: "x")
        self.assertFalse(tool.is_local)
        self.assertFalse(tool.has_rollback)

    def test_a_tool_without_a_hook_runs_on_the_client(self):
        self.assertTrue(tools.ToolEntry("t", "d", []).is_local)

    def test_rollback_can_be_a_hook_or_a_promise(self):
        self.assertTrue(
            tools.ToolEntry("t", "d", [], hook=lambda c: "", rollback=lambda c: "").has_rollback
        )
        self.assertTrue(tools.ToolEntry("t", "d", [], remote_rollback=True).has_rollback)
        self.assertFalse(tools.ToolEntry("t", "d", [], hook=lambda c: "").has_rollback)

    def test_namespace_defaults_to_the_name(self):
        self.assertEqual(tools.ToolEntry("get_magic", "d", []).namespace, "get_magic")
        self.assertEqual(
            tools.ToolEntry("get_magic", "d", [], state_namespace="magic").namespace,
            "magic",
        )

    def test_external_effects_is_a_declaration_of_its_own(self):
        tool = tools.ToolEntry("t", "d", [], hook=lambda c: "", external_effects=True)
        self.assertTrue(tool.external_effects)
        self.assertFalse(tool.has_rollback)  # declaring effects is not an undo

    def test_a_param_carries_name_type_and_description(self):
        param = tools.ToolParam("a", "integer", "the first addend")
        self.assertEqual((param.name, param.type, param.description), ("a", "integer", "the first addend"))


class ToolContextTests(unittest.TestCase):
    def test_result_defaults_to_none(self):
        context = tools.ToolContext(
            tool=tools.ToolEntry("t", "d", []),
            agent=None,
            call_id="c1",
            arguments={"a": 1},
            raw_arguments='{"a": 1}',
            state={},
        )
        self.assertIsNone(context.result)
        self.assertEqual(context.arguments, {"a": 1})
        self.assertEqual(context.raw_arguments, '{"a": 1}')
        self.assertEqual(context.call_id, "c1")
        self.assertEqual(context.state, {})


class BuiltinToolTests(unittest.TestCase):
    def test_the_catalogue_is_the_five_builtins(self):
        self.assertEqual(
            [tool.name for tool in tools.builtin_tools],
            [
                "get_system_info",
                "get_current_time",
                "set_magic_number",
                "get_magic_number",
                "web_fetch",
            ],
        )

    def test_the_system_info_tool_reports_the_model_in_use(self):
        state = {}
        context = tools.ToolContext(
            tool=tools.get_system_info_tool,
            agent=SimpleNamespace(model="deepseek/deepseek-v4.1-flash"),
            call_id="c",
            arguments={},
            raw_arguments="{}",
            state=state,
        )
        text = tools.get_system_info_executor(context)
        self.assertIn("Model: deepseek/deepseek-v4.1-flash", text)
        self.assertIn(f"protocol version {tools.PROTOCOL_VERSION}", text)
        self.assertEqual(state, {})

    def test_the_system_info_tool_reports_an_unknown_model_without_a_block(self):
        context = tools.ToolContext(
            tool=tools.get_system_info_tool,
            agent=None,
            call_id="c",
            arguments={},
            raw_arguments="{}",
            state={},
        )
        self.assertIn("Model: <unknown>", tools.get_system_info_executor(context))

    def test_the_time_tool_reports_utc_and_local(self):
        def context(state=None):
            return tools.ToolContext(
                tool=tools.get_current_time_tool,
                agent=None,
                call_id="c",
                arguments={},
                raw_arguments="{}",
                state=state if state is not None else {},
            )

        before = datetime.now(timezone.utc)
        text = tools.get_current_time_executor(context())
        after = datetime.now(timezone.utc)
        self.assertTrue(text.startswith("UTC: "))
        self.assertIn("\nLocal: ", text)

        reported = datetime.strptime(
            re.match(r"UTC: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC", text).group(1),
            "%Y-%m-%d %H:%M:%S",
        ).replace(tzinfo=timezone.utc)
        self.assertLessEqual(before - timedelta(seconds=5), reported)
        self.assertLessEqual(reported, after + timedelta(seconds=5))

    def test_the_time_tool_touches_no_state(self):
        state = {}
        context = tools.ToolContext(
            tool=tools.get_current_time_tool,
            agent=None,
            call_id="c",
            arguments={},
            raw_arguments="{}",
            state=state,
        )
        tools.get_current_time_executor(context)
        self.assertEqual(state, {})

    def test_the_magic_pair_shares_one_namespace(self):
        self.assertEqual(tools.set_magic_number_tool.namespace, "magic")
        self.assertEqual(tools.get_magic_number_tool.namespace, "magic")
        self.assertNotEqual(
            tools.get_current_time_tool.namespace,
            tools.get_magic_number_tool.namespace,
        )

    def test_set_then_get_magic(self):
        state = {}
        setter = tools.ToolContext(
            tool=tools.set_magic_number_tool,
            agent=None,
            call_id="c1",
            arguments={"magic": "7"},
            raw_arguments='{"magic": "7"}',
            state=state,
        )
        # a hook is always called as `hook(context, **arguments)`
        self.assertEqual(
            tools.set_magic_number_executor(setter, **setter.arguments),
            "Magic is set to 7",
        )
        self.assertEqual(state, {"magic": "7"})

        getter = tools.ToolContext(
            tool=tools.get_magic_number_tool,
            agent=None,
            call_id="c2",
            arguments={},
            raw_arguments="{}",
            state=state,
        )
        self.assertEqual(tools.get_magic_number_executor(getter), "7")

    def test_get_magic_before_it_is_set(self):
        context = tools.ToolContext(
            tool=tools.get_magic_number_tool,
            agent=None,
            call_id="c",
            arguments={},
            raw_arguments="{}",
            state={},
        )
        self.assertEqual(tools.get_magic_number_executor(context), "<null>")

    def test_the_magic_tools_declare_their_parameter(self):
        self.assertEqual(
            tools.set_magic_number_tool.params,
            [tools.ToolParam("magic", "string", "the new magic number")],
        )
        self.assertEqual(tools.get_magic_number_tool.params, [])

    def test_the_builtins_promise_nothing_they_do_not_have(self):
        for tool in tools.builtin_tools:
            self.assertFalse(tool.external_effects)
            self.assertFalse(tool.has_rollback)
            self.assertFalse(tool.is_local)
            self.assertTrue(tool.description)


class WebFetchToolTests(unittest.TestCase):
    """`web_fetch` reads through Jina's reader, so the HTTP call is stubbed here."""

    def call(self, **arguments):
        """Call the hook the way the harness does: `hook(context, **arguments)`."""
        context = tools.ToolContext(
            tool=tools.web_fetch_tool,
            agent=None,
            call_id="c",
            arguments=arguments,
            raw_arguments="{}",
            state={},
        )
        return tools.web_fetch_executor(context, **context.arguments)

    def test_it_says_the_answer_comes_back_as_markdown(self):
        description = tools.web_fetch_tool.description
        self.assertIn("Markdown", description)
        self.assertIn("Jina", description)

    def test_it_reads_through_the_jina_reader(self):
        response = mock.Mock(ok=True, status_code=200, content=b"# Title\n\nbody")
        with mock.patch("requests.get", return_value=response) as get:
            text = self.call(url="example.com/docs")
        self.assertEqual(text, "# Title\n\nbody")
        self.assertEqual(get.call_args.args[0], "https://r.jina.ai/https://example.com/docs")
        self.assertEqual(get.call_args.kwargs["headers"]["X-Return-Format"], "markdown")
        self.assertEqual(get.call_args.kwargs["timeout"], tools.WEB_FETCH_TIMEOUT)

    def test_a_url_that_already_has_a_scheme_is_left_alone(self):
        response = mock.Mock(ok=True, status_code=200, content=b"x")
        with mock.patch("requests.get", return_value=response) as get:
            self.call(url="http://example.com")
        self.assertEqual(get.call_args.args[0], "https://r.jina.ai/http://example.com")

    def test_the_key_goes_out_when_the_environment_has_one(self):
        response = mock.Mock(ok=True, status_code=200, content=b"x")
        with mock.patch("requests.get", return_value=response) as get, mock.patch.dict(
            os.environ, {"JINA_API_KEY": "jina_k"}
        ):
            self.call(url="https://example.com")
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer jina_k")

    def test_an_http_error_is_reported_to_the_model(self):
        response = mock.Mock(
            ok=False, status_code=429, content=b"slow down", reason="Too Many Requests"
        )
        with mock.patch("requests.get", return_value=response):
            text = self.call(url="https://example.com")
        self.assertIn("429", text)
        self.assertIn("slow down", text)

    def test_a_dead_network_is_reported_to_the_model(self):
        with mock.patch("requests.get", side_effect=requests.ConnectionError("no route")):
            text = self.call(url="https://example.com")
        self.assertTrue(text.startswith("Error"))
        self.assertIn("no route", text)

    def test_an_empty_document_is_reported_to_the_model(self):
        response = mock.Mock(ok=True, status_code=200, content=b"   \n")
        with mock.patch("requests.get", return_value=response):
            text = self.call(url="https://example.com")
        self.assertIn("empty", text)

    def test_a_huge_page_is_truncated(self):
        response = mock.Mock(
            ok=True, status_code=200, content=b"y" * (tools.WEB_FETCH_MAX_CHARS + 500)
        )
        with mock.patch("requests.get", return_value=response):
            text = self.call(url="https://example.com")
        self.assertIn("truncated", text)
        self.assertLess(len(text), tools.WEB_FETCH_MAX_CHARS + 200)


if __name__ == "__main__":
    unittest.main()
