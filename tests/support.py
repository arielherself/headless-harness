"""Shared infrastructure for the headless-harness test-suite.

Everything here runs offline, over loopback sockets only:

* `FakeProvider` is a scripted stand-in for the OpenAI chat/completions API. It
  speaks real HTTP + SSE so `requests` and `HHAgent._sse_payloads` are exercised
  for real, and it records every request it was sent.
* `Client` is a JSONL client for `HHServer` with a background reader and
  blocking waits.
* `ServerFixture` runs a real `HHServer` in-process on an ephemeral port, with a
  temporary SQLite store, and `HHTestCase` wires that into `unittest`.

No test needs a provider key or outbound network access.
"""

import collections
import http.server
import json
import pathlib
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import agent  # noqa: E402
import server  # noqa: E402
import store  # noqa: E402
import tools  # noqa: E402

TEST_KEY = "test-key-that-is-not-a-secret"
TEST_MODEL = "test-model"


# --- provider response scripting ----------------------------------------


def delta(**fields):
    """A chunk carrying `fields` inside `choices[0].delta`."""
    return {"choices": [{"index": 0, "delta": fields}]}


def finish(reason="stop"):
    """A chunk that only reports why the model stopped."""
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def usage(prompt_tokens=10, completion_tokens=5):
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def call_delta(index, call_id=None, name=None, arguments=None):
    """One `tool_calls` fragment, as a provider streams them."""
    fragment = {"index": index}
    if call_id:
        fragment["id"] = call_id
        fragment["type"] = "function"
    function = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    if function:
        fragment["function"] = function
    return {"choices": [{"index": 0, "delta": {"tool_calls": [fragment]}}]}


class Response:
    """One scripted provider response.

    `chunks` is a list where a `dict` becomes `data: <json>` and a `str` becomes
    a `data:` line too (unless it already starts with `data:` or `:`, which lets
    a test emit a comment or a malformed payload). `gap` sleeps between chunks
    and `truncate` cuts the body short after declaring its full length, which is
    how a mid-stream connection failure is reproduced.
    """

    def __init__(
        self,
        chunks=None,
        status=200,
        body=None,
        headers=None,
        gap=0.0,
        delay=0.0,
        truncate=False,
    ):
        self.chunks = list(chunks or [])
        self.status = status
        self.body = body
        self.headers = dict(headers or {})
        self.gap = gap
        self.delay = delay
        self.truncate = truncate

    # --- constructors
    @classmethod
    def text(
        cls,
        text,
        chunk_size=3,
        usage_chunk=None,
        finish_reason="stop",
        done=True,
        **kwargs,
    ):
        """A plain assistant reply, streamed in `chunk_size` pieces."""
        chunks = [
            delta(content=text[i : i + chunk_size])
            for i in range(0, len(text), chunk_size)
        ]
        if finish_reason:
            chunks.append(finish(finish_reason))
        if usage_chunk is not None:
            chunks.append(usage_chunk)
        if done:
            chunks.append("[DONE]")
        return cls(chunks=chunks, **kwargs)

    @classmethod
    def tool_calls(
        cls,
        calls,
        text="",
        arguments_split=1,
        name_split=False,
        usage_chunk=None,
        finish_reason="tool_calls",
        done=True,
        **kwargs,
    ):
        """Tool calls, each optionally fragmented across several deltas.

        `calls` is a list of `(name, arguments)` or `(name, arguments, call_id)`
        tuples; `arguments` may be a dict, which is serialised like a provider
        would. `arguments_split` slices the JSON into that many deltas so the
        reassembly in `_Turn.absorb` is exercised for real, and `name_split`
        sends the name itself in two fragments, as some providers do.
        """
        chunks = []
        if text:
            chunks += [
                delta(content=text[i : i + 3]) for i in range(0, len(text), 3)
            ]
        for index, call in enumerate(calls):
            name, arguments = call[0], call[1]
            call_id = call[2] if len(call) > 2 else f"call_{index + 1}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            if name_split and len(name) > 1:
                half = len(name) // 2
                chunks.append(call_delta(index, call_id=call_id, name=name[:half]))
                chunks.append(call_delta(index, name=name[half:]))
            else:
                chunks.append(call_delta(index, call_id=call_id, name=name))
            for piece in _split(arguments, arguments_split):
                chunks.append(call_delta(index, arguments=piece))
        if finish_reason:
            chunks.append(finish(finish_reason))
        if usage_chunk is not None:
            chunks.append(usage_chunk)
        if done:
            chunks.append("[DONE]")
        return cls(chunks=chunks, **kwargs)

    @classmethod
    def tool_call(cls, name, arguments, call_id=None, **kwargs):
        """A single tool call: `Response.tool_calls([(name, arguments)])`."""
        call = (name, arguments) if call_id is None else (name, arguments, call_id)
        return cls.tool_calls([call], **kwargs)

    @classmethod
    def steady(cls, pieces=8, chars=700, gap=0.01, fill="x", **kwargs):
        """A slow stream of large deltas.

        Each delta is written as its own read-sized block (over 512 bytes), so
        the client really does receive the reply piece by piece — which is what
        lets a test watch it arrive and interrupt it mid-flight.
        """
        return cls.text(fill * chars * pieces, chunk_size=chars, gap=gap, **kwargs)

    @classmethod
    def error(cls, status=500, body=b"provider exploded", **kwargs):
        """An HTTP refusal, not a stream."""
        return cls(status=status, body=body, **kwargs)

    def encode_chunk(self, chunk):
        if isinstance(chunk, bytes):
            return chunk
        if isinstance(chunk, dict):
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
        text = str(chunk)
        if not text.startswith("data:") and not text.startswith(":"):
            text = f"data: {text}"
        if not text.endswith("\n\n"):
            text += "\n\n"
        return text.encode("utf-8")


def _split(text, pieces):
    if pieces <= 1 or not text:
        return [text]
    size = max(len(text) // pieces, 1)
    return [text[i : i + size] for i in range(0, len(text), size)]


class RecordedRequest:
    """One request the fake provider received."""

    def __init__(self, path, headers, raw, payload):
        self.path = path
        self.headers = headers
        self.raw = raw
        self.payload = payload

    @property
    def messages(self):
        return (self.payload or {}).get("messages") or []

    def __repr__(self):
        return f"<RecordedRequest {self.path} {len(self.raw)} bytes>"


# --- the fake provider ---------------------------------------------------


class _ProviderHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FakeProvider/1.0"

    def log_message(self, *args):  # keep the test output clean
        pass

    def do_POST(self):
        provider = self.server.provider
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = None
        provider.record(RecordedRequest(self.path, dict(self.headers), raw, payload))
        response = provider.next_response(payload)
        try:
            if response.delay:
                time.sleep(response.delay)
            self._write(response)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True

    def _write(self, response):
        if response.body is not None:
            self.send_response(response.status)
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            self.wfile.write(response.body)
            self.wfile.flush()
            return

        encoded = [response.encode_chunk(chunk) for chunk in response.chunks]
        body = b"".join(encoded)
        self.send_response(response.status)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        for key, value in response.headers.items():
            self.send_header(key, value)
        if response.truncate:
            # declare the full body, then hang up half way through it
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            cut = max(len(body) // 2, 1)
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.close_connection = True
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        for piece in encoded:
            self.wfile.write(piece)
            self.wfile.flush()
            if response.gap:
                time.sleep(response.gap)


class FakeProvider:
    """A scripted chat/completions endpoint on loopback.

    Responses are served in the order they were pushed. An empty script answers
    with an HTTP 500 naming the problem, so a request a test forgot to script
    fails visibly instead of hanging.
    """

    def __init__(self):
        self.requests = []
        self._responses = collections.deque()
        self._routes = []  # (needle, response), matched against the payload
        self._lock = threading.Lock()
        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _ProviderHandler
        )
        self._server.daemon_threads = True
        self._server.provider = self
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="fake-provider",
            daemon=True,
        )
        self._thread.start()
        self.endpoint = f"http://127.0.0.1:{self._server.server_address[1]}"

    # --- scripting
    def push(self, response):
        with self._lock:
            self._responses.append(response)
        return response

    def text(self, text, **kwargs):
        return self.push(Response.text(text, **kwargs))

    def tool_call(self, name, arguments, **kwargs):
        return self.push(Response.tool_call(name, arguments, **kwargs))

    def error(self, status=500, body=b"provider exploded", **kwargs):
        return self.push(Response.error(status=status, body=body, **kwargs))

    def script(self, *responses):
        for response in responses:
            self.push(response)

    def match(self, needle, response):
        """Serve `response` to the next request whose payload mentions `needle`.

        Unlike the FIFO script this survives requests arriving in an order the
        test cannot control, which is what concurrent turns need.
        """
        with self._lock:
            self._routes.append((needle, response))
        return response

    def text_for(self, needle, text, **kwargs):
        return self.match(needle, Response.text(text, **kwargs))

    def clear_script(self):
        with self._lock:
            self._responses.clear()

    def pending(self):
        with self._lock:
            return len(self._responses)

    def next_response(self, payload=None):
        text = json.dumps(payload, ensure_ascii=False) if payload is not None else ""
        with self._lock:
            for index, (needle, response) in enumerate(self._routes):
                if needle in text:
                    self._routes.pop(index)
                    return response
            if self._responses:
                return self._responses.popleft()
        return Response.error(body=b"fake provider has no scripted response")

    # --- recording
    def record(self, request):
        with self._lock:
            self.requests.append(request)

    @property
    def count(self):
        with self._lock:
            return len(self.requests)

    def payloads(self):
        with self._lock:
            return [request.payload for request in self.requests]

    def last_payload(self):
        with self._lock:
            if not self.requests:
                raise AssertionError("the provider received no request")
            return self.requests[-1].payload

    def wait_for(self, count, timeout=10.0):
        """Block until at least `count` requests have been received."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.count >= count:
                return
            time.sleep(0.005)
        raise AssertionError(
            f"expected {count} provider requests, saw {self.count}"
        )

    def close(self):
        self._server.shutdown()
        self._server.server_close()


# --- the JSONL client ----------------------------------------------------


class Client:
    """A line-delimited JSON client for a running `HHServer`."""

    def __init__(self, host, port, timeout=15.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(None)
        self._file = self.sock.makefile("rb")
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.events = []
        self.closed = False
        self._reader = threading.Thread(
            target=self._read_loop, name="test-client", daemon=True
        )
        self._reader.start()

    def _read_loop(self):
        try:
            for line in self._file:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    event = {"event": "__malformed__", "line": line.decode("utf-8", "replace")}
                with self._cond:
                    self.events.append(event)
                    self._cond.notify_all()
        except (OSError, ValueError):
            pass
        finally:
            with self._cond:
                self.closed = True
                self._cond.notify_all()

    # --- sending
    def send(self, command=None, **fields):
        payload = dict(fields)
        if command is not None:
            payload["command"] = command
        self.send_line(json.dumps(payload, ensure_ascii=False))
        return payload

    def send_line(self, text):
        self.sock.sendall((text + "\n").encode("utf-8"))

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def half_close(self):
        """Stop sending, but keep reading: the server sees EOF and closes."""
        self.sock.shutdown(socket.SHUT_WR)

    # --- waiting
    def mark(self):
        """An index into `events`, for waits that only care about what is new."""
        with self._lock:
            return len(self.events)

    def wait_ready(self, ready, timeout=15.0, description="matching event", since=0):
        """Wait until `ready(events[since:])` is truthy, and return its value."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                value = ready(self.events[since:])
                if value:
                    return value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"timed out after {timeout}s waiting for {description}; "
                        f"events seen: {[e.get('event') for e in self.events[since:]]}"
                    )
                self._cond.wait(min(0.05, remaining))

    def wait_event(self, name, timeout=15.0, since=0, **criteria):
        """The first event of that name (and matching every criterion)."""

        def ready(events):
            for event in events:
                if event.get("event") != name:
                    continue
                if all(event.get(key) == value for key, value in criteria.items()):
                    return event
            return None

        return self.wait_ready(ready, timeout, f"{name} {criteria}", since)

    def wait_events(self, name, count=1, timeout=15.0, since=0, **criteria):
        """Wait for at least `count` matching events, and return all of them."""

        def ready(events):
            found = [
                event
                for event in events
                if event.get("event") == name
                and all(event.get(key) == value for key, value in criteria.items())
            ]
            return found if len(found) >= count else None

        return self.wait_ready(ready, timeout, f"{count}× {name} {criteria}", since)

    def wait_command(self, rid, timeout=15.0, since=0):
        """Wait for the outcome of one command: its completion or its error."""

        def ready(events):
            for event in events:
                if event.get("rid") != rid:
                    continue
                if event.get("event") == "command_finished":
                    return event
                if event.get("event") == "error":
                    return event
            return None

        return self.wait_ready(ready, timeout, f"a reply to rid={rid!r}", since)

    def command(self, command, wait=15.0, expect="command_finished", rid=None, **fields):
        """Send a command and wait for its reply, asserting a clean status.

        The wait arguments are named `wait`/`expect` rather than `timeout`, so a
        command's own `timeout` field is sent as a field.
        """
        rid = rid if rid is not None else f"rid-{self.mark()}-{command}"
        self.send(command, rid=rid, **fields)
        reply = self.wait_command(rid, timeout=wait)
        self.assert_event(reply, expect)
        return reply

    def assert_event(self, event, name):
        if event.get("event") != name:
            raise AssertionError(
                f"expected {name}, got {event.get('event')}: {event}"
            )
        return event

    def wait_error(self, rid, code=None, timeout=15.0, since=0):
        event = self.wait_command(rid, timeout=timeout, since=since)
        self.assert_event(event, "error")
        if code is not None and event.get("code") != code:
            raise AssertionError(f"expected error {code!r}, got {event!r}")
        return event

    def assert_no_event(self, name, timeout=0.25, since=0, **criteria):
        """Assert that no matching event arrives within `timeout`."""
        try:
            event = self.wait_event(name, timeout=timeout, since=since, **criteria)
        except AssertionError:
            return None
        raise AssertionError(f"unexpected {name} event: {event}")

    def settle(self, seconds=0.2):
        """Give the server time to send anything it was going to send."""
        time.sleep(seconds)

    def named(self, name, since=0):
        with self._lock:
            events = list(self.events[since:])
        return [event for event in events if event.get("event") == name]

    def names(self, since=0):
        with self._lock:
            events = list(self.events[since:])
        return [event.get("event") for event in events]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- the in-process server ----------------------------------------------


class ServerFixture:
    """A real `HHServer` on an ephemeral port, with a temporary store."""

    def __init__(
        self,
        tmpdir,
        provider,
        defaults=None,
        with_store=True,
        max_db_bytes=0,
        db_name="harness.db",
    ):
        self.tmpdir = pathlib.Path(tmpdir)
        self.provider = provider
        self.db_path = self.tmpdir / db_name
        self.with_store = with_store
        self.store = None
        self.restore_warnings = []
        if with_store:
            self.store = store.HHStore(self.db_path, max_bytes=max_db_bytes)
        if defaults is None:
            defaults = {
                "endpoint": provider.endpoint,
                "key": TEST_KEY,
                "model": TEST_MODEL,
            }
        self.server = server.HHServer(
            ("127.0.0.1", 0), defaults=defaults, store=self.store
        )
        self.restore_warnings = list(self.server.restore_warnings)
        self.host, self.port = self.server.server_address[:2]
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="test-server",
            daemon=True,
        )
        self._thread.start()
        self._clients = []

    def client(self, **kwargs):
        created = Client(self.host, self.port, **kwargs)
        self._clients.append(created)
        return created

    def agent(self, agent_id):
        """The live block, for asserting on internals."""
        with self.server._registry_lock:
            return self.server._agents.get(agent_id)

    def stop(self):
        for created in self._clients:
            created.close()
        self.server.shutdown()
        self.server.server_close()
        if self.store is not None:
            self.store.close()


# --- unittest wiring -----------------------------------------------------


class HHTestCase(unittest.TestCase):
    """Base class: a temp dir, a fake provider, and auto-cleaned servers."""

    with_store = False
    max_db_bytes = 0
    server_defaults = None

    def setUp(self):
        self.tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="hh-tests-"))
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.provider = FakeProvider()
        self.addCleanup(self.provider.close)
        self._fixtures = []
        if self.server_defaults is None:
            self.server_defaults = {
                "endpoint": self.provider.endpoint,
                "key": TEST_KEY,
                "model": TEST_MODEL,
            }

    def tearDown(self):
        for fixture in reversed(self._fixtures):
            fixture.stop()
        self._fixtures.clear()

    # --- servers
    def start_server(self, **kwargs):
        kwargs.setdefault("with_store", self.with_store)
        kwargs.setdefault("max_db_bytes", self.max_db_bytes)
        kwargs.setdefault("defaults", dict(self.server_defaults))
        fixture = ServerFixture(self.tmpdir, self.provider, **kwargs)
        self._fixtures.append(fixture)
        return fixture

    def stop_server(self, fixture):
        if fixture in self._fixtures:
            self._fixtures.remove(fixture)
        fixture.stop()

    def client(self, fixture=None, **kwargs):
        if fixture is None:
            fixture = self._fixtures[0] if self._fixtures else self.start_server()
        return fixture.client(**kwargs)

    # --- agents
    def root(self, **kwargs):
        kwargs.setdefault("endpoint", self.provider.endpoint)
        kwargs.setdefault("key", TEST_KEY)
        kwargs.setdefault("model", TEST_MODEL)
        return agent.HHAgent.root(**kwargs)

    def turn(self, block, on_event=None):
        """Run a block's turn on a background thread."""
        return TurnThread(block, on_event=on_event).start()

    def run_turn(self, block, on_event=None, timeout=15.0):
        """Run a block's turn to completion and return the joined reply."""
        thread = self.turn(block, on_event=on_event)
        thread.join(timeout)
        if thread.error is not None:
            raise thread.error
        return thread.text

    def wait_until(self, predicate, timeout=10.0, description="condition"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError(f"timed out after {timeout}s waiting for {description}")


class TurnThread:
    """One block's turn, on a thread a test can observe and interrupt."""

    def __init__(self, block, on_event=None):
        self.block = block
        self.events = []
        self.chunks = []
        self.error = None
        self.finished = threading.Event()
        self._extra = on_event
        self._thread = threading.Thread(
            target=self._run, name=f"turn-{block.id}", daemon=True
        )

    def _run(self):
        try:
            for chunk in self.block.stream(self._on_event):
                self.chunks.append(chunk)
        except BaseException as exc:  # noqa: BLE001 - the test asserts on it
            self.error = exc
        finally:
            self.finished.set()

    def _on_event(self, event):
        self.events.append(event)
        if self._extra is not None:
            self._extra(event)

    def start(self):
        self._thread.start()
        return self

    def join(self, timeout=15.0):
        if not self.finished.wait(timeout):
            raise AssertionError(f"turn for {self.block.id} never finished")
        return self

    @property
    def text(self):
        return "".join(self.chunks)

    def named(self, name):
        return [event for event in self.events if event.get("event") == name]

    def last(self, name):
        found = self.named(name)
        if not found:
            raise AssertionError(f"no {name} event; saw {[e.get('event') for e in self.events]}")
        return found[-1]

    def wait_event(self, name, timeout=10.0, **criteria):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in list(self.events):
                if event.get("event") != name:
                    continue
                if all(event.get(key) == value for key, value in criteria.items()):
                    return event
            time.sleep(0.005)
        raise AssertionError(
            f"turn never emitted {name} {criteria}; saw {[e.get('event') for e in self.events]}"
        )


# --- small helpers -------------------------------------------------------


def local_tool(name="ask_operator", description="Ask the operator.", params=None, rollback=False, external_effects=False):
    """A client-run tool definition, as `local_tools` would declare it."""
    return tools.ToolEntry(
        name=name,
        description=description,
        params=params or [],
        hook=None,
        remote_rollback=rollback,
        external_effects=external_effects,
    )


def server_tool(name, hook, params=None, rollback=None, namespace="", external_effects=False):
    """A server-run tool definition for agent-level tests."""
    return tools.ToolEntry(
        name=name,
        description=f"{name} test tool",
        params=params or [],
        hook=hook,
        state_namespace=namespace,
        rollback=rollback,
        external_effects=external_effects,
    )


def param(name, type="string", description="a parameter"):
    return tools.ToolParam(name=name, type=type, description=description)


def pick(events, name):
    """Every event of that name, from a list of event dicts."""
    return [event for event in events if event.get("event") == name]


def inner(chunk):
    """The `choices[0].delta` of a chunk: what `_Turn.absorb` consumes."""
    return chunk["choices"][0]["delta"]
