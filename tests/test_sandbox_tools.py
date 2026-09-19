"""Tests for `src/sandbox_tools.py`: the registry, the reaper and the tool text.

Nothing here builds a real sandbox. The registry takes its creator as a
parameter, so a fake that speaks the small part of the `Sandbox` surface the
tools touch drives the whole lifecycle; the fixed configuration is a
`SandboxSpec` value object, which pins every limit without a kernel, Nix or
bubblewrap. The `sandbox` package's own integration tests are what prove the
limits are enforced.
"""

import base64
import itertools
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from sandbox import ExecResult, SandboxError
from tests.support import local_tool, param, sandbox_tools, server_tool, tools


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeSandbox:
    """The slice of `sandbox.Sandbox` the `nix_*` tools use."""

    def __init__(self, packages=("bash", "coreutils")):
        self._packages = tuple(packages)
        self.files = {}
        self.exec_calls = []
        self.exec_result = ExecResult(command=("true",), exit_code=0, stdout="", stderr="")
        self.destroyed = False
        self.destroy_error = None
        self.warnings = []
        self.failure = None

    @property
    def packages(self):
        return self._packages

    def _fail_if_asked(self):
        if self.failure is not None:
            raise self.failure

    def add_packages(self, *packages):
        self._fail_if_asked()
        for package in packages:
            if package not in self._packages:
                self._packages += (package,)
        return self._packages

    def remove_packages(self, *packages):
        self._fail_if_asked()
        for package in packages:
            if package in ("bash", "coreutils"):
                raise SandboxError(
                    f"{package} cannot be removed: bash and coreutils are always in the environment"
                )
            if package not in self._packages:
                raise SandboxError(f"not in this sandbox's package list: {package}")
            self._packages = tuple(name for name in self._packages if name != package)
        return self._packages

    def exec(self, command, *, timeout=None, max_output=None):
        self._fail_if_asked()
        self.exec_calls.append((tuple(command), timeout, max_output))
        return self.exec_result

    def put_file(self, path, data):
        self._fail_if_asked()
        if not (path == "/workspace" or path.startswith("/workspace/")):
            raise SandboxError(f"{path} is not under a writable mount: ['/workspace']")
        self.files[path] = bytes(data)

    def get_file(self, path):
        self._fail_if_asked()
        if path.startswith("/tmp/"):
            raise SandboxError(f"{path} is on a tmpfs, which exists for one command and is gone")
        if path not in self.files:
            raise SandboxError(f"could not read {path}: No such file or directory")
        return self.files[path]

    def destroy(self):
        self.destroyed = True
        if self.destroy_error is not None:
            raise self.destroy_error


class RegistryTestCase(unittest.TestCase):
    """Builds a registry whose sandboxes are fakes and whose ids are countable."""

    def make(self, *, create_error=None, **kwargs):
        clock = kwargs.pop("clock", FakeClock())
        created = []
        counter = itertools.count(1)

        def create():
            if create_error is not None:
                raise create_error
            sandbox = FakeSandbox()
            created.append(sandbox)
            return sandbox

        registry = sandbox_tools.SandboxRegistry(
            clock=clock,
            create=create,
            id_factory=lambda: f"sbx-{next(counter):04d}",
            **kwargs,
        )
        self.addCleanup(registry.shutdown)
        return registry, created


class FixedConfigurationTests(unittest.TestCase):
    """Spawn takes no configuration; the spec is the one fixed container."""

    def test_the_spec_is_the_fixed_configuration(self):
        spec = sandbox_tools.sandbox_spec()
        self.assertEqual(spec.packages, ())
        self.assertEqual(spec.cwd, sandbox_tools.SANDBOX_WORKSPACE)
        self.assertEqual(spec.writable, (sandbox_tools.SANDBOX_WORKSPACE,))
        self.assertEqual(spec.resources.memory, 256 * 1024**2)
        self.assertEqual(spec.resources.disk, 512 * 1024**2)
        self.assertEqual(spec.resources.pids, 256)
        self.assertEqual(spec.resources.cpu, 1.0)
        self.assertEqual(spec.resources.timeout, 600.0)
        self.assertEqual(spec.network.mode, "none")

    def test_the_workspace_is_the_writable_working_directory(self):
        spec = sandbox_tools.sandbox_spec()
        mount = spec.mount_for("/workspace/file.txt")
        self.assertIsNotNone(mount)
        self.assertTrue(mount.writable)
        self.assertFalse(mount.ephemeral)


class SpawnTests(RegistryTestCase):
    def test_spawn_returns_an_id_and_the_fixed_shape(self):
        registry, created = self.make()
        text = registry.spawn()
        self.assertIn("Created sandbox sbx-0001", text)
        self.assertIn("/workspace", text)
        self.assertIn("256M", text)
        self.assertIn("512M", text)
        self.assertIn("network none", text)
        self.assertIn("bash, coreutils", text)
        self.assertIn("nix_add_dependency", text)
        self.assertEqual(registry.live_ids(), ("sbx-0001",))
        self.assertEqual(len(created), 1)

    def test_ids_are_unique_across_sandboxes(self):
        registry, _ = self.make()
        first = registry.spawn().split()[2].rstrip(".")
        second = registry.spawn().split()[2].rstrip(".")
        self.assertNotEqual(first, second)
        self.assertEqual(registry.live_ids(), tuple(sorted((first, second))))

    def test_spawn_refuses_an_eleventh_sandbox(self):
        registry, _ = self.make()
        for _ in range(sandbox_tools.MAX_SANDBOXES):
            self.assertIn("Created sandbox", registry.spawn())
        self.assertEqual(len(registry.live_ids()), 10)

        text = registry.spawn()
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("10 already exist", text)
        self.assertIn("nix_destroy_sandbox", text)
        self.assertEqual(len(registry.live_ids()), 10)

        registry.destroy("sbx-0001")
        self.assertIn("Created sandbox", registry.spawn())
        self.assertEqual(len(registry.live_ids()), 10)

    def test_a_failing_create_is_reported_and_leaves_no_entry(self):
        registry, _ = self.make(create_error=SandboxError("nix is not installed"))
        text = registry.spawn()
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("nix is not installed", text)
        self.assertEqual(registry.live_ids(), ())


class StatusTests(RegistryTestCase):
    def test_status_reports_a_live_sandbox(self):
        registry, _ = self.make()
        registry.spawn()
        text = registry.status("sbx-0001")
        self.assertIn("Sandbox sbx-0001 is live", text)
        self.assertIn("created less than a second ago", text)
        self.assertIn("packages: bash, coreutils", text)
        self.assertIn(
            "limits: memory 256M, disk 512M, pids 256, cpu 1.0, network none", text
        )
        self.assertIn("/workspace", text)

    def test_status_says_a_destroyed_sandbox_is_gone(self):
        registry, _ = self.make()
        registry.spawn()
        registry.destroy("sbx-0001")
        text = registry.status("sbx-0001")
        self.assertIn("is not live", text)
        self.assertIn("nix_spawn_sandbox", text)

    def test_status_says_an_unknown_id_is_gone(self):
        registry, _ = self.make()
        self.assertIn("is not live", registry.status("sbx-never-issued"))

    def test_a_warning_from_the_sandbox_is_reported(self):
        registry, created = self.make()
        registry.spawn()
        created[0].warnings.append("memory is not enforced here")
        self.assertIn("memory is not enforced here", registry.status("sbx-0001"))


class IdleTests(RegistryTestCase):
    def test_a_call_restarts_the_idle_clock(self):
        clock = FakeClock()
        registry, created = self.make(clock=clock, idle_timeout=600.0)
        registry.spawn()
        clock.advance(599)
        registry.status("sbx-0001")  # any call about it counts as use
        clock.advance(599)
        self.assertEqual(registry.sweep(), [])
        self.assertEqual(registry.live_ids(), ("sbx-0001",))
        clock.advance(1)  # now exactly the idle timeout since the status call
        self.assertEqual(registry.sweep(), ["sbx-0001"])
        self.assertTrue(created[0].destroyed)
        self.assertEqual(registry.live_ids(), ())

    def test_a_failed_operation_still_counts_as_a_call(self):
        clock = FakeClock()
        registry, _ = self.make(clock=clock, idle_timeout=600.0)
        registry.spawn()
        clock.advance(599)
        self.assertTrue(registry.exec("sbx-0001", "true", 0).startswith("Error:"))
        clock.advance(599)
        self.assertEqual(registry.sweep(), [])

    def test_sweep_releases_only_the_idle_ones(self):
        clock = FakeClock()
        registry, created = self.make(clock=clock, idle_timeout=600.0)
        registry.spawn()
        registry.spawn()
        clock.advance(601)
        registry.add_dependency("sbx-0002", "git")
        self.assertEqual(registry.sweep(), ["sbx-0001"])
        self.assertTrue(created[0].destroyed)
        self.assertFalse(created[1].destroyed)
        self.assertEqual(registry.live_ids(), ("sbx-0002",))

    def test_a_failing_destroy_does_not_stop_the_sweep(self):
        clock = FakeClock()
        registry, created = self.make(clock=clock, idle_timeout=600.0)
        registry.spawn()
        registry.spawn()
        created[0].destroy_error = SandboxError("could not kill the cgroup")
        clock.advance(601)
        self.assertEqual(registry.sweep(), ["sbx-0001", "sbx-0002"])
        self.assertEqual(registry.live_ids(), ())

    def test_a_call_in_flight_is_not_reaped(self):
        entered = threading.Event()
        release = threading.Event()
        clock = FakeClock()
        registry, created = self.make(clock=clock, idle_timeout=600.0)
        registry.spawn()

        def slow_exec(command, *, timeout=None, max_output=None):
            entered.set()
            release.wait(5.0)
            return ExecResult(command=tuple(command), exit_code=0, stdout="", stderr="")

        created[0].exec = slow_exec
        worker = threading.Thread(target=registry.exec, args=("sbx-0001", "sleep 600", 600))
        worker.start()
        try:
            self.assertTrue(entered.wait(5.0))
            clock.advance(10_000)
            self.assertEqual(registry.sweep(), [])
            self.assertFalse(created[0].destroyed)
        finally:
            release.set()
            worker.join(5.0)
        self.assertFalse(worker.is_alive())
        # the call finishing is itself a use, so the sandbox is still not idle
        self.assertEqual(registry.sweep(), [])

    def test_the_reaper_sweeps_on_its_own(self):
        registry, created = self.make(idle_timeout=0.0, sweep_interval=0.01)
        registry.spawn()
        deadline = time.monotonic() + 5.0
        while not created[0].destroyed and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(created[0].destroyed)
        self.assertEqual(registry.live_ids(), ())


class ExecTests(RegistryTestCase):
    def setUp(self):
        self.registry, self.created = self.make()
        self.registry.spawn()

    def test_exec_runs_bash_and_reports_both_streams(self):
        self.created[0].exec_result = ExecResult(
            command=("bash",), exit_code=3, stdout="out\n", stderr="err\n", duration=0.25
        )
        text = self.registry.exec("sbx-0001", "echo out; echo err >&2; exit 3", 30)
        self.assertIn("exit code 3 in 0.25s", text)
        self.assertIn("stdout:\nout", text)
        self.assertIn("stderr:\nerr", text)
        command, timeout, max_output = self.created[0].exec_calls[0]
        self.assertEqual(command, ("bash", "-c", "echo out; echo err >&2; exit 3"))
        self.assertEqual(timeout, 30.0)
        self.assertEqual(max_output, sandbox_tools.EXEC_CAPTURE_BYTES)

    def test_exec_says_so_when_there_is_no_output(self):
        text = self.registry.exec("sbx-0001", "true", 30)
        self.assertIn("exit code 0", text)
        self.assertIn("(no output)", text)

    def test_exec_requires_a_timeout_within_ten_minutes(self):
        for bad in (0, -1, 601, 10000, "30", None, True):
            with self.subTest(timeout=bad):
                text = self.registry.exec("sbx-0001", "true", bad)
                self.assertTrue(text.startswith("Error:"))
                self.assertIn("between 1 and 600", text)
        self.assertEqual(self.created[0].exec_calls, [])
        self.assertIn("exit code 0", self.registry.exec("sbx-0001", "true", 600))
        self.assertEqual(self.created[0].exec_calls[0][1], 600.0)

    def test_exec_rejects_an_empty_command(self):
        for bad in ("", "   ", None, 5):
            with self.subTest(command=bad):
                self.assertTrue(self.registry.exec("sbx-0001", bad, 30).startswith("Error:"))
        self.assertEqual(self.created[0].exec_calls, [])

    def test_exec_reports_a_timeout_with_the_partial_output(self):
        self.created[0].exec_result = ExecResult(
            command=("bash",),
            exit_code=124,
            stdout="half a line",
            stderr="",
            duration=30.0,
            timed_out=True,
        )
        text = self.registry.exec("sbx-0001", "sleep 60", 30)
        self.assertIn("timed out after 30.0s and was killed (exit code 124)", text)
        self.assertIn("half a line", text)

    def test_exec_clips_a_flood_of_output(self):
        self.created[0].exec_result = ExecResult(
            command=("bash",),
            exit_code=0,
            stdout="x" * (sandbox_tools.EXEC_OUTPUT_MAX_CHARS + 500),
            stderr="",
        )
        text = self.registry.exec("sbx-0001", "cat big", 30)
        self.assertIn("truncated", text)
        self.assertLess(len(text), sandbox_tools.EXEC_OUTPUT_MAX_CHARS + 500)

    def test_exec_says_when_the_sandbox_capped_the_capture(self):
        self.created[0].exec_result = ExecResult(
            command=("bash",), exit_code=0, stdout="y", stderr="", truncated=True
        )
        self.assertIn("output was capped", self.registry.exec("sbx-0001", "cat big", 30))

    def test_exec_reports_a_sandbox_failure(self):
        self.created[0].failure = SandboxError("this sandbox has been destroyed")
        text = self.registry.exec("sbx-0001", "true", 30)
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("this sandbox has been destroyed", text)

    def test_exec_on_a_released_sandbox_says_so(self):
        self.registry.destroy("sbx-0001")
        text = self.registry.exec("sbx-0001", "true", 30)
        self.assertIn("is not live", text)
        self.assertEqual(self.created[0].exec_calls, [])


class DependencyTests(RegistryTestCase):
    def setUp(self):
        self.registry, self.created = self.make()
        self.registry.spawn()

    def test_add_dependency_lands_in_the_next_environment(self):
        text = self.registry.add_dependency("sbx-0001", "git")
        self.assertIn("Added git", text)
        self.assertIn("next nix_exec", text)
        self.assertIn("bash, coreutils, git", text)
        self.assertEqual(self.created[0].packages, ("bash", "coreutils", "git"))

    def test_add_dependency_is_a_noop_when_already_present(self):
        text = self.registry.add_dependency("sbx-0001", "bash")
        self.assertIn("already in", text)

    def test_add_dependency_reports_a_nix_failure(self):
        self.created[0].failure = SandboxError("undefined variable 'nope'")
        text = self.registry.add_dependency("sbx-0001", "nope")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("undefined variable 'nope'", text)

    def test_add_dependency_rejects_an_empty_name(self):
        for bad in ("", "  ", None, 7):
            with self.subTest(package=bad):
                self.assertTrue(self.registry.add_dependency("sbx-0001", bad).startswith("Error:"))

    def test_remove_dependency_removes_it(self):
        self.registry.add_dependency("sbx-0001", "ripgrep")
        text = self.registry.remove_dependency("sbx-0001", "ripgrep")
        self.assertIn("Removed ripgrep", text)
        self.assertIn("bash, coreutils", text)
        self.assertEqual(self.created[0].packages, ("bash", "coreutils"))

    def test_remove_dependency_reports_a_refusal(self):
        text = self.registry.remove_dependency("sbx-0001", "bash")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("bash cannot be removed", text)

    def test_remove_dependency_reports_an_unknown_package(self):
        text = self.registry.remove_dependency("sbx-0001", "nope")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("not in this sandbox's package list: nope", text)


class AddFileTests(RegistryTestCase):
    def setUp(self):
        self.registry, self.created = self.make()
        self.registry.spawn()

    def test_add_file_decodes_binary_base64(self):
        payload = bytes(range(256))
        encoded = base64.b64encode(payload).decode("ascii")
        text = self.registry.add_file("sbx-0001", "/workspace/data.bin", encoded)
        self.assertEqual(self.created[0].files["/workspace/data.bin"], payload)
        self.assertIn("Wrote 256 bytes to /workspace/data.bin", text)

    def test_add_file_accepts_wrapped_and_unpadded_base64(self):
        encoded = base64.b64encode(b"hello world").decode("ascii").rstrip("=")
        wrapped = encoded[:4] + "\n" + encoded[4:]
        self.registry.add_file("sbx-0001", "/workspace/hello.txt", wrapped)
        self.assertEqual(self.created[0].files["/workspace/hello.txt"], b"hello world")

    def test_add_file_marks_a_shebang_executable(self):
        content = base64.b64encode(b"#!/bin/sh\necho hi\n").decode("ascii")
        text = self.registry.add_file("sbx-0001", "/workspace/run.sh", content)
        self.assertIn("executable", text)

    def test_add_file_rejects_bad_base64(self):
        text = self.registry.add_file("sbx-0001", "/workspace/x.bin", "not base64!!")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("base64", text)
        self.assertEqual(self.created[0].files, {})

    def test_add_file_rejects_a_relative_path(self):
        text = self.registry.add_file("sbx-0001", "workspace/x.bin", "aGk=")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("absolute", text)

    def test_add_file_reports_a_path_outside_the_workspace(self):
        text = self.registry.add_file("sbx-0001", "/etc/passwd", "aGk=")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("not under a writable mount", text)

    def test_add_file_is_bounded(self):
        payload = base64.b64encode(b"x" * 64).decode("ascii")
        with mock.patch.object(sandbox_tools, "ADD_FILE_MAX_BYTES", 16):
            text = self.registry.add_file("sbx-0001", "/workspace/big.bin", payload)
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("at most 16 bytes", text)
        self.assertEqual(self.created[0].files, {})

    def test_add_file_takes_files_up_to_two_hundred_megabytes(self):
        self.assertEqual(sandbox_tools.ADD_FILE_MAX_BYTES, 200 * 1024**2)

    def test_add_file_allows_a_file_exactly_at_the_cap(self):
        payload = base64.b64encode(b"x" * 64).decode("ascii")
        with mock.patch.object(sandbox_tools, "ADD_FILE_MAX_BYTES", 64):
            text = self.registry.add_file("sbx-0001", "/workspace/ok.bin", payload)
        self.assertIn("Wrote 64 bytes", text)
        self.assertEqual(self.created[0].files["/workspace/ok.bin"], b"x" * 64)

    def test_add_file_on_a_released_sandbox_says_so(self):
        self.registry.destroy("sbx-0001")
        self.assertIn("is not live", self.registry.add_file("sbx-0001", "/workspace/x", "aGk="))


class CatFileTests(RegistryTestCase):
    def setUp(self):
        self.registry, self.created = self.make()
        self.registry.spawn()

    def test_cat_file_returns_the_bytes(self):
        payload = bytes(range(256))
        self.created[0].files["/workspace/out.bin"] = payload
        self.assertEqual(self.registry.cat_file("sbx-0001", "/workspace/out.bin"), payload)

    def test_cat_file_reads_an_empty_file(self):
        self.created[0].files["/workspace/empty.txt"] = b""
        self.assertEqual(self.registry.cat_file("sbx-0001", "/workspace/empty.txt"), b"")

    def test_cat_file_rejects_a_relative_path(self):
        text = self.registry.cat_file("sbx-0001", "workspace/out.bin")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("absolute", text)

    def test_cat_file_reports_a_file_the_sandbox_will_not_read(self):
        text = self.registry.cat_file("sbx-0001", "/tmp/scratch.bin")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("tmpfs", text)

    def test_cat_file_reports_a_missing_file(self):
        text = self.registry.cat_file("sbx-0001", "/workspace/gone.bin")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("No such file or directory", text)

    def test_cat_file_reports_a_sandbox_failure(self):
        self.created[0].failure = SandboxError("the sandbox has been destroyed")
        text = self.registry.cat_file("sbx-0001", "/workspace/out.bin")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("the sandbox has been destroyed", text)

    def test_cat_file_refuses_a_file_past_the_limit(self):
        with mock.patch.object(sandbox_tools, "CAT_FILE_MAX_BYTES", 16):
            self.created[0].files["/workspace/big.bin"] = b"x" * 17
            text = self.registry.cat_file("sbx-0001", "/workspace/big.bin")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("17 bytes", text)
        self.assertIn("16 byte nix_cat_file limit", text)
        self.assertIn("nix_exec", text)

    def test_cat_file_allows_a_file_exactly_at_the_limit(self):
        with mock.patch.object(sandbox_tools, "CAT_FILE_MAX_BYTES", 64):
            self.created[0].files["/workspace/ok.bin"] = b"x" * 64
            self.assertEqual(self.registry.cat_file("sbx-0001", "/workspace/ok.bin"), b"x" * 64)

    def test_cat_file_takes_files_up_to_two_hundred_megabytes(self):
        self.assertEqual(sandbox_tools.CAT_FILE_MAX_BYTES, 200 * 1024**2)

    def test_cat_file_on_a_released_sandbox_says_so(self):
        self.registry.destroy("sbx-0001")
        self.assertIn("is not live", self.registry.cat_file("sbx-0001", "/workspace/out.bin"))


class DestroyTests(RegistryTestCase):
    def test_destroy_stops_the_sandbox_and_frees_the_slot(self):
        registry, created = self.make()
        registry.spawn()
        text = registry.destroy("sbx-0001")
        self.assertIn("Destroyed sandbox sbx-0001", text)
        self.assertTrue(created[0].destroyed)
        self.assertEqual(registry.live_ids(), ())

    def test_destroy_on_a_gone_sandbox_says_so(self):
        registry, _ = self.make()
        text = registry.destroy("sbx-0001")
        self.assertIn("nothing to destroy", text)

    def test_destroy_reports_a_failing_teardown(self):
        registry, created = self.make()
        registry.spawn()
        created[0].destroy_error = SandboxError("could not remove the state directory")
        text = registry.destroy("sbx-0001")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("could not remove the state directory", text)
        self.assertEqual(registry.live_ids(), ())


class ToolWiringTests(unittest.TestCase):
    """The ToolEntry hooks are thin: they hand their arguments to the registry."""

    def setUp(self):
        self.context = SimpleNamespace(
            tools={
                "consume": server_tool(
                    "consume",
                    lambda context, path, data: "",
                    params=[param("path"), param("data")],
                )
            }
        )

    def test_the_hooks_call_the_module_registry(self):
        with mock.patch.object(sandbox_tools, "SANDBOXES") as registry:
            registry.spawn.return_value = "spawned"
            registry.status.return_value = "status"
            registry.add_dependency.return_value = "added"
            registry.remove_dependency.return_value = "removed"
            registry.exec.return_value = "executed"
            registry.add_file.return_value = "written"
            registry.cat_file.return_value = b"bytes"
            registry.destroy.return_value = "destroyed"

            self.assertEqual(tools.nix_spawn_sandbox_executor(self.context), "spawned")
            self.assertEqual(
                tools.nix_sandbox_status_executor(self.context, sandbox_id="sbx-1"), "status"
            )
            self.assertEqual(
                tools.nix_add_dependency_executor(self.context, sandbox_id="sbx-1", package="git"),
                "added",
            )
            self.assertEqual(
                tools.nix_remove_dependency_executor(
                    self.context, sandbox_id="sbx-1", package="git"
                ),
                "removed",
            )
            self.assertEqual(
                tools.nix_exec_executor(
                    self.context, sandbox_id="sbx-1", command="true", timeout=30
                ),
                "executed",
            )
            self.assertEqual(
                tools.nix_add_file_executor(
                    self.context, sandbox_id="sbx-1", path="/workspace/x", content_base64="aGk="
                ),
                "written",
            )
            piped = tools.nix_cat_file_executor(
                self.context, sandbox_id="sbx-1", path="/workspace/x", tool_name="consume"
            )
            self.assertEqual(
                piped.call,
                tools.ToolCall("consume", {"path": "/workspace/x", "data": b"bytes"}),
            )
            self.assertEqual(
                tools.nix_destroy_sandbox_executor(self.context, sandbox_id="sbx-1"), "destroyed"
            )

        registry.exec.assert_called_once_with("sbx-1", "true", 30)
        registry.add_file.assert_called_once_with("sbx-1", "/workspace/x", "aGk=")
        registry.cat_file.assert_called_once_with("sbx-1", "/workspace/x")

    def test_spawn_takes_no_parameters(self):
        tool = next(tool for tool in tools.builtin_tools if tool.name == "nix_spawn_sandbox")
        self.assertEqual(tool.params, [])

    def test_exec_requires_all_three_parameters(self):
        tool = next(tool for tool in tools.builtin_tools if tool.name == "nix_exec")
        self.assertEqual([param.name for param in tool.params], ["sandbox_id", "command", "timeout"])
        self.assertEqual(tool.params[-1].type, "integer")
        self.assertIn("600", tool.params[-1].description)

    def test_cat_file_takes_a_sandbox_a_path_and_a_target_tool(self):
        tool = next(tool for tool in tools.builtin_tools if tool.name == "nix_cat_file")
        self.assertEqual(
            [param.name for param in tool.params], ["sandbox_id", "path", "tool_name"]
        )
        self.assertIn("200 MiB", tool.description)

    def test_the_sandbox_tools_declare_their_effects(self):
        for tool in tools.builtin_tools:
            if not tool.name.startswith("nix_"):
                continue
            with self.subTest(tool=tool.name):
                self.assertFalse(tool.is_local)
                self.assertFalse(tool.has_rollback)
                self.assertTrue(tool.description)
                self.assertEqual(tool.namespace, tool.name)
                if tool.name in ("nix_sandbox_status", "nix_cat_file"):
                    self.assertFalse(tool.external_effects)
                else:
                    self.assertTrue(tool.external_effects)


class CatFileToolTests(unittest.TestCase):
    """A read file becomes the target tool's two arguments: path, then bytes."""

    def setUp(self):
        self.payload = bytes(range(64))
        self.registry = mock.MagicMock()
        self.registry.cat_file.return_value = self.payload
        patcher = mock.patch.object(sandbox_tools, "SANDBOXES", self.registry)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.context = SimpleNamespace(tools={})
        self.context.tools["consume"] = server_tool(
            "consume",
            lambda context, path, data: "",
            params=[param("path"), param("data")],
        )

    def cat(self, tool_name="consume", path="/workspace/out.bin", sandbox_id="sbx-1"):
        return tools.nix_cat_file_executor(
            self.context, sandbox_id=sandbox_id, path=path, tool_name=tool_name
        )

    def test_the_path_and_the_bytes_become_the_two_arguments(self):
        result = self.cat()
        self.registry.cat_file.assert_called_once_with("sbx-1", "/workspace/out.bin")
        self.assertIsInstance(result, tools.ToolResult)
        self.assertEqual(
            result.call,
            tools.ToolCall("consume", {"path": "/workspace/out.bin", "data": self.payload}),
        )
        # the note is for the trace: it never reaches the model, the pipe does
        self.assertEqual(
            result.text, "read 64 bytes from /workspace/out.bin in sandbox sbx-1"
        )

    def test_the_arguments_follow_the_targets_own_parameter_order(self):
        self.context.tools["store"] = server_tool(
            "store",
            lambda context, name, payload: "",
            params=[param("name"), param("payload")],
        )
        piped = self.cat(tool_name="store")
        self.assertEqual(
            piped.call.arguments,
            {"name": "/workspace/out.bin", "payload": self.payload},
        )

    def test_a_read_that_fails_is_text_not_a_pipe(self):
        self.registry.cat_file.return_value = "Error: sandbox sbx-1 is not live"
        self.assertEqual(self.cat(), "Error: sandbox sbx-1 is not live")

    def test_a_missing_tool_name_is_refused_before_anything_is_read(self):
        text = self.cat(tool_name="")
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("tool_name", text)
        self.registry.cat_file.assert_not_called()

    def test_an_unknown_tool_is_refused_before_anything_is_read(self):
        text = self.cat(tool_name="nowhere")
        self.assertIn("unknown tool 'nowhere'", text)
        self.assertIn("consume", text)
        self.registry.cat_file.assert_not_called()

    def test_a_tool_that_takes_too_many_parameters_is_refused(self):
        self.context.tools["write"] = server_tool(
            "write",
            lambda context, path, data, mode: "",
            params=[param("path"), param("data"), param("mode")],
        )
        text = self.cat(tool_name="write")
        self.assertIn("3 parameters (path, data, mode)", text)
        self.assertIn("exactly two", text)
        self.registry.cat_file.assert_not_called()

    def test_a_tool_that_takes_one_parameter_is_refused(self):
        self.context.tools["show"] = server_tool(
            "show", lambda context, data: "", params=[param("data")]
        )
        text = self.cat(tool_name="show")
        self.assertIn("1 parameters (data)", text)
        self.assertIn("exactly two", text)
        self.registry.cat_file.assert_not_called()

    def test_a_tool_that_takes_no_parameters_is_refused(self):
        self.context.tools["ping"] = server_tool("ping", lambda context: "")
        self.assertIn("0 parameters", self.cat(tool_name="ping"))

    def test_a_tool_that_runs_on_the_client_is_refused(self):
        self.context.tools["upload"] = local_tool(
            "upload", params=[param("path"), param("data")]
        )
        text = self.cat(tool_name="upload")
        self.assertIn("runs on the client", text)
        self.assertIn("runs on the server", text)
        self.registry.cat_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
