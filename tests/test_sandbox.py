"""Tests for the `sandbox` package: the spec, the filter, the argv, the limits.

The unit tests need nothing but a checkout. The integration tests create real
sandboxes and prove the isolation properties end to end; they are skipped unless
bubblewrap is installed, user namespaces are available, and there is an
environment to run (the host's own binaries, so no Nix or network is needed).
"""

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from sandbox import (
    IoLimits,
    Mount,
    NetworkPolicy,
    ResourceLimits,
    Sandbox,
    SandboxDiskExceeded,
    SandboxError,
    SandboxSpec,
    SpecError,
    SyscallPolicy,
    seccomp,
    toolchain as toolchain_module,
)
from sandbox import bubblewrap, limits as limits_module
from sandbox import __main__ as main
from sandbox.limits import CgroupLimiter, LimitsReport, create_limiter


class SizeAndDurationTests(unittest.TestCase):
    def test_sizes_are_binary_and_named_in_the_documentation(self):
        self.assertEqual(ResourceLimits.from_mapping({"memory": "512M"}).memory, 512 * 1024**2)
        self.assertEqual(ResourceLimits.from_mapping({"disk": "1G"}).disk, 1024**3)
        self.assertEqual(ResourceLimits.from_mapping({"memory": "1.5GiB"}).memory, int(1.5 * 1024**3))
        self.assertEqual(ResourceLimits.from_mapping({"memory": 4096}).memory, 4096)

    def test_a_bare_number_is_seconds_and_units_compound(self):
        self.assertEqual(ResourceLimits.from_mapping({"timeout": 30}).timeout, 30.0)
        self.assertEqual(ResourceLimits.from_mapping({"timeout": "1m30s"}).timeout, 90.0)
        self.assertEqual(ResourceLimits.from_mapping({"timeout": "500ms"}).timeout, 0.5)

    def test_nonsense_is_refused(self):
        for bad in ("", "lots", "5 potatoes"):
            with self.subTest(bad=bad):
                with self.assertRaises(SpecError):
                    ResourceLimits.from_mapping({"memory": bad})


class SpecTests(unittest.TestCase):
    def test_writable_paths_become_mounts_without_a_host_directory(self):
        spec = SandboxSpec(writable=["/work", "/out"])
        mounts = {mount.path: mount for mount in spec.all_mounts()}
        self.assertEqual(set(mounts), {"/work", "/out"})
        self.assertTrue(mounts["/out"].writable)
        self.assertIsNone(mounts["/out"].source)

    def test_a_tmpfs_mount_can_be_declared_inside_writable(self):
        spec = SandboxSpec(writable=["/work", Mount.tmpfs("/scratch", "64M")])
        mount = spec.mount_for("/scratch/deep/file")
        self.assertEqual(mount.kind, "tmpfs")
        self.assertEqual(mount.size, 64 * 1024**2)
        self.assertTrue(spec.mount_for("/work/file").writable)
        self.assertFalse(spec.mount_for("/work/file").ephemeral)

    def test_the_deepest_mount_wins(self):
        spec = SandboxSpec(
            writable=["/work"],
            mounts=[Mount.ro_bind("/host/data", "/work/inputs")],
        )
        self.assertEqual(spec.mount_for("/work/inputs/a.json").kind, "ro-bind")
        self.assertEqual(spec.mount_for("/work/other.txt").path, "/work")

    def test_two_mounts_for_one_path_are_refused(self):
        with self.assertRaises(SpecError):
            SandboxSpec(
                writable=["/work"],
                mounts=[Mount.ro_bind("/host", "/work")],
            )

    def test_a_read_only_mount_needs_a_source(self):
        with self.assertRaises(SpecError):
            Mount(path="/data", kind="ro-bind")

    def test_paths_must_be_absolute_and_free_of_dot_dot(self):
        with self.assertRaises(SpecError):
            SandboxSpec(writable=["work"])
        with self.assertRaises(SpecError):
            SandboxSpec(files={"../escape": "x"})

    def test_network_can_be_declared_in_several_shapes(self):
        self.assertTrue(SandboxSpec(network=False).network.isolated)
        self.assertEqual(SandboxSpec(network="host").network.mode, "host")
        self.assertEqual(SandboxSpec(network={"allow": ["pypi.org"]}).network.mode, "allow")
        self.assertEqual(SandboxSpec(network=NetworkPolicy.none()).network.allow, ())

    def test_packages_and_env_dir_together_are_a_contradiction(self):
        with self.assertRaises(SpecError):
            SandboxSpec(packages=["python312"], env_dir="/nix/store/whatever")

    def test_the_defaults_are_a_work_directory_and_a_watchdog(self):
        spec = SandboxSpec()
        self.assertEqual(spec.cwd, "/work")
        self.assertEqual(spec.writable, ("/work",))
        self.assertEqual(spec.resources.timeout, 60.0)
        self.assertTrue(spec.network.isolated)
        self.assertTrue(spec.syscalls.enabled)


class SeccompTests(unittest.TestCase):
    def setUp(self):
        self.program = seccomp.build(SyscallPolicy())

    def test_the_filter_is_a_bare_array_of_eight_byte_instructions(self):
        # bubblewrap computes len/8 for sock_fprog.len, so there is no header
        self.assertEqual(len(self.program.data) % 8, 0)
        self.assertEqual(self.program.instruction_count, len(self.program.data) // 8)
        self.assertLess(self.program.instruction_count, 4096)

    def test_the_program_checks_the_architecture_first(self):
        instructions = seccomp.decode(self.program.data)
        self.assertEqual(instructions[0], (0x20, 0, 0, 4))  # ld [4] = arch
        self.assertEqual(instructions[1], (0x15, 1, 0, seccomp.AUDIT_ARCH_X86_64))
        self.assertEqual(instructions[2][0], 0x06)  # ret
        self.assertEqual(instructions[2][3], seccomp.SECCOMP_RET_KILL_PROCESS)

    def test_x32_syscall_numbers_are_refused_on_x86_64(self):
        instructions = seccomp.decode(self.program.data)
        if self.program.arch != "x86_64":
            self.skipTest("the x32 ABI only exists on x86_64")
        self.assertEqual(instructions[4], (0x35, 0, 1, 0x40000000))  # jge x32 bit

    def test_every_denied_syscall_returns_errno_not_death(self):
        instructions = seccomp.decode(self.program.data)
        returns = [entry for entry in instructions if entry[0] == 0x06]
        errno_returns = [entry for entry in returns if entry[3] == 0x00050000 | 1]
        # one per denied syscall, plus the x32 guard in front of the list
        self.assertEqual(len(errno_returns), len(self.program.denied) + 1)
        self.assertEqual(instructions[-1], (0x06, 0, 0, seccomp.SECCOMP_RET_ALLOW))

    def test_the_default_list_covers_the_design(self):
        for name in (
            "mount",
            "umount2",
            "pivot_root",
            "ptrace",
            "kexec_load",
            "init_module",
            "finit_module",
            "delete_module",
            "bpf",
            "perf_event_open",
            "reboot",
            "swapon",
            "swapoff",
        ):
            with self.subTest(syscall=name):
                self.assertIn(name, self.program.denied)

    def test_allow_removes_an_entry_and_deny_adds_one(self):
        policy = SyscallPolicy(allow=("ptrace", "bpf"), deny=("socket",))
        program = seccomp.build(policy)
        self.assertNotIn("ptrace", program.denied)
        self.assertNotIn("bpf", program.denied)
        self.assertIn("socket", program.denied)
        self.assertIn("mount", program.denied)

    def test_enosys_is_a_different_errno(self):
        program = seccomp.build(SyscallPolicy(errno="ENOSYS"))
        instructions = seccomp.decode(program.data)
        self.assertEqual(instructions[-2][3], 0x00050000 | 38)

    def test_a_filter_that_is_off_produces_nothing(self):
        self.assertIsNone(seccomp.build(SyscallPolicy(enabled=False)))

    def test_an_unknown_architecture_fails_closed(self):
        with self.assertRaises(SpecError):
            seccomp.build(SyscallPolicy(), arch="mips")

    def test_a_syscall_can_be_denied_by_number_and_bad_numbers_are_refused(self):
        program = seccomp.build(SyscallPolicy(deny=(9999,)))
        self.assertIn("#9999", program.denied)
        self.assertIn(9999, program.numbers)
        for bad in (-1, 0x40000000):
            with self.subTest(number=bad):
                with self.assertRaises(SpecError):
                    seccomp.build(SyscallPolicy(deny=(bad,)))

    def test_names_missing_from_an_architecture_are_reported_not_guessed(self):
        program = seccomp.build(SyscallPolicy(), arch="aarch64")
        self.assertIn("iopl", program.skipped)  # x86 only
        self.assertNotIn("iopl", program.denied)
        self.assertIn("mount", program.denied)


class BubblewrapArgvTests(unittest.TestCase):
    def setUp(self):
        self.toolchain = toolchain_module.Toolchain(
            path="/nix/store/xyz-sandbox-env", packages=("bash",), origin="nix"
        )
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)

    def build(self, **spec_kwargs):
        """The argv for a spec, planned exactly the way `Sandbox.create` plans one."""
        from sandbox.sandbox import Layout, _plan_mounts

        new_session = spec_kwargs.pop("new_session", True)
        spec = SandboxSpec(**spec_kwargs)
        layout = Layout(
            root=Path(self.state.name),
            etc=Path(self.state.name) / "etc",
            ro=Path(self.state.name) / "ro",
            files=Path(self.state.name) / "fs",
        )
        for directory in (layout.etc, layout.ro, layout.files):
            directory.mkdir(exist_ok=True)
        mounts = _plan_mounts(spec, layout)
        return bubblewrap.build_argv(
            spec=spec,
            mounts=mounts,
            file_binds=(("/state/etc/passwd", "/etc/passwd"),),
            toolchain=self.toolchain,
            env={"PATH": "/nix/store/xyz-sandbox-env/bin"},
            cwd=spec.cwd,
            command=("echo", "hi"),
            seccomp_fd=7,
            info_fd=8,
            block_fd=9,
            new_session=new_session,
        )

    def test_a_session_keeps_the_pty_it_was_given(self):
        # --new-session exists to detach a sandbox from the caller's terminal.
        # A session runs on a pty of its own — none of the harness's terminal is
        # passed in — so it passes False, or `setsid` would take away the
        # controlling terminal the shell needs for job control.
        self.assertIn("--new-session", self.build())
        session = self.build(new_session=False)
        self.assertNotIn("--new-session", session)
        self.assertIn("--die-with-parent", session)

    def test_the_namespaces_are_asked_for_explicitly(self):
        argv = self.build()
        self.assertIn("--unshare-all", argv)
        self.assertIn("--unshare-user", argv)  # required by --disable-userns
        self.assertIn("--disable-userns", argv)
        self.assertIn("--die-with-parent", argv)
        self.assertIn("--new-session", argv)
        self.assertNotIn("--share-net", argv)
        self.assertEqual(argv[argv.index("--cap-drop") + 1], "ALL")

    def test_the_filter_and_the_handshake_fds_are_passed(self):
        argv = self.build()
        self.assertEqual(argv[argv.index("--seccomp") + 1], "7")
        self.assertEqual(argv[argv.index("--info-fd") + 1], "8")
        self.assertEqual(argv[argv.index("--block-fd") + 1], "9")

    def test_the_store_is_read_only_and_the_work_directory_is_not(self):
        argv = self.build(writable=["/work"])
        self.assertEqual(argv[argv.index("--ro-bind-try") + 1], "/nix/store")
        bind = argv.index("/work")
        self.assertEqual(argv[bind - 2], "--bind")
        self.assertTrue(argv[bind - 1].endswith("/work"))  # the state directory

    def test_engines_that_do_not_place_a_pid_do_not_ask_for_one(self):
        for limiter in (
            limits_module.NullLimiter(),
            limits_module.RlimitLimiter(LimitsReport("rlimit")),
            limits_module.SystemdLimiter("x", (), LimitsReport("systemd")),
        ):
            with self.subTest(engine=limiter.report.engine):
                self.assertFalse(limiter.needs_placement)

    def test_a_tmpfs_mount_carries_its_size(self):
        argv = self.build(writable=["/work"], mounts=[Mount.tmpfs("/scratch", "32M")])
        at = argv.index("/scratch")
        self.assertEqual(argv[at - 1], "--tmpfs")
        self.assertEqual(argv[at - 2], str(32 * 1024**2))
        self.assertEqual(argv[at - 3], "--size")

    def test_a_nested_mount_is_applied_after_its_parent(self):
        # the host-tools environment lives under /tmp, and /tmp is a tmpfs: the
        # bind has to come second or the tmpfs hides it
        argv = self.build(mounts=[Mount.ro_bind("/tmp/host-tools", "/tmp/host-tools")])
        tmpfs_at = argv.index("/tmp", argv.index("--tmpfs"))
        bind_at = argv.index("/tmp/host-tools")
        self.assertLess(tmpfs_at, bind_at)
        self.assertEqual(argv[argv.index("/tmp", argv.index("--tmpfs")) - 1], "--tmpfs")

    def test_bin_and_usr_bin_point_at_the_environment(self):
        argv = self.build()
        self.assertEqual(argv[argv.index("--symlink") + 1], "/nix/store/xyz-sandbox-env/bin")
        self.assertEqual(argv[argv.index("--symlink") + 2], "/bin")

    def test_a_symlink_under_a_mounted_tree_is_skipped(self):
        toolchain = toolchain_module.Toolchain(
            path="/env",
            mounts=(Mount.ro_bind("/usr"),),
            origin="host",
            link_paths=("/bin", "/usr/bin"),
        )
        argv = bubblewrap.build_argv(
            spec=SandboxSpec(writable=[]),
            mounts=(),
            file_binds=(),
            toolchain=toolchain,
            env={},
            cwd="/",
            command=("true",),
            seccomp_fd=None,
            info_fd=None,
            block_fd=None,
        )
        links = [argv[i + 2] for i, part in enumerate(argv) if part == "--symlink"]
        self.assertEqual(links, ["/bin"])  # /usr/bin is inside the mounted /usr

    def test_the_environment_starts_empty_and_the_command_is_last(self):
        argv = self.build()
        self.assertIn("--clearenv", argv)
        self.assertEqual(argv[argv.index("--setenv") + 1], "PATH")
        self.assertEqual(argv[-2:], ["echo", "hi"])
        self.assertEqual(argv[argv.index("--chdir") + 1], "/work")

    def test_the_root_is_mounted_read_only_unless_asked_otherwise(self):
        argv = self.build()
        self.assertEqual(argv[argv.index("--remount-ro") + 1], "/")
        self.assertNotIn("--remount-ro", self.build(read_only_root=False))

    def test_a_declared_mount_wins_over_a_built_in_one(self):
        # the built-ins are emitted first so the spec's own mount lands on top;
        # otherwise a declared /dev/shm or /proc would be silently overridden
        argv = self.build(mounts=[Mount.tmpfs("/dev/shm", "8M")])
        positions = [at for at, part in enumerate(argv) if part == "/dev/shm"]
        self.assertEqual(len(positions), 1)
        self.assertEqual(argv[positions[0] - 2], str(8 * 1024**2))
        nested = self.build(mounts=[Mount.ro_bind("/host/data", "/dev/data")])
        self.assertLess(nested.index("/dev"), nested.index("/dev/data"))

    def test_a_file_bind_lands_on_the_hosts_etc_paths(self):
        argv = self.build()
        at = argv.index("/etc/passwd", argv.index("--dev"))
        self.assertEqual(argv[at - 1], "/state/etc/passwd")

    def test_a_filename_is_never_mistaken_for_an_option(self):
        argv = self.build()
        self.assertEqual(argv[-2], "echo")
        separator = len(argv) - 3
        self.assertEqual(argv[separator], "--")


class CgroupLimiterTests(unittest.TestCase):
    """The cgroup engine's bookkeeping, exercised without a real cgroupfs.

    A delegated cgroup cannot be conjured in a test, so the base is patched for
    the success path and the interface files are laid out by hand; everything
    after that — which files get written, what `place` and `close` do — is the
    real code, and a directory that will not rmdir (as here) is the one place
    the fake differs from a kernel-managed group.
    """

    def setUp(self):
        self.tree = tempfile.TemporaryDirectory()
        self.base = Path(self.tree.name)
        for name in ("cgroup.controllers", "cgroup.subtree_control"):
            (self.base / name).write_text("")
        self.addCleanup(self.tree.cleanup)

    def _prepared(self) -> Path:
        """A cgroup directory with the interface files a kernel would provide."""
        target = self.base / "hh-sandbox-test"
        target.mkdir()
        for name in (
            "memory.max", "memory.swap.max", "memory.oom.group", "cpu.max",
            "pids.max", "cgroup.procs", "cgroup.kill",
        ):
            (target / name).write_text("")
        return target

    def test_the_limits_are_written_as_the_kernel_wants_them(self):
        target = self._prepared()
        limits = ResourceLimits.from_mapping({"memory": "512M", "cpu": 1.5, "pids": 64})
        applied, warnings = limits_module._apply(target, limits, str(self.base))
        self.assertEqual((target / "memory.max").read_text(), str(512 * 1024**2))
        self.assertEqual((target / "memory.swap.max").read_text(), "0")
        self.assertEqual((target / "memory.oom.group").read_text(), "1")
        self.assertEqual((target / "cpu.max").read_text(), "150000 100000")
        self.assertEqual((target / "pids.max").read_text(), "64")
        self.assertEqual(warnings, [])
        self.assertEqual(applied["cpu.max"], "150000 100000")

    def test_a_missing_interface_file_is_a_warning_not_a_failure(self):
        target = self.base / "hh-sandbox-plain"
        target.mkdir()
        applied, warnings = limits_module._apply(
            target, ResourceLimits(memory=1024), str(self.base)
        )
        self.assertEqual(applied, {})
        self.assertTrue(any("memory.max" in warning for warning in warnings))

    def test_place_puts_the_pid_in_the_group_before_it_can_fork(self):
        target = self._prepared()
        limiter = CgroupLimiter(target, LimitsReport("cgroup", {"memory.max": "1"}))
        self.assertTrue(limiter.needs_placement)
        limiter.place(4242)
        self.assertEqual((target / "cgroup.procs").read_text(), "4242")

    def test_kill_uses_cgroup_kill_and_close_releases_the_directory(self):
        target = self._prepared()
        limiter = CgroupLimiter(target, LimitsReport("cgroup", {}))
        limiter.kill()
        self.assertEqual((target / "cgroup.kill").read_text(), "1")
        limiter.close()
        # the fake directory holds real files, so only the bookkeeping can be
        # asserted here; `close` after `kill` on a real cgroup removes it
        self.assertIsNone(limiter.directory)

    def test_close_removes_an_empty_group(self):
        target = self.base / "hh-sandbox-empty"
        target.mkdir()
        limiter = CgroupLimiter(target, LimitsReport("cgroup", {}))
        limiter.close()
        self.assertFalse(target.exists())
        self.assertIsNone(limiter.directory)

    def test_a_group_that_will_not_go_away_is_not_an_error(self):
        target = self._prepared()
        limiter = CgroupLimiter(target, LimitsReport("cgroup", {}))
        limiter.close()  # the files a fake directory holds block rmdir; no raise
        self.assertIsNone(limiter.directory)
        for limiter in (
            limits_module.NullLimiter(),
            limits_module.RlimitLimiter(LimitsReport("rlimit")),
            limits_module.SystemdLimiter("x", (), LimitsReport("systemd")),
        ):
            with self.subTest(engine=limiter.report.engine):
                self.assertFalse(limiter.needs_placement)

    def test_systemd_scopes_get_a_fresh_unit_per_command(self):
        limiter = limits_module.SystemdLimiter(
            "abc", ("MemoryMax=1024",), LimitsReport("systemd")
        )
        first = limiter.spawn_prefix()
        second = limiter.spawn_prefix()
        self.assertIn("--unit=hh-sandbox-abc-1", first)
        self.assertIn("--unit=hh-sandbox-abc-2", second)
        self.assertIn("MemoryMax=1024", first)
        self.assertEqual(first[-1], "--")

    def test_without_a_cgroup_and_without_systemd_the_engine_says_so(self):
        with mock.patch.dict(os.environ, {"HH_SANDBOX_CGROUP": str(self.base)}):
            with mock.patch.object(limits_module, "_systemd_run_usable", return_value=False):
                limiter = create_limiter(
                    ResourceLimits(memory=1024 * 1024),
                    state_dir=str(self.base),
                    sandbox_id="test",
                )
        self.assertEqual(limiter.report.engine, "rlimit")
        self.assertFalse(limiter.report.enforced)
        self.assertTrue(any("RLIMIT_AS" in warning for warning in limiter.report.warnings))
        self.assertTrue(any("prlimit" in part for part in limiter.prefix))

    def test_the_systemd_engine_keeps_the_single_file_cap(self):
        with mock.patch.object(limits_module, "_systemd_run_usable", return_value=True):
            limiter = limits_module.create_limiter(
                ResourceLimits(memory=1024 * 1024, disk=4096),
                state_dir=str(self.base),
                sandbox_id="t",
                engine="systemd",
            )
        self.assertEqual(limiter.report.engine, "systemd")
        prefix = limiter.spawn_prefix()
        self.assertIn("--fsize=4096", prefix)  # prlimit, after the systemd-run part
        self.assertGreater(prefix.index("--fsize=4096"), prefix.index("--"))
        self.assertTrue(any(part.startswith("--unit=hh-sandbox-t") for part in prefix))

    def test_the_report_does_not_claim_rlimits_that_were_not_applied(self):
        with mock.patch.object(limits_module.shutil, "which", return_value=None):
            limiter = limits_module.create_limiter(
                ResourceLimits(memory=1024, disk=2048),
                state_dir=str(self.base),
                sandbox_id="t",
                engine="rlimit",
            )
        self.assertEqual(limiter.report.applied, {})
        self.assertEqual(limiter.prefix, ())
        self.assertTrue(any("prlimit" in warning for warning in limiter.report.warnings))

    def test_no_limits_requested_means_no_engine(self):
        limiter = create_limiter(
            ResourceLimits(), state_dir=str(self.base), sandbox_id="test", engine="auto"
        )
        self.assertEqual(limiter.report.engine, "none")
        self.assertEqual(limiter.prefix, ())

    def test_io_limits_name_the_device_when_there_is_one(self):
        limits = IoLimits(read_bps=1024, write_iops=10)
        self.assertEqual(limits.cgroup_value((8, 1)), "8:1 rbps=1024 wiops=10")
        self.assertFalse(IoLimits().any)
        self.assertTrue(limits.any)


class ToolchainTests(unittest.TestCase):
    def test_the_expression_names_attributes_and_passes_store_paths_through(self):
        expression = toolchain_module.expression(("python312", "ripgrep"))
        self.assertIn("pkgs.python312", expression)
        self.assertIn("pkgs.ripgrep", expression)
        self.assertIn("buildEnv", expression)

    def test_a_package_name_cannot_smuggle_nix_code(self):
        with self.assertRaises(toolchain_module.ToolchainError):
            toolchain_module.expression(("python312; builtins.exec \"rm -rf /\"",))

    def test_bash_and_coreutils_are_always_in_the_environment(self):
        self.assertEqual(
            toolchain_module.with_base_packages(("python312",)),
            ("python312", "bash", "coreutils"),
        )

    def test_a_given_directory_skips_nix_entirely(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "bin").mkdir()
            (Path(directory) / "bin" / "sh").symlink_to("/bin/sh")
            called = []

            def runner(argv, timeout):  # pragma: no cover - must not be called
                called.append(argv)
                raise AssertionError("nix-build should not run")

            resolved = toolchain_module.resolve(
                SandboxSpec(env_dir=directory), runner=runner, use_cache=False
            )
        self.assertEqual(resolved.origin, "given")
        self.assertEqual(called, [])
        self.assertEqual(resolved.warnings, ())

    def test_an_environment_without_a_shell_is_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "bin").mkdir()
            resolved = toolchain_module.resolve(
                SandboxSpec(env_dir=directory), use_cache=False
            )
        self.assertTrue(any("bin/sh" in warning for warning in resolved.warnings))

    def test_a_failed_build_reports_what_nix_said(self):
        def runner(argv, timeout):
            return subprocess.CompletedProcess(argv, 1, b"", b"error: undefined variable 'nope'")

        with self.assertRaises(toolchain_module.ToolchainError) as caught:
            toolchain_module.build(("nope",), runner=runner)
        self.assertIn("undefined variable", str(caught.exception))

    def test_the_resolved_store_path_is_cached_in_process(self):
        calls = []

        with tempfile.TemporaryDirectory() as fake_store:
            env_path = Path(fake_store) / "fake-env"
            (env_path / "bin").mkdir(parents=True)
            (env_path / "bin" / "sh").symlink_to("/bin/sh")

            def runner(argv, timeout):
                calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, str(env_path).encode() + b"\n", b"")

            with mock.patch.object(
                toolchain_module, "_load_cache", side_effect=lambda: toolchain_module._env_cache
            ), mock.patch.object(toolchain_module, "_save_cache"):
                toolchain_module._env_cache.clear()
                first = toolchain_module.resolve(SandboxSpec(packages=["hello"]), runner=runner)
                second = toolchain_module.resolve(SandboxSpec(packages=["hello"]), runner=runner)
        self.assertEqual(first.path, second.path)
        self.assertEqual(len(calls), 1)


def _bwrap_available() -> bool:
    """Whether a real sandbox can be started here.

    The probe has to run a real binary inside the namespace, with the loader and
    the libraries bound the way `host_toolchain` binds them, or a machine where
    everything works would still report no.
    """
    binary = shutil.which("bwrap")
    if binary is None:
        return False
    true = shutil.which("true")
    if true is None or not os.path.exists(true):
        return False
    argv = [binary, "--unshare-all", "--unshare-user", "--ro-bind", "/usr", "/usr"]
    for path in ("/lib", "/lib64"):
        if os.path.exists(path):
            argv += ["--ro-bind", os.path.realpath(path), path]
    argv += ["--", true]
    try:
        return subprocess.run(argv, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(_bwrap_available(), "bubblewrap with user namespaces is not available here")
class SandboxIntegrationTests(unittest.TestCase):
    """Real sandboxes, built from the host's own binaries so no Nix is needed."""

    @classmethod
    def setUpClass(cls):
        cls.tools = toolchain_module.host_toolchain()
        cls.state = tempfile.mkdtemp(prefix="hh-sandbox-tests-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tools.path, ignore_errors=True)
        shutil.rmtree(cls.state, ignore_errors=True)

    def create(self, **kwargs):
        kwargs.setdefault("env_dir", self.tools.path)
        kwargs.setdefault("mounts", list(self.tools.mounts))
        kwargs.setdefault("state_root", self.state)
        sandbox = Sandbox.create(**kwargs)
        self.addCleanup(sandbox.destroy)
        return sandbox

    def test_a_command_runs_and_reports_its_status(self):
        sandbox = self.create(writable=["/work"])
        result = sandbox.exec(["/bin/sh", "-c", "echo out; echo err >&2; exit 3"])
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(result.stdout, "out\n")
        self.assertEqual(result.stderr, "err\n")
        self.assertFalse(result.ok)
        self.assertEqual(result.duration > 0, True)

    def test_the_root_holds_only_what_the_spec_mounted(self):
        sandbox = self.create(writable=["/work"])
        result = sandbox.exec(["/bin/sh", "-c", "ls -A /"])
        entries = set(result.stdout.split())
        for expected in ("work", "tmp", "proc", "dev", "usr"):
            self.assertIn(expected, entries)
        # the only home is the sandbox's own
        homes = sandbox.exec(["/bin/sh", "-c", "ls -A /home"])
        self.assertEqual(homes.stdout.split(), ["user"])
        private = sandbox.exec(["/bin/sh", "-c", "test -e /etc/shadow && echo visible || echo absent"])
        self.assertEqual(private.stdout.strip(), "absent")
        home = sandbox.exec(["/bin/sh", "-c", f"test -e {os.path.expanduser('~')} && echo visible || echo absent"])
        self.assertEqual(home.stdout.strip(), "absent")

    def test_the_host_is_read_only_and_the_work_directory_is_not(self):
        sandbox = self.create(writable=["/work"])
        blocked = sandbox.exec(["/bin/sh", "-c", "touch /usr/hh-probe 2>&1 || echo refused"])
        self.assertIn("refused", blocked.stdout)
        written = sandbox.exec(["/bin/sh", "-c", "echo data > /work/file.txt && cat /work/file.txt"])
        self.assertEqual(written.stdout.strip(), "data")

    def test_files_can_be_handed_in_and_read_back(self):
        sandbox = self.create(
            writable=["/work"],
            files={"/input/data.json": '{"n": 1}', "/work/main.sh": "#!/bin/sh\necho script\n"},
        )
        self.assertEqual(sandbox.exec(["cat", "/input/data.json"]).stdout, '{"n": 1}')
        # a shebang makes the staged file executable
        self.assertEqual(sandbox.exec(["/work/main.sh"]).stdout.strip(), "script")
        blocked = sandbox.exec(["/bin/sh", "-c", "echo x >> /input/data.json 2>&1 || echo refused"])
        self.assertIn("refused", blocked.stdout)
        sandbox.put_file("/work/made.txt", "written by the harness\n")
        self.assertEqual(sandbox.get_file("/work/made.txt"), b"written by the harness\n")

    def test_scratch_space_belongs_to_one_command(self):
        # every command gets fresh namespaces, so a tmpfs is created, used and
        # thrown away within that command; the harness is told so plainly
        # rather than handed a file that is not there
        sandbox = self.create(writable=["/work", Mount.tmpfs("/scratch", "8M")])
        wrote = sandbox.exec(["/bin/sh", "-c", "mkdir -p /scratch/deep && echo hi > /scratch/deep/f"])
        self.assertTrue(wrote.ok)
        later = sandbox.exec(["/bin/sh", "-c", "cat /scratch/deep/f 2>&1 || echo not-there"])
        self.assertIn("not-there", later.stdout)
        with self.assertRaises(SandboxError) as caught:
            sandbox.put_file("/scratch/file.txt", "x")
        self.assertIn("tmpfs", str(caught.exception))
        with self.assertRaises(SandboxError):
            sandbox.get_file("/scratch/file.txt")

    def test_the_network_namespace_has_only_loopback(self):
        sandbox = self.create(writable=["/work"])
        result = sandbox.exec(["/bin/sh", "-c", "cat /proc/net/dev"])
        self.assertIn("lo:", result.stdout)
        self.assertEqual(
            [line.split(":")[0].strip() for line in result.stdout.splitlines() if ":" in line],
            ["lo"],
        )
        dial = sandbox.exec(
            ["/bin/sh", "-c", "timeout 3 sh -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null || echo unreachable"]
        )
        self.assertIn("unreachable", dial.stdout)

    def test_the_pid_namespace_is_private(self):
        sandbox = self.create(writable=["/work"])
        result = sandbox.exec(["/bin/sh", "-c", "cat /proc/1/cmdline"])
        self.assertIn("bwrap", result.stdout)

    def test_a_command_can_be_fed_standard_input(self):
        sandbox = self.create(writable=["/work"])
        result = sandbox.exec(["cat"], stdin="piped in\n")
        self.assertEqual(result.stdout, "piped in\n")
        # with no input the sandbox must not inherit the harness's stdin
        empty = sandbox.exec(["cat"])
        self.assertEqual((empty.exit_code, empty.stdout), (0, ""))

    def test_the_working_directory_and_environment_can_be_given_per_command(self):
        sandbox = self.create(writable=["/work", Mount.tmpfs("/scratch", "4M")])
        result = sandbox.exec(["pwd"], cwd="/scratch", env={"MARK": "here"})
        self.assertEqual(result.stdout.strip(), "/scratch")
        self.assertEqual(
            sandbox.exec(["sh", "-c", "echo $MARK"], cwd="/tmp", env={"MARK": "here"}).stdout.strip(),
            "here",
        )
        self.assertEqual(sandbox.exec(["pwd"]).stdout.strip(), "/work")

    def test_output_beyond_the_cap_is_dropped_and_reported(self):
        sandbox = self.create(writable=["/work"])
        result = sandbox.exec(
            ["sh", "-c", "dd if=/dev/zero bs=1k count=64 2>/dev/null | tr '\\0' 'x'"],
            max_output=1024,
        )
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.stdout), 1024)

    def test_the_watchdog_kills_a_command_that_overruns(self):
        sandbox = self.create(writable=["/work"])
        result = sandbox.exec(["/bin/sh", "-c", "sleep 30"], timeout=1.0)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, 124)
        self.assertLess(result.duration, 10)

    def test_background_work_does_not_survive_its_command(self):
        sandbox = self.create(writable=["/work"])
        sandbox.exec(["/bin/sh", "-c", "sleep 30 & echo started"])
        left = sandbox.exec(["/bin/sh", "-c", "cat /proc/[0-9]*/comm 2>/dev/null | grep -c '^sleep$' || true"])
        self.assertEqual(left.stdout.strip(), "0")

    def test_the_syscall_filter_refuses_a_denied_call(self):
        sandbox = self.create(writable=["/work"])
        have_python = sandbox.exec(["/bin/sh", "-c", "command -v python3 >/dev/null && echo yes"])
        if have_python.stdout.strip() != "yes":
            self.skipTest("no python3 on the host to make a raw syscall")
        number = seccomp.syscall_number("keyctl")
        script = (
            "import ctypes; libc = ctypes.CDLL(None, use_errno=True); "
            f"libc.syscall({number}, 0, -2, 1, 0, 0); print(ctypes.get_errno())"
        )
        filtered = sandbox.exec(["python3", "-c", script])
        self.assertEqual(filtered.stdout.strip(), "1")  # EPERM
        unfiltered = self.create(writable=["/work"], syscalls=SyscallPolicy(enabled=False))
        self.assertNotEqual(unfiltered.exec(["python3", "-c", script]).stdout.strip(), "1")

    def test_a_disk_budget_stops_the_next_command(self):
        # two files that each fit under RLIMIT_FSIZE but together exceed the budget:
        # the budget is on the tree, not on any single file
        sandbox = self.create(writable=["/work"], resources={"disk": "32K", "timeout": 30})
        sandbox.exec(
            ["/bin/sh", "-c",
             "dd if=/dev/urandom of=/work/one bs=1k count=24 2>/dev/null; "
             "dd if=/dev/urandom of=/work/two bs=1k count=24 2>/dev/null; true"]
        )
        self.assertGreater(sandbox.disk_used, 32 * 1024)
        with self.assertRaises(SandboxDiskExceeded):
            sandbox.exec(["/bin/sh", "-c", "true"])

    def test_a_scratch_mount_refuses_to_grow_past_its_size(self):
        sandbox = self.create(writable=["/work", Mount.tmpfs("/scratch", "1M")])
        result = sandbox.exec(
            ["/bin/sh", "-c", "dd if=/dev/zero of=/scratch/big bs=1k count=4096 2>&1 || echo refused"]
        )
        self.assertIn("refused", result.stdout)

    def test_destroying_a_sandbox_removes_its_state(self):
        sandbox = self.create(writable=["/work"])
        sandbox.exec(["/bin/sh", "-c", "echo x > /work/file"])
        root = sandbox.layout.root
        sandbox.destroy()
        self.assertFalse(root.exists())
        with self.assertRaises(Exception):
            sandbox.exec(["/bin/sh", "-c", "true"])

    def test_a_domain_allow_list_is_refused_rather_than_faked(self):
        with self.assertRaises(NotImplementedError):
            self.create(writable=["/work"], network={"allow": ["pypi.org"]})

    def test_the_described_sandbox_matches_what_was_asked_for(self):
        sandbox = self.create(
            writable=["/work"],
            resources={"memory": "64M", "pids": 16, "timeout": 5},
            env={"EXTRA": "value"},
        )
        described = sandbox.describe()
        self.assertEqual(described["network"], "none")
        self.assertIn("/work", described["writable"])
        self.assertEqual(sandbox.exec(["/bin/sh", "-c", "echo $EXTRA"]).stdout.strip(), "value")
        self.assertTrue(sandbox.exec(["/bin/sh", "-c", "echo $HOME"]).stdout.strip())

    # -- sessions -----------------------------------------------------------

    def open_session(self, sandbox, command=("bash", "--norc", "-i")):
        session = sandbox.open_session(command)
        self.addCleanup(session.close)
        return session

    def read_until_line(self, session, expected, timeout=15.0):
        """The session's output, up to the first line that is exactly `expected`.

        A pty echoes what is typed, so "the marker appeared somewhere" is not
        enough — the echo of the command that prints it carries the marker too.
        Whole-line equality is what tells output apart from the echo.
        """
        text = ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            chunk = session.read(timeout=0.2)
            if chunk:
                text += chunk.decode("utf-8", "replace")
            if any(line.strip() == expected for line in text.splitlines()):
                return text
            if session.poll() is not None and not chunk:
                break
        self.fail(f"the session never printed a line {expected!r}; it said:\n{text}")

    def test_a_session_gives_the_shell_a_terminal_and_job_control(self):
        # the pty is the whole point of a session: without a controlling
        # terminal a shell has neither line editing nor job control, and
        # --new-session would detach it from the pty it was given
        sandbox = self.create(writable=["/work"])
        session = self.open_session(sandbox)
        session.write("[ -t 0 ] && echo TTY=yes || echo TTY=no\n")
        self.read_until_line(session, "TTY=yes")
        session.write('case "$-" in *m*) echo JOB=on ;; *) echo JOB=off ;; esac\n')
        self.read_until_line(session, "JOB=on")

    def test_a_session_keeps_state_from_one_line_to_the_next(self):
        # what exec cannot do: one process holds the namespaces for the whole
        # session, so `cd`, exported variables and a tmpfs all survive
        sandbox = self.create(writable=["/work", Mount.tmpfs("/scratch", "8M")])
        session = self.open_session(sandbox)
        session.write("cd /scratch && export WHERE=carried\n")
        session.write('echo "$PWD $WHERE" > /scratch/where\n')
        session.write("cat /scratch/where\n")
        self.read_until_line(session, "/scratch carried")

    def test_a_session_reports_the_status_the_shell_exits_with(self):
        sandbox = self.create(writable=["/work"])
        session = self.open_session(sandbox)
        session.write("exit 7\n")
        self.assertEqual(session.wait(timeout=15), 7)

    def test_exec_refuses_while_a_session_is_open(self):
        sandbox = self.create(writable=["/work"])
        session = self.open_session(sandbox)
        with self.assertRaises(SandboxError) as caught:
            sandbox.exec(["/bin/sh", "-c", "true"])
        self.assertIn("session", str(caught.exception))
        # closing it hands the sandbox back
        session.close()
        self.assertTrue(sandbox.exec(["/bin/sh", "-c", "echo again"]).ok)

    def test_destroying_a_sandbox_ends_its_session(self):
        sandbox = self.create(writable=["/work"])
        session = self.open_session(sandbox)
        self.assertIsNone(session.poll())
        sandbox.destroy()
        self.assertIsNotNone(session.poll())

    def test_a_session_starts_without_a_pid_handshake_too(self):
        # the rlimit engine has no cgroup to place a pid in; the pty setup must
        # not depend on which limiter is in use
        sandbox = self.create(writable=["/work"], limits_engine="rlimit")
        session = self.open_session(sandbox)
        session.write("echo STARTED\n")
        self.read_until_line(session, "STARTED")


class SelfTestProbeTests(unittest.TestCase):
    """The self-test's own machinery: the parsers and the probe scripts.

    These are the parts that told the wrong story on a real machine once — a
    probe that used `awk`, a fork storm that hung instead of counting, an
    allocation that never touched memory — so they are worth a cheap test each.
    """

    def test_cgroup_counters_are_read_from_their_lines(self):
        text = "low 0\nmax 3\noom 1\noom_kill 7\n"
        self.assertEqual(main._counter(text, "oom_kill"), 7)
        self.assertEqual(main._counter("max 5\n", "max"), 5)
        self.assertIsNone(main._counter(text, "high"))
        self.assertIsNone(main._counter(None, "max"))

    def test_the_probe_scripts_are_valid_python(self):
        compile(main.FORK_STORM, "<fork storm>", "exec")
        compile(main.MEMORY_TOUCH.format(bytes=1024), "<memory touch>", "exec")

    def test_the_interface_list_parses_out_of_proc_net_dev(self):
        # the same parsing the network probe does, on a real interface list
        text = Path("/proc/self/net/dev").read_text()
        names = [
            line.split(":")[0].strip() for line in text.splitlines()[2:] if ":" in line
        ]
        self.assertIn("lo", names)

    def test_the_fork_storm_stops_at_its_cap(self):
        self.assertLessEqual(main.FORK_STORM_CAP, 256)
        self.assertIn(str(main.FORK_STORM_CAP), main.FORK_STORM)


class SandboxContractTests(unittest.TestCase):
    """Anything that does not need a working bubblewrap to be true."""

    def test_a_path_that_climbs_out_of_its_mount_is_refused(self):
        # an agent-supplied path must not be able to name another mount's
        # backing store, or the state directory itself
        sandbox = self.create_without_bwrap()
        for path in ("/work/../elsewhere/file", "/../escape", "/work/./x", "relative"):
            with self.subTest(path=path):
                with self.assertRaises(SpecError):
                    sandbox.put_file(path, "x")
                with self.assertRaises(SpecError):
                    sandbox.get_file(path)

    def test_files_cannot_be_moved_after_destroy(self):
        sandbox = self.create_without_bwrap()
        root = sandbox.layout.root
        sandbox.destroy()
        with self.assertRaises(SandboxError):
            sandbox.put_file("/anything", "x")
        with self.assertRaises(SandboxError):
            sandbox.get_file("/anything")
        self.assertFalse(root.exists())

    def test_a_spec_and_keyword_arguments_together_are_refused(self):
        with self.assertRaises(SpecError):
            Sandbox.create(SandboxSpec(packages=["bash"]), packages=["python312"])

    def test_a_path_outside_every_writable_mount_is_named_in_the_error(self):
        sandbox = self.create_without_bwrap()
        with self.assertRaises(SandboxError) as caught:
            sandbox.put_file("/nope/file", "x")
        self.assertIn("/nope/file", str(caught.exception))

    def test_destroying_twice_is_harmless(self):
        sandbox = self.create_without_bwrap()
        sandbox.destroy()
        sandbox.destroy()

    def create_without_bwrap(self) -> Sandbox:
        spec = SandboxSpec(writable=[], resources=ResourceLimits())
        layout = mock.MagicMock()
        layout.root = Path(tempfile.mkdtemp(prefix="hh-contract-"))
        self.addCleanup(shutil.rmtree, layout.root, True)
        sandbox = Sandbox(
            spec=spec,
            toolchain=toolchain_module.Toolchain(path="/env", origin="given"),
            layout=layout,
            mounts=(),
            file_binds=(),
            limiter=limits_module.NullLimiter(),
            program=None,
            identity="contract",
        )
        return sandbox


if __name__ == "__main__":
    unittest.main()
