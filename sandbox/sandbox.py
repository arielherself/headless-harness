"""A live sandbox: create it, run commands in it, move files in and out, destroy it.

The shape is the one the design calls for — a spec goes in, an object with
`exec`, `put_file`, `get_file` and `destroy` comes out — with one deliberate
difference: nothing is spawned until a command is run. Creating a sandbox means
building a toolchain, a directory layout and a cgroup; running a command means
starting bubblewrap with the namespaces, mounts and seccomp filter that the spec
describes. A sandbox that is created and never used costs a directory.

State lives where the mounts say it does. Writable paths are directories under
the sandbox's state root, so the harness can put a file in, run something that
reads it, and read a file back out without a second mechanism — those directories
are also what `ResourceLimits.disk` measures. Mounts declared as `tmpfs` are the
opposite: invisible from outside, gone on destroy, and the only writable places
with a hard size cap.

Nothing here is asynchronous, on purpose: the harness runs tool hooks
synchronously on threads of their own, and blocking is what they want.

The toolchain is not fixed for the sandbox's lifetime. `add_packages` and
`remove_packages` ask Nix for another `buildEnv` store path and swap it in; the
next command mounts it, and a store path already built for a package list is
reused, so going back to a list costs nothing.
"""

from __future__ import annotations

import os
import re
import select
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import bubblewrap
from . import limits as limits_module
from . import seccomp as seccomp_module
from . import toolchain as toolchain_module
from .bubblewrap import ResolvedMount
from .session import Session
from .spec import (
    DEFAULT_HOME,
    DEFAULT_TMP,
    DEFAULT_WORK,
    Mount,
    NetworkPolicy,
    SandboxSpec,
    SpecError,
    SyscallPolicy,
    check_sandbox_path,
    network_from,
    resources_from,
)
from .toolchain import Toolchain

# Output beyond this is dropped and the result says so; a command that prints
# gigabytes should not become a gigabyte in the harness's memory.
DEFAULT_MAX_OUTPUT = 1024 * 1024
# Reading a file out of a tmpfs mount goes through a command, so the cap has to
# be generous enough for real files while still bounded.
FILE_MAX_OUTPUT = 64 * 1024 * 1024
# How long bubblewrap gets to set up namespaces and report its child PID.
SETUP_TIMEOUT = 30.0
# Where sandbox state is created unless the operator says otherwise.
DEFAULT_STATE_ROOT = "/tmp/headless-harness-sandboxes"

# A package change cannot reach into a command that is already running: its
# environment was fixed when bubblewrap started.
_SESSION_ENV_WARNING = (
    "a session is open, and it keeps the environment it started with; "
    "the change applies to the next command"
)

_UNSET = object()


class SandboxError(RuntimeError):
    """The sandbox could not be created, or a command could not be run."""


class SandboxDiskExceeded(SandboxError):
    """The sandbox has used its disk budget; nothing else will run."""


@dataclass(frozen=True)
class ExecResult:
    """What a command did.

    `exit_code` follows the shell convention: 0 for success, 128+N when a
    process died from signal N, and 124 when the watchdog fired. `truncated`
    says output was dropped, which is worth surfacing rather than silently
    returning half an answer.
    """

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    stdout_bytes: bytes = field(repr=False, default=b"")
    stderr_bytes: bytes = field(repr=False, default=b"")
    duration: float = 0.0
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self, limit: int = 2000) -> str:
        """A short description, which is what a tool result usually wants."""
        if self.timed_out:
            return f"timed out after {self.duration:.1f}s"
        text = self.stdout.strip() or self.stderr.strip()
        if len(text) > limit:
            text = text[:limit] + "..."
        return f"exit {self.exit_code}" + (f": {text}" if text else "")


@dataclass(frozen=True)
class Layout:
    """Where one sandbox keeps its host-side state."""

    root: Path
    etc: Path
    ro: Path
    files: Path

    @staticmethod
    def create(base: str | None = None) -> "Layout":
        base_path = Path(
            base or os.environ.get("HH_SANDBOX_ROOT") or DEFAULT_STATE_ROOT
        )
        # a predictable path is convenient and, on a shared machine, a thing to
        # check: state must land in a directory this user owns
        if base_path.is_symlink():
            raise SandboxError(f"the sandbox root {base_path} is a symlink")
        try:
            os.makedirs(base_path, mode=0o700, exist_ok=True)
        except OSError as exc:
            raise SandboxError(f"could not create the sandbox root {base_path}: {exc}") from exc
        if base_path.stat().st_uid != os.getuid():
            raise SandboxError(f"the sandbox root {base_path} is not owned by this user")
        root = Path(tempfile.mkdtemp(prefix="sandbox-", dir=base_path))
        os.chmod(root, 0o700)
        layout = Layout(root=root, etc=root / "etc", ro=root / "ro", files=root / "fs")
        for directory in (layout.etc, layout.ro, layout.files):
            directory.mkdir()
        return layout

    def host_dir(self, sandbox_path: str) -> Path:
        """The host directory backing a writable mount inside the sandbox."""
        return self.files / sandbox_path.lstrip("/")

    def ro_file(self, sandbox_path: str) -> Path:
        return self.ro / sandbox_path.lstrip("/")


ETC_PASSWD = """\
root:x:0:0:root:/root:/bin/sh
user:x:{uid}:{gid}:sandbox user:{home}:/bin/sh
"""
ETC_GROUP = """\
root:x:0:
user:x:{gid}:
"""
ETC_HOSTS = """\
127.0.0.1 localhost
::1 localhost
"""


class _StreamReader(threading.Thread):
    """Drain a pipe, keeping at most `cap` bytes, so the child never blocks."""

    def __init__(self, stream: Any, cap: int) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.cap = cap
        self.data = b""
        self.truncated = False

    def run(self) -> None:
        chunks: list[bytes] = []
        total = 0
        try:
            while True:
                try:
                    chunk = self.stream.read(65536)
                except (OSError, ValueError):
                    break
                if not chunk:
                    break
                if total < self.cap:
                    keep = chunk[: self.cap - total]
                    chunks.append(keep)
                    total += len(keep)
                    if len(keep) != len(chunk):
                        self.truncated = True
                else:
                    self.truncated = True
        finally:
            self.data = b"".join(chunks)
            try:
                self.stream.close()
            except (OSError, ValueError):
                pass


class Sandbox:
    """A created sandbox. Use `Sandbox.create`.

    One command runs at a time: a lock serializes `exec`, `put_file` and the
    package operations so a threaded harness cannot have two bubblewrap
    processes racing over the same writable mounts, or a command starting while
    its environment is being replaced. Nothing is started at creation time, so a
    sandbox that is created and never used costs a directory.
    """

    def __init__(
        self,
        *,
        spec: SandboxSpec,
        toolchain: Toolchain,
        layout: Layout,
        mounts: tuple[ResolvedMount, ...],
        file_binds: tuple[tuple[str, str], ...],
        limiter: limits_module.Limiter,
        program: seccomp_module.Program | None,
        identity: str,
    ) -> None:
        self.spec = spec
        self.toolchain = toolchain
        self.layout = layout
        self.mounts = mounts
        self.limiter = limiter
        self.program = program
        self.identity = identity
        self._toolchain_warnings = list(toolchain.warnings)
        self.warnings: list[str] = list(toolchain.warnings)
        self.disk_used: int = 0
        self._file_binds = file_binds
        self._lock = threading.Lock()
        self._closed = False
        self._session: Session | None = None
        self._budgeted = tuple(
            sorted(
                {
                    mount.source
                    for mount in mounts
                    if mount.kind == "rw-bind"
                    and mount.source is not None
                    and str(layout.root) in mount.source
                }
            )
        )

    # -- creation -----------------------------------------------------------

    @classmethod
    def create(
        cls,
        spec: SandboxSpec | None = None,
        /,
        *,
        packages: Sequence[str] = (),
        files: Mapping[str, bytes | str] | None = None,
        writable: Sequence[str | Mount] | None = None,
        mounts: Sequence[Mount] = (),
        network: Any = None,
        resources: Any = None,
        syscalls: SyscallPolicy | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        nixpkgs: str | None = None,
        env_dir: str | None = None,
        limits_engine: str = "auto",
        state_root: str | None = None,
    ) -> "Sandbox":
        """Build a sandbox from a spec, or from the keyword arguments of one."""
        if spec is not None:
            provided = [
                name
                for name, value in (
                    ("packages", packages),
                    ("files", files),
                    ("writable", writable),
                    ("mounts", mounts),
                    ("network", network),
                    ("resources", resources),
                    ("syscalls", syscalls),
                    ("env", env),
                    ("cwd", cwd),
                    ("nixpkgs", nixpkgs),
                    ("env_dir", env_dir),
                )
                if value not in (None, ())
            ]
            if provided:
                raise SpecError(
                    "pass a spec or the keyword arguments of one, not both: "
                    + ", ".join(provided)
                )
        else:
            spec = SandboxSpec(
                packages=tuple(packages),
                files=dict(files or {}),
                writable=(
                    tuple(writable) if writable is not None else (DEFAULT_WORK,)
                ),
                mounts=tuple(mounts),
                network=network_from(network) if network is not None else NetworkPolicy.none(),
                resources=resources_from(resources),
                syscalls=syscalls or SyscallPolicy(),
                env=dict(env or {}),
                cwd=cwd or DEFAULT_WORK,
                nixpkgs=nixpkgs,
                env_dir=env_dir,
            )
        cls._refuse_unimplemented(spec)
        toolchain = toolchain_module.resolve(spec)
        layout = Layout.create(state_root)
        try:
            identity = layout.root.name.removeprefix("sandbox-")[:12]
            planned = _plan_mounts(spec, layout)
            _check_cwd(spec, planned)
            file_binds = _stage(spec, layout, planned)
            # the filter is built before the limiter so that a spec the filter
            # cannot express (an architecture with no syscall table, say) does
            # not leave a cgroup behind
            program = seccomp_module.build(spec.syscalls)
            limiter = limits_module.create_limiter(
                spec.resources,
                state_dir=str(layout.root),
                sandbox_id=identity,
                engine=limits_engine,
            )
        except BaseException:
            shutil.rmtree(layout.root, ignore_errors=True)
            raise
        sandbox = cls(
            spec=spec,
            toolchain=toolchain,
            layout=layout,
            mounts=planned,
            file_binds=file_binds,
            limiter=limiter,
            program=program,
            identity=identity,
        )
        if sandbox.spec.resources.memory and not limiter.report.enforced:
            sandbox.warnings.append(
                "memory is not enforced by the kernel here: " + limiter.report.describe()
            )
        sandbox._warn_about_external_writes()
        return sandbox

    @staticmethod
    def _refuse_unimplemented(spec: SandboxSpec) -> None:
        if spec.network.mode == "allow":
            raise NotImplementedError(
                "a domain allow-list needs a CONNECT proxy outside the network "
                "namespace to enforce it; until that exists, use NetworkPolicy.none() "
                "or NetworkPolicy.host(). The policy shape is already here so callers "
                "can declare intent: " + ", ".join(spec.network.allow)
            )
        if spec.packages and spec.env_dir:
            raise SpecError("pass either packages= or env_dir=, not both")

    # -- changing the environment -------------------------------------------

    @property
    def packages(self) -> tuple[str, ...]:
        """The packages the next command will have, base packages included."""
        return self.toolchain.packages

    def add_packages(self, *packages: str) -> tuple[str, ...]:
        """Add Nix packages to a sandbox that already exists.

        Each call asks Nix for one `buildEnv` store path holding the new list.
        The change takes effect on the next command, because every command
        mounts the current store path, and the resulting package list is
        returned. A package that is already in the environment is ignored, and
        a package Nix cannot build leaves the sandbox exactly as it was.
        """
        names = _package_names(packages)
        if not names:
            return self.packages
        with self._lock:
            self._check_open()
            self._require_nix_environment()
            missing = [name for name in names if name not in self.toolchain.packages]
            if not missing:
                return self.packages
            self._rebuild_toolchain(
                tuple(dict.fromkeys((*self.spec.packages, *missing)))
            )
            return self.packages

    def remove_packages(self, *packages: str) -> tuple[str, ...]:
        """Remove Nix packages named at creation or added since.

        `bash` and `coreutils` cannot be removed — a Nix environment always has
        them, because `/bin/sh` and `#!/usr/bin/env` have to resolve — and a
        name the sandbox was never asked for is refused rather than silently
        ignored, so a typo does not look like a removal. As with `add_packages`,
        the new environment is built first and takes effect on the next command.
        """
        names = _package_names(packages)
        if not names:
            return self.packages
        with self._lock:
            self._check_open()
            self._require_nix_environment()
            requested = tuple(self.spec.packages)
            # a list made only of store paths is taken literally: `with_base_packages`
            # adds nothing to it, so what it names is all there is
            literal_only = bool(requested) and all(name.startswith("/") for name in requested)
            if not literal_only:
                base = [name for name in names if name in toolchain_module.BASE_PACKAGES]
                if base:
                    raise SandboxError(
                        ", ".join(base)
                        + " cannot be removed: bash and coreutils are always in a Nix "
                        "environment, because `/bin/sh` and `#!/usr/bin/env` have to "
                        "resolve"
                    )
            absent = [name for name in names if name not in requested]
            if absent:
                raise SandboxError(
                    "not in this sandbox's package list: "
                    + ", ".join(absent)
                    + (
                        f" (it has {', '.join(requested)})"
                        if requested
                        else " (it has only the default bash and coreutils)"
                    )
                )
            self._rebuild_toolchain(
                tuple(name for name in requested if name not in names)
            )
            return self.packages

    def _require_nix_environment(self) -> None:
        """Refuse package changes where there is no Nix environment to change."""
        if self.spec.env_dir or self.toolchain.origin != "nix":
            raise SandboxError(
                "this sandbox's environment is not built by Nix "
                f"({self.toolchain.describe()}), so its packages cannot change; "
                "create the sandbox with packages= instead of env_dir="
            )

    def _rebuild_toolchain(self, packages: tuple[str, ...]) -> None:
        """Build the environment for a new package list, then adopt it.

        The build runs before anything is changed, so a package that does not
        exist leaves the old environment in place. A session that is already
        open keeps the environment it started with — its `PATH` was fixed when
        bubblewrap started — which is said out loud in `warnings` because the
        change is otherwise invisible from inside it.
        """
        self._require_nix_environment()
        rebuilt = toolchain_module.resolve_packages(
            packages, nixpkgs=self.spec.nixpkgs
        )
        previous_warnings, self._toolchain_warnings = (
            self._toolchain_warnings,
            list(rebuilt.warnings),
        )
        self.toolchain = rebuilt
        self.spec = replace(self.spec, packages=packages)
        for warning in previous_warnings:
            if warning in self.warnings:
                self.warnings.remove(warning)
        for warning in self._toolchain_warnings:
            if warning not in self.warnings:
                self.warnings.append(warning)
        if self._live_session() is not None and _SESSION_ENV_WARNING not in self.warnings:
            self.warnings.append(_SESSION_ENV_WARNING)

    # -- running things -----------------------------------------------------

    def exec(
        self,
        command: Sequence[str],
        *,
        timeout: float | None | object = _UNSET,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        stdin: bytes | str | None = None,
        max_output: int = DEFAULT_MAX_OUTPUT,
    ) -> ExecResult:
        """Run one command in the sandbox and wait for it.

        Each call gets fresh namespaces: background processes from an earlier
        call do not survive into this one, and files only persist where the
        mounts say so. `timeout=_UNSET` uses the spec's; `timeout=None` waits
        forever, which is occasionally what a caller wants and never what a
        harness wants.
        """
        self._check_open()
        command = tuple(str(part) for part in command)
        if not command:
            raise SandboxError("no command to run")
        budget = self.spec.resources.timeout if timeout is _UNSET else timeout  # type: ignore[assignment]
        workdir = cwd or self.spec.cwd

        with self._lock:
            # both of these are re-checked under the lock: a destroy() may have
            # won the race while this call was on its way in
            self._check_open()
            # the environment is built under the lock too, so a package added or
            # removed on another thread is either in this command's `PATH` or
            # not in it at all, never half of each
            environment = bubblewrap.default_environment(
                self.spec, self.toolchain, _home_dir(self.mounts), extra=env or {}
            )
            running = self._live_session()
            if running is not None:
                raise SandboxError(
                    "this sandbox has a session open "
                    f"(`{' '.join(running.command)}`), and its namespaces belong to "
                    "that command until it ends: close the session, or use a second "
                    "sandbox"
                )
            self._refuse_when_over_budget()
            result = self._spawn(
                command,
                environment=environment,
                cwd=workdir,
                stdin=stdin,
                timeout=budget,  # type: ignore[arg-type]
                max_output=max_output,
            )
            self._measure_disk()
        return result

    def open_session(
        self,
        command: Sequence[str] = ("bash",),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> Session:
        """Start a long-lived command on a pty, in one bubblewrap process.

        `exec` runs a command and waits for it, with fresh namespaces every
        time; a session runs *until the command ends*, with the namespaces held
        open by it. The command — an interactive shell, usually — is then the
        parent of everything done in it, so the working directory, the
        environment and background jobs persist from one line to the next, and
        the mounts the spec declared (a `tmpfs` included) stay put for the
        whole session.

        The pty is what makes a shell interactive: line editing, Ctrl-C, job
        control and window resizing all need a controlling terminal, and the
        pty is the sandbox's own rather than the harness's. It also means the
        session's output is a terminal stream — it echoes what is typed and
        translates newlines — and that only one session can be open: `exec`
        refuses while one is, because the namespaces belong to its process.
        `resources.timeout` does not apply to a session; it ends when the
        command does, or with `Session.close()`.
        """
        self._check_open()
        command = tuple(str(part) for part in command)
        if not command:
            raise SandboxError("no command to run")
        with self._lock:
            self._check_open()
            self._refuse_when_over_budget()
            running = self._live_session()
            if running is not None:
                raise SandboxError(
                    "this sandbox already has a session open "
                    f"(`{' '.join(running.command)}`): its namespaces belong to that "
                    "command until it ends — close the session, or create another "
                    "sandbox"
                )
            session = self._start_session(
                command, env=dict(env or {}), cwd=cwd or self.spec.cwd
            )
            self._session = session
            return session

    def _spawn(
        self,
        command: tuple[str, ...],
        *,
        environment: Mapping[str, str],
        cwd: str,
        stdin: bytes | str | None,
        timeout: float | None,
        max_output: int,
    ) -> ExecResult:
        seccomp_fd = _seccomp_fd(self.program)
        info_read = info_write = block_read = block_write = None
        try:
            if self.limiter.needs_placement:
                info_read, info_write = os.pipe()
                block_read, block_write = os.pipe()
            argv = list(self.limiter.spawn_prefix()) + bubblewrap.build_argv(
                spec=self.spec,
                mounts=self.mounts,
                file_binds=self._file_binds,
                toolchain=self.toolchain,
                env=environment,
                cwd=cwd,
                command=command,
                seccomp_fd=seccomp_fd,
                info_fd=info_write,
                block_fd=block_read,
                hostname="sandbox",
            )
            pass_fds = tuple(
                fd for fd in (seccomp_fd, info_write, block_read) if fd is not None
            )
            payload = stdin.encode() if isinstance(stdin, str) else stdin
            started = time.monotonic()
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    pass_fds=pass_fds,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                return ExecResult(command, 127, "", f"could not start bwrap: {exc}")
            # the child holds its own copies now; closing ours here is what
            # makes a closed pipe mean "bwrap died" rather than "nobody wrote"
            seccomp_fd = _close(seccomp_fd)
            info_write = _close(info_write)
            block_read = _close(block_read)

            placed, failure = self._place(process, info_read, block_write)
            if not placed:
                # kill it before closing the block pipe: bubblewrap treats EOF
                # on that fd as permission to start, so closing first would let
                # an unplaced sandbox run exactly what was refused
                self._terminate(process)
                info_read = _close(info_read)
                block_write = _close(block_write)
                return ExecResult(command, 126, "", failure or "sandbox setup failed")
            info_read = _close(info_read)
            block_write = _close(block_write)

            readers = [
                _StreamReader(process.stdout, max_output),
                _StreamReader(process.stderr, max_output),
            ]
            for reader in readers:
                reader.start()
            writer = None
            if payload is not None and process.stdin is not None:
                writer = threading.Thread(
                    target=_feed, args=(process.stdin, payload), daemon=True
                )
                writer.start()

            timed_out = False
            try:
                exit_code = process.wait(timeout=timeout)  # type: ignore[arg-type]
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate(process)
                exit_code = 124
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                for reader in readers:
                    reader.join(timeout=10)
                if writer is not None:
                    writer.join(timeout=5)
            return ExecResult(
                command=command,
                exit_code=exit_code,
                stdout=readers[0].data.decode("utf-8", "replace"),
                stderr=readers[1].data.decode("utf-8", "replace"),
                stdout_bytes=readers[0].data,
                stderr_bytes=readers[1].data,
                duration=time.monotonic() - started,
                timed_out=timed_out,
                truncated=readers[0].truncated or readers[1].truncated,
            )
        finally:
            for fd in (seccomp_fd, info_read, info_write, block_read, block_write):
                _close(fd)

    def _place(
        self, process: subprocess.Popen, info_read: int | None, block_write: int | None
    ) -> tuple[bool, str | None]:
        """Put the sandbox process where its limits are, then let it start.

        bubblewrap reports the real PID of the process that is about to become
        the sandbox and blocks it there, so the cgroup covers everything it will
        ever fork. If the handshake does not complete the command is refused:
        running it unlimited would quietly undo the reason the handshake exists.
        """
        if info_read is None or block_write is None:
            return True, None
        pid = _read_child_pid(info_read, SETUP_TIMEOUT)
        if pid is None:
            # nothing was placed, so nothing may run: the caller kills the
            # process before the block pipe is closed (see `_spawn`)
            detail = _drain(process, release=False)
            return False, (
                "bwrap did not report a child pid before starting; refusing to run "
                f"without applying limits{': ' + detail if detail else ''}"
            )
        try:
            self.limiter.place(pid)
        except OSError as exc:
            return False, f"could not place pid {pid} under its limits: {exc}"
        try:
            os.write(block_write, b"\x01")
        except OSError:
            pass
        return True, None

    def _terminate(self, process: subprocess.Popen) -> None:
        """Kill the sandbox and everything it forked.

        The process group covers the bubblewrap processes; `--die-with-parent`
        and the cgroup cover what bubblewrap itself started, including anything
        that called `setsid` and left the group.
        """
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
        self.limiter.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def _start_session(
        self, command: tuple[str, ...], *, env: Mapping[str, str], cwd: str
    ) -> Session:
        """`open_session`, once the lock is held: fork the pty, place the child.

        The argv and the placement handshake are the same as `_spawn`'s; the
        difference is that the child's standard streams are a pty rather than
        pipes, so `os.forkpty` starts it instead of `subprocess.Popen`.
        """
        seccomp_fd = _seccomp_fd(self.program)
        info_read = info_write = block_read = block_write = None
        master_fd: int | None = None
        pid: int | None = None
        try:
            if self.limiter.needs_placement:
                info_read, info_write = os.pipe()
                block_read, block_write = os.pipe()
                # `Popen(pass_fds=...)` does this for a command; between
                # `forkpty` and the exec there is nobody to ask, so the child's
                # ends of the handshake are marked inheritable by hand
                os.set_inheritable(info_write, True)
                os.set_inheritable(block_read, True)
            if seccomp_fd is not None:
                os.set_inheritable(seccomp_fd, True)
            extra = dict(env)
            if "TERM" not in extra and os.environ.get("TERM"):
                # `default_environment` says TERM=dumb, and a dumb terminal has
                # no line editing; a session is the one case where borrowing
                # the caller's terminal type is the right answer
                extra["TERM"] = os.environ["TERM"]
            environment = bubblewrap.default_environment(
                self.spec, self.toolchain, _home_dir(self.mounts), extra=extra
            )
            argv = list(self.limiter.spawn_prefix()) + bubblewrap.build_argv(
                spec=self.spec,
                mounts=self.mounts,
                file_binds=self._file_binds,
                toolchain=self.toolchain,
                env=environment,
                cwd=cwd,
                command=command,
                seccomp_fd=seccomp_fd,
                info_fd=info_write,
                block_fd=block_read,
                hostname="sandbox",
                # the pty is the sandbox's terminal, so it must not be detached
                # from the session that owns it (see bubblewrap.build_argv)
                new_session=False,
            )
            pid, master_fd = os.forkpty()
            if pid == 0:
                _exec_in_pty(argv)
            # the child holds its own copies now; a closed pipe has to mean
            # "bwrap died" rather than "nobody wrote"
            seccomp_fd = _close(seccomp_fd)
            info_write = _close(info_write)
            block_read = _close(block_read)
            placed, failure = self._place_session(info_read, block_write, master_fd)
            if not placed:
                _kill_group(pid)
                self.limiter.kill()
                raise SandboxError(failure or "the session could not be started")
            session = Session(self, pid, master_fd, command)
            master_fd = None  # the session owns it now
            return session
        except BaseException:
            # nothing may outlive a failed start, the same way an unplaced
            # command is killed before its block pipe is released
            if pid is not None:
                _kill_group(pid)
            raise
        finally:
            for fd in (seccomp_fd, info_read, info_write, block_read, block_write, master_fd):
                _close(fd)

    def _place_session(
        self, info_read: int | None, block_write: int | None, master_fd: int | None
    ) -> tuple[bool, str | None]:
        """The handshake `_place` does, for a pid Python did not start.

        bubblewrap reports the sandbox's real PID on the info pipe and waits on
        the block pipe; placing that PID in its cgroup before releasing it is
        what keeps `pids.max` and `memory.max` covering everything the session
        will fork. If the handshake does not complete, the session is refused
        rather than run unlimited.
        """
        if info_read is None or block_write is None:
            return True, None
        child = _read_child_pid(info_read, SETUP_TIMEOUT)
        if child is None:
            detail = _seen_on_pty(master_fd)
            return False, (
                "bwrap did not report a child pid before starting; refusing to run "
                f"without applying limits{': ' + detail if detail else ''}"
            )
        try:
            self.limiter.place(child)
        except OSError as exc:
            return False, f"could not place pid {child} under its limits: {exc}"
        try:
            os.write(block_write, b"\x01")
        except OSError:
            pass
        return True, None

    def _live_session(self) -> Session | None:
        """The session still running, if any; a finished one is tidied up here."""
        session = self._session
        if session is None:
            return None
        if session.poll() is not None:
            # a command that ended on its own still owns a pty and an entry in
            # `_session`; close() is a no-op on the process and does both
            session.close()
            return None
        return session

    def _session_done(self, session: Session) -> None:
        """Called by a `Session` that has ended, however it ended."""
        if self._session is session:
            self._session = None
        if not self._closed:
            self._measure_disk()

    # -- files --------------------------------------------------------------

    def put_file(self, path: str, data: bytes | str) -> None:
        """Write a file inside the sandbox, through the mount that covers it.

        The path has to be under a host-backed writable mount. A `tmpfs` mount
        is recreated for every command — each one runs in fresh namespaces — so
        there is nowhere for a file to be put that the next command would see.
        """
        self._check_open()
        path = check_sandbox_path(path, "file path")
        payload = data.encode() if isinstance(data, str) else data
        with self._lock:
            self._check_open()
            mount = self._writable_mount(path)
            if mount is None:
                raise SandboxError(
                    f"{path} is not under a writable mount: {self.describe()['writable']}"
                )
            if mount.ephemeral:
                raise SandboxError(
                    f"{path} is on a tmpfs, which exists for one command and is gone by the "
                    "next; declare the path in `writable` instead (a host-backed directory), "
                    "or write it from inside the command that needs it"
                )
            target = Path(mount.source or "") / os.path.relpath(path, mount.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                handle.write(payload)
            os.chmod(target, _mode_for(payload))

    def get_file(self, path: str) -> bytes:
        """Read a file out of the sandbox, through the mount that covers it."""
        self._check_open()
        path = check_sandbox_path(path, "file path")
        mount = self._writable_mount(path)
        if mount is not None and mount.ephemeral:
            raise SandboxError(
                f"{path} is on a tmpfs, which exists for one command and is gone by the "
                "next; copy what you need into a host-backed mount (`writable=[...]`) "
                "from inside the command"
            )
        if mount is not None:
            target = Path(mount.source or "") / os.path.relpath(path, mount.path)
            with self._lock:
                self._check_open()
                try:
                    return target.read_bytes()
                except OSError as exc:
                    raise SandboxError(f"could not read {path}: {exc}") from exc
        # not a writable path: read it the way anything else would, from inside
        result = self.exec(["cat", "--", path], max_output=FILE_MAX_OUTPUT)
        if not result.ok:
            raise SandboxError(f"could not read {path}: {result.summary()}")
        if result.truncated:
            raise SandboxError(f"{path} is larger than the {FILE_MAX_OUTPUT} byte read cap")
        return result.stdout_bytes

    def _warn_about_external_writes(self) -> None:
        """Say so when `resources.disk` cannot cover a mount the caller supplied.

        The budget is measured over the sandbox's own state directories. A
        host directory the caller bound themselves is not the sandbox's to
        measure — it may have been full before the sandbox existed — so the only
        honest thing to do is say that the budget does not apply to it.
        """
        if self.spec.resources.disk is None:
            return
        outside = [
            mount.path
            for mount in self.mounts
            if mount.kind == "rw-bind"
            and mount.source is not None
            and self.layout.root not in Path(mount.source).parents
            and Path(mount.source) != self.layout.root
        ]
        if outside:
            self.warnings.append(
                f"resources.disk covers the sandbox's own state, not {', '.join(outside)} "
                "(a host directory you supplied); use Mount.tmpfs(...) for a hard cap there"
            )

    def _writable_mount(self, path: str) -> ResolvedMount | None:
        best: ResolvedMount | None = None
        for mount in self.mounts:
            if not mount.writable:
                continue
            if path == mount.path or path.startswith(mount.path.rstrip("/") + "/"):
                if best is None or len(mount.path) > len(best.path):
                    best = mount
        return best

    # -- budget, reporting, teardown ----------------------------------------

    def _measure_disk(self) -> None:
        if self.spec.resources.disk is None or not self._budgeted:
            return
        self.disk_used = sum(_tree_size(directory) for directory in self._budgeted)
        budget = self.spec.resources.disk
        if self.disk_used > budget:
            warning = (
                f"sandbox state is {_human(self.disk_used)} of a {_human(budget)} budget"
            )
            if warning not in self.warnings:
                self.warnings.append(warning)

    def _refuse_when_over_budget(self) -> None:
        budget = self.spec.resources.disk
        if budget is None or not self._budgeted:
            return
        self._measure_disk()
        if self.disk_used > budget:
            raise SandboxDiskExceeded(
                f"the sandbox has written {_human(self.disk_used)} of its "
                f"{_human(budget)} budget; raise resources.disk or destroy it"
            )

    def _check_open(self) -> None:
        if self._closed:
            raise SandboxError("this sandbox has been destroyed")

    def describe(self) -> dict[str, Any]:
        """Everything worth logging about this sandbox, in one dict."""
        return {
            "id": self.identity,
            "state_root": str(self.layout.root),
            "toolchain": self.toolchain.describe(),
            "packages": list(self.toolchain.packages),
            "mounts": [
                {
                    "path": mount.path,
                    "kind": mount.kind,
                    "source": mount.source,
                    "size": mount.size,
                }
                for mount in self.mounts
            ],
            "writable": [mount.path for mount in self.mounts if mount.writable],
            "network": self.spec.network.mode,
            "limits": self.limiter.report.describe(),
            "seccomp": seccomp_module.describe(self.program),
            "warnings": list(self.warnings),
        }

    def destroy(self, keep: bool = False) -> None:
        """Kill anything still running and remove the sandbox's state."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            session, self._session = self._session, None
            if session is not None:
                session.close(force=True)
            self.limiter.close()
        if not keep and not os.environ.get("HH_SANDBOX_KEEP"):
            shutil.rmtree(self.layout.root, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.destroy()

    def __repr__(self) -> str:
        return (
            f"<Sandbox {self.identity} at {self.layout.root} "
            f"({'destroyed' if self._closed else 'live'})>"
        )


# -- planning helpers -------------------------------------------------------


def _plan_mounts(spec: SandboxSpec, layout: Layout) -> tuple[ResolvedMount, ...]:
    """Every mount the sandbox will have, sources settled, shallowest first."""
    declared = list(spec.all_mounts())
    present = {mount.path for mount in declared}
    if DEFAULT_TMP not in present:
        declared.append(Mount.tmpfs(DEFAULT_TMP))
    if DEFAULT_HOME not in present:
        declared.append(Mount(DEFAULT_HOME, "rw-bind"))
    resolved: list[ResolvedMount] = []
    for mount in declared:
        if mount.kind == "tmpfs":
            resolved.append(
                ResolvedMount(
                    mount.path,
                    "tmpfs",
                    None,
                    mount.size or spec.resources.tmpfs_size,
                )
            )
        else:
            source = mount.source
            if source is None:
                # a writable path with no host directory of its own gets one
                # under the sandbox's state, which is also what `put_file` and
                # `get_file` reach through and what `resources.disk` measures
                directory = layout.host_dir(mount.path)
                if mount.kind == "rw-bind":
                    directory.mkdir(parents=True, exist_ok=True)
                source = str(directory)
            resolved.append(ResolvedMount(mount.path, mount.kind, source, None))
    # bubblewrap applies mounts in order and the last one wins, so the shallow
    # ones go down first and anything nested lands on top
    resolved.sort(key=lambda mount: mount.path.count("/"))
    return tuple(resolved)


def _stage(
    spec: SandboxSpec, layout: Layout, mounts: Sequence[ResolvedMount]
) -> tuple[tuple[str, str], ...]:
    """Write the sandbox's files, and return the read-only binds for them.

    A file under a writable mount is written into that mount's host directory,
    which is what makes `files={"/work/main.py": ...}` editable by the sandbox. A
    file anywhere else — including under a `tmpfs` mount, which has no host
    directory at all — is staged on the host and bound read-only, so `files=`
    can also mean "this input cannot be changed".

    The binds are *files*, never the `/etc` directory: bubblewrap creates a mount
    point before it mounts over it, so a later `--ro-bind /etc/x /etc/x` works
    only while `/etc` itself is still an ordinary writable directory in the root
    tmpfs. Mounting the staging directory at `/etc` would make every later file
    bind fail with `Read-only file system`.
    """
    _write_etc(layout, home=_home_dir(mounts))
    binds: list[tuple[str, str]] = [
        (str(path), f"/etc/{path.name}") for path in sorted(layout.etc.iterdir())
    ]
    for path, content in spec.files.items():
        payload = content.encode() if isinstance(content, str) else content
        mount = _deepest_writable(mounts, path)
        if mount is not None and not mount.ephemeral:
            target = Path(mount.source or "") / os.path.relpath(path, mount.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            os.chmod(target, _mode_for(payload))
        else:
            target = layout.ro_file(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            os.chmod(target, _mode_for(payload))
            binds.append((str(target), path))
    return tuple(binds)


def _write_etc(layout: Layout, *, home: str) -> None:
    """A minimal `/etc` — an identity, and nothing that names the host."""
    files = {
        "passwd": ETC_PASSWD.format(uid=os.getuid(), gid=os.getgid(), home=home),
        "group": ETC_GROUP.format(gid=os.getgid()),
        "hosts": ETC_HOSTS,
    }
    for name, content in files.items():
        (layout.etc / name).write_text(content)


def _home_dir(mounts: Sequence[ResolvedMount]) -> str:
    for mount in mounts:
        if mount.path == DEFAULT_HOME:
            return mount.path
    return DEFAULT_HOME


def _deepest_writable(
    mounts: Iterable[ResolvedMount], path: str
) -> ResolvedMount | None:
    best: ResolvedMount | None = None
    for mount in mounts:
        if not mount.writable:
            continue
        if path == mount.path or path.startswith(mount.path.rstrip("/") + "/"):
            if best is None or len(mount.path) > len(best.path):
                best = mount
    return best


def _check_cwd(spec: SandboxSpec, mounts: Sequence[ResolvedMount]) -> None:
    if spec.cwd == "/":
        return
    if not any(
        spec.cwd == mount.path or spec.cwd.startswith(mount.path.rstrip("/") + "/")
        for mount in mounts
    ):
        raise SpecError(
            f"cwd {spec.cwd} is not inside any mount; "
            f"add it to `writable`, or declare a mount for it"
        )


def _mode_for(payload: bytes) -> int:
    """Scripts are made executable; everything else is a plain file."""
    return 0o755 if payload.startswith(b"#!") else 0o644


def _package_names(packages: Sequence[str]) -> tuple[str, ...]:
    """Package arguments as a deduplicated tuple, order preserved."""
    names = tuple(str(name) for name in packages)
    if any(not name for name in names):
        raise SandboxError("a package name cannot be empty")
    return tuple(dict.fromkeys(names))


def _seccomp_fd(program: seccomp_module.Program | None) -> int | None:
    """The filter in a memfd, which is the shape `bwrap --seccomp` reads."""
    if program is None:
        return None
    fd = os.memfd_create("hh-seccomp")  # type: ignore[attr-defined]
    os.write(fd, program.data)
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


def _close(fd: int | None) -> None:
    if fd is None:
        return None
    try:
        os.close(fd)
    except OSError:
        pass
    return None


_INFO_PID_RE = re.compile(rb'"child-pid"\s*:\s*(\d+)')


def _read_child_pid(fd: int, timeout: float) -> int | None:
    """Read bubblewrap's `--info-fd` line, which names the sandbox's real PID."""
    deadline = time.monotonic() + timeout
    collected = bytearray()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], min(0.5, max(0.0, deadline - time.monotonic())))
        if not ready:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        collected += chunk
        if b"}" in collected:
            break
    match = _INFO_PID_RE.search(bytes(collected))
    return int(match.group(1)) if match else None


def _exec_in_pty(argv: list[str]) -> None:
    """The child half of `open_session`: turn the process into bubblewrap.

    Runs between `os.forkpty` and the exec, where the pty is already the
    controlling terminal and file descriptors 0, 1 and 2 — so the only work
    left is to replace the process. Nothing that allocates, logs or waits
    belongs here; this side of the fork has no threads and no clean shutdown.
    """
    try:
        os.execvp(argv[0], argv)
    except BaseException as exc:  # there is nothing sensible to raise after a fork
        try:
            os.write(2, f"could not start {argv[0]}: {exc}\n".encode())
        except OSError:
            pass
        os._exit(127)


def _seen_on_pty(master_fd: int | None, limit: int = 8192) -> str:
    """Whatever the child has already said on its pty, for an error message."""
    if master_fd is None:
        return ""
    text = b""
    try:
        while len(text) < limit:
            ready, _, _ = select.select([master_fd], [], [], 0)
            if not ready:
                break
            chunk = os.read(master_fd, 4096)
            if not chunk:
                break
            text += chunk
    except OSError:
        pass
    stripped = text.decode("utf-8", "replace").strip()
    return stripped.splitlines()[-1] if stripped else ""


def _kill_group(pid: int) -> None:
    """SIGKILL a session's process group, ignoring a group that is already gone."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


def _feed(stream: Any, payload: bytes) -> None:
    try:
        stream.write(payload)
        stream.flush()
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _drain(process: subprocess.Popen, *, release: bool = True) -> str:
    """Whatever the process managed to say, for the error message.

    A held sandbox must not be waited for — it is waiting for us — so the pipes
    are read without blocking. Everything bubblewrap has written so far is still
    in them, which is where a setup failure ends up.
    """
    if release:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    text = b""
    for stream in (process.stderr, process.stdout):
        if stream is None:
            continue
        try:
            fd = stream.fileno()
            os.set_blocking(fd, False)
        except (OSError, ValueError):
            continue
        try:
            while len(text) < 8192:
                try:
                    chunk = os.read(fd, 4096)
                except (BlockingIOError, OSError):
                    break
                if not chunk:
                    break
                text += chunk
        finally:
            try:
                os.set_blocking(fd, True)
            except OSError:
                pass
    stripped = text.decode("utf-8", "replace").strip()
    return stripped.splitlines()[-1] if stripped else ""


def _tree_size(directory: str) -> int:
    """Disk actually used by a directory, in bytes (blocks, not apparent size)."""
    total = 0
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(info.st_mode):
                stack.append(entry.path)
            elif stat.S_ISREG(info.st_mode):
                total += info.st_blocks * 512
    return total


def _human(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size} B"
