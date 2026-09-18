"""The live-sandbox registry behind the `nix_*` tools.

The sandbox itself is the `sandbox` package at the project root; this module is
the harness side of it. It keeps the shape the tools promise:

* one fixed container — 256M memory, 512M disk, 256 pids, one CPU, the host
  network, and a writable `/workspace` as the working directory. None of that
  is a tool parameter: the model chooses *what* runs in a sandbox, never how
  much of the machine it gets.
* ids live in this process's memory only, so nothing about a sandbox is written
  into the block chain. Forking a block cannot resurrect one and a restarted
  server has forgotten them all; the model carries the id in the conversation.
* at most `MAX_SANDBOXES` exist at once; one more `spawn` is refused rather
  than evicting a sandbox somebody else may still be using.
* a daemon thread sweeps every `SWEEP_INTERVAL` and destroys sandboxes that no
  tool call has named for `IDLE_TIMEOUT`.

Every method that names an id touches that sandbox's idle clock, so a status
poll keeps a sandbox alive exactly as an exec does — and a call that is still
running keeps it alive for its whole duration, so an install slower than the
idle timeout cannot have its sandbox swept out from under it. Methods return
the text a tool should hand the model instead of raising: an unknown id, a
full registry and a failing command are normal answers, not crashes.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import secrets
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The one configuration every harness sandbox gets. The tool table in
# `tools.py` and the spawn result both describe these values, and the tests
# pin them; the model never passes any of them.
SANDBOX_WORKSPACE = "/workspace"
SANDBOX_MEMORY_BYTES = 256 * 1024**2
SANDBOX_DISK_BYTES = 512 * 1024**2
SANDBOX_PIDS = 256
SANDBOX_CPU = 1.0
SANDBOX_NETWORK = True  # the host's network namespace: a sandbox that can fetch

# How many sandboxes may exist at once, how long one may go untouched before a
# sweep may destroy it, and how often the sweeper runs. The idle timeout is
# shorter than the interval on purpose: a release can lag the last call by up
# to one sweep, which `nix_sandbox_status` is how the model discovers.
MAX_SANDBOXES = 10
IDLE_TIMEOUT = 10 * 60.0
SWEEP_INTERVAL = 20 * 60.0

# `nix_exec` must always pass a timeout, and the harness refuses anything past
# ten minutes rather than silently clamping it.
EXEC_TIMEOUT_MAX = 10 * 60.0
# Tool results stay in the transcript of every later request, so a command that
# prints megabytes is clipped for the model; the capture cap keeps the harness
# itself bounded.
EXEC_OUTPUT_MAX_CHARS = 20_000
EXEC_CAPTURE_BYTES = 200_000
# A file handed in as base64 in a tool call is bounded by the model's own
# context, but the server should not allocate unboundedly on a bad caller.
ADD_FILE_MAX_BYTES = 16 * 1024 * 1024

_IDLE_TEXT = f"{IDLE_TIMEOUT / 60:g} minutes"


@dataclass
class SandboxEntry:
    """One live sandbox plus the timestamps the reaper reads."""

    id: str
    sandbox: Any
    created: float
    last_used: float
    # calls currently in flight; `sweep` leaves an entry alone while this is
    # non-zero, however stale its clock looks
    active: int = 0


class SandboxRegistry:
    """The sandboxes this harness process knows about, keyed by id.

    The `create` and `id_factory` hooks exist so the tests can run the whole
    lifecycle against a fake sandbox without Nix or bubblewrap; production
    code uses the defaults, which build one real sandbox from `sandbox_spec()`.
    """

    def __init__(
        self,
        *,
        max_sandboxes: int = MAX_SANDBOXES,
        idle_timeout: float = IDLE_TIMEOUT,
        sweep_interval: float = SWEEP_INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        create: Callable[[], Any] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.max_sandboxes = max_sandboxes
        self.idle_timeout = idle_timeout
        self.sweep_interval = sweep_interval
        self._clock = clock
        self._create = create or _create_sandbox
        self._id_factory = id_factory or _new_id
        self._entries: dict[str, SandboxEntry] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------

    def spawn(self) -> str:
        """Create a sandbox and return the tool result text.

        The lock is held across the creation so the capacity check cannot be
        raced by two callers; a Nix build can take a while, and the price is
        that a concurrent status call waits it out.
        """
        with self._lock:
            if len(self._entries) >= self.max_sandboxes:
                return self._full_text()
            sandbox_id = self._unused_id()
            try:
                sandbox = self._create()
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                return _failure("could not create a sandbox", exc)
            now = self._clock()
            self._entries[sandbox_id] = SandboxEntry(
                id=sandbox_id, sandbox=sandbox, created=now, last_used=now
            )
            self._start_reaper()
        return _spawn_text(sandbox_id, sandbox)

    def status(self, sandbox_id: str) -> str:
        """Say whether one sandbox is still live, and what it looks like."""
        with self._using(sandbox_id) as entry:
            if entry is None:
                return self._gone_text(sandbox_id)
            now = self._clock()
            packages = ", ".join(entry.sandbox.packages)
            lines = [
                f"Sandbox {sandbox_id} is live.",
                f"  created {_age(now - entry.created)} ago; last used {_age(now - entry.last_used)} ago",
                f"  workspace: {SANDBOX_WORKSPACE} (writable; also the working directory)",
                f"  packages: {packages}",
                f"  limits: {_limits_text()}",
            ]
            warnings = list(getattr(entry.sandbox, "warnings", ()) or ())
            if warnings:
                lines.append(
                    "  warnings: " + "; ".join(str(warning) for warning in warnings)
                )
            return "\n".join(lines)

    def destroy(self, sandbox_id: str) -> str:
        """Destroy one sandbox now, freeing its slot."""
        with self._using(sandbox_id) as entry:
            if entry is None:
                return (
                    f"Sandbox {sandbox_id} is not live; nothing to destroy. "
                    + self._gone_reason()
                )
            with self._lock:
                self._entries.pop(sandbox_id, None)
            try:
                entry.sandbox.destroy()
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                return _failure(
                    f"removed {sandbox_id} from the registry, but cleaning it up failed",
                    exc,
                )
            return (
                f"Destroyed sandbox {sandbox_id}: its processes are stopped and "
                "its files are gone."
            )

    # -- operations ---------------------------------------------------------

    def add_dependency(self, sandbox_id: str, package: Any) -> str:
        """Add one Nix package; it lands in the environment of the next exec."""
        with self._using(sandbox_id) as entry:
            if entry is None:
                return self._gone_text(sandbox_id)
            if not isinstance(package, str) or not package.strip():
                return (
                    "Error: package must be a Nix package name such as python312, "
                    "git or ripgrep."
                )
            package = package.strip()
            if package in entry.sandbox.packages:
                return (
                    f"{package} is already in sandbox {sandbox_id} "
                    f"(packages: {', '.join(entry.sandbox.packages)})."
                )
            try:
                packages = entry.sandbox.add_packages(package)
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                return _failure(f"could not add {package!r} to sandbox {sandbox_id}", exc)
            return (
                f"Added {package} to sandbox {sandbox_id}; the next nix_exec will see it "
                f"(packages: {', '.join(packages)})."
            )

    def remove_dependency(self, sandbox_id: str, package: Any) -> str:
        """Remove one Nix package added earlier."""
        with self._using(sandbox_id) as entry:
            if entry is None:
                return self._gone_text(sandbox_id)
            if not isinstance(package, str) or not package.strip():
                return (
                    "Error: package must be a Nix package name such as python312, "
                    "git or ripgrep."
                )
            package = package.strip()
            try:
                packages = entry.sandbox.remove_packages(package)
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                return _failure(f"could not remove {package!r} from sandbox {sandbox_id}", exc)
            return (
                f"Removed {package} from sandbox {sandbox_id}; the next nix_exec will see it "
                f"(packages: {', '.join(packages)})."
            )

    def exec(self, sandbox_id: str, command: Any, timeout: Any) -> str:
        """Run one shell command in a sandbox and format everything it did."""
        with self._using(sandbox_id) as entry:
            if entry is None:
                return self._gone_text(sandbox_id)
            if not isinstance(command, str) or not command.strip():
                return "Error: command must be a non-empty shell command line."
            seconds = _exec_timeout(timeout)
            if seconds is None:
                return (
                    "Error: timeout must be a number of seconds between 1 and "
                    f"{int(EXEC_TIMEOUT_MAX)}; got {_short(timeout)}."
                )
            try:
                result = entry.sandbox.exec(
                    ["bash", "-c", command],
                    timeout=seconds,
                    max_output=EXEC_CAPTURE_BYTES,
                )
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                return _failure(f"could not run the command in sandbox {sandbox_id}", exc)
            return _exec_text(result)

    def add_file(self, sandbox_id: str, path: Any, content_base64: Any) -> str:
        """Write one file, handed over as base64, into a sandbox."""
        with self._using(sandbox_id) as entry:
            if entry is None:
                return self._gone_text(sandbox_id)
            if not isinstance(path, str) or not path.startswith("/"):
                return (
                    "Error: path must be an absolute path inside the sandbox, "
                    f"e.g. {SANDBOX_WORKSPACE}/input.bin; got {_short(path)}."
                )
            if not isinstance(content_base64, str):
                return "Error: content_base64 must be the file's bytes as a base64 string."
            try:
                data = _decode_base64(content_base64)
            except ValueError as exc:
                return f"Error: content_base64 is not valid base64: {exc}"
            if len(data) > ADD_FILE_MAX_BYTES:
                return (
                    f"Error: the file is {len(data)} bytes; nix_add_file accepts at most "
                    f"{ADD_FILE_MAX_BYTES} bytes."
                )
            try:
                entry.sandbox.put_file(path, data)
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                return _failure(f"could not write {path} in sandbox {sandbox_id}", exc)
            note = " (executable: it starts with '#!')" if data.startswith(b"#!") else ""
            return f"Wrote {len(data)} bytes to {path} in sandbox {sandbox_id}{note}."

    # -- sweeping -----------------------------------------------------------

    def sweep(self, now: float | None = None) -> list[str]:
        """Destroy every sandbox idle for `idle_timeout`; returns their ids.

        Entries leave the registry before their teardown runs, so nothing can
        call a half-destroyed sandbox, and one failing `destroy` cannot stop
        the rest or kill the reaper.
        """
        moment = self._clock() if now is None else now
        with self._lock:
            expired = [
                entry
                for entry in self._entries.values()
                if entry.active == 0 and moment - entry.last_used >= self.idle_timeout
            ]
            for entry in expired:
                self._entries.pop(entry.id, None)
        for entry in expired:
            with contextlib.suppress(Exception):  # a janitor must not die
                entry.sandbox.destroy()
        return [entry.id for entry in expired]

    def live_ids(self) -> tuple[str, ...]:
        """The ids that are currently live, sorted."""
        with self._lock:
            return tuple(sorted(self._entries))

    def shutdown(self) -> None:
        """Stop the reaper; live sandboxes are left running for the caller to end."""
        self._stop.set()
        reaper = self._reaper
        if reaper is not None and reaper.is_alive():
            reaper.join(timeout=5.0)

    # -- internals ----------------------------------------------------------

    @contextlib.contextmanager
    def _using(self, sandbox_id: str) -> Iterator[SandboxEntry | None]:
        """Yield the entry for `sandbox_id` with one call counted against it.

        The idle clock is touched on the way in — any call about a sandbox is
        use — and again on the way out, and the entry is marked busy for as long
        as the call runs, so `sweep` cannot destroy a sandbox under an install
        that takes longer than the idle timeout.
        """
        now = self._clock()
        with self._lock:
            entry = self._entries.get(sandbox_id)
            if entry is not None:
                entry.last_used = now
                entry.active += 1
        try:
            yield entry
        finally:
            if entry is not None:
                with self._lock:
                    entry.active -= 1
                    entry.last_used = self._clock()

    def _unused_id(self) -> str:
        while True:
            candidate = self._id_factory()
            if not candidate:
                raise ValueError("the sandbox id factory returned an empty id")
            if candidate not in self._entries:
                return candidate

    def _full_text(self) -> str:
        # the caller holds the lock, so this reads the entries directly
        live = ", ".join(sorted(self._entries))
        return (
            f"Error: cannot create a sandbox: {self.max_sandboxes} already exist "
            f"({live}), which is the limit; destroy one with nix_destroy_sandbox "
            "before creating another."
        )

    def _gone_reason(self) -> str:
        return (
            f"it was destroyed, released after {_IDLE_TEXT} without a call, or the "
            "harness has restarted since it was created (sandbox ids live in memory "
            "only)"
        )

    def _gone_text(self, sandbox_id: str) -> str:
        return (
            f"Sandbox {sandbox_id} is not live: {self._gone_reason()}. "
            "Create a new one with nix_spawn_sandbox."
        )

    def _start_reaper(self) -> None:
        if self._stop.is_set():
            return
        if self._reaper is not None and self._reaper.is_alive():
            return
        self._reaper = threading.Thread(
            target=self._reap_forever, name="sandbox-reaper", daemon=True
        )
        self._reaper.start()

    def _reap_forever(self) -> None:
        while not self._stop.wait(self.sweep_interval):
            self.sweep()


# -- the fixed configuration -------------------------------------------------


def sandbox_spec() -> Any:
    """The one spec every harness sandbox is created from.

    Kept separate from `Sandbox.create` so the configuration is a plain value
    object: a test can pin every limit without a kernel, Nix or bubblewrap.
    """
    module = _sandbox_module()
    return module.SandboxSpec(
        packages=(),
        writable=(SANDBOX_WORKSPACE,),
        cwd=SANDBOX_WORKSPACE,
        network=SANDBOX_NETWORK,
        resources=module.ResourceLimits(
            memory=SANDBOX_MEMORY_BYTES,
            disk=SANDBOX_DISK_BYTES,
            pids=SANDBOX_PIDS,
            cpu=SANDBOX_CPU,
            timeout=EXEC_TIMEOUT_MAX,
        ),
    )


def _create_sandbox() -> Any:
    """Build one sandbox from the fixed spec."""
    module = _sandbox_module()
    return module.Sandbox.create(sandbox_spec())


_SANDBOX_MODULE: Any = None


def _sandbox_module() -> Any:
    """Import the `sandbox` package, which lives beside `src/`.

    The server is normally started as `python src/server.py`, so `src/` is the
    only directory on `sys.path`; the project root is added on first use, the
    same way `sandbox/test.py` finds the package when run directly.
    """
    global _SANDBOX_MODULE
    if _SANDBOX_MODULE is None:
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        import sandbox

        _SANDBOX_MODULE = sandbox
    return _SANDBOX_MODULE


# -- formatting --------------------------------------------------------------


def _new_id() -> str:
    return f"sbx-{secrets.token_hex(6)}"


def _exec_timeout(value: Any) -> float | None:
    """Seconds from a model-supplied timeout, or None when it is out of range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 1 or value > EXEC_TIMEOUT_MAX:
        return None
    return float(value)


def _decode_base64(content: str) -> bytes:
    """Bytes from base64 a model wrote, tolerant of wrapping and missing padding."""
    text = "".join(content.split())
    text = text.replace("-", "+").replace("_", "/")
    text += "=" * (-len(text) % 4)
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _exec_text(result: Any) -> str:
    if result.timed_out:
        head = (
            f"timed out after {result.duration:.1f}s and was killed "
            f"(exit code {result.exit_code})"
        )
    else:
        head = f"exit code {result.exit_code} in {result.duration:.2f}s"
    parts = [head]
    stdout = str(result.stdout)
    stderr = str(result.stderr)
    if stdout.strip():
        parts += ["stdout:", _clip(stdout, "stdout")]
    if stderr.strip():
        parts += ["stderr:", _clip(stderr, "stderr")]
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output)")
    if result.truncated:
        parts.append(
            f"[output was capped at {EXEC_CAPTURE_BYTES} bytes; write what you need "
            f"to a file under {SANDBOX_WORKSPACE} and read it in pieces]"
        )
    return "\n".join(parts)


def _clip(text: str, name: str) -> str:
    if len(text) <= EXEC_OUTPUT_MAX_CHARS:
        return text
    return (
        text[:EXEC_OUTPUT_MAX_CHARS]
        + f"\n[{name} truncated: showing the first {EXEC_OUTPUT_MAX_CHARS} of "
        f"{len(text)} characters]"
    )


def _spawn_text(sandbox_id: str, sandbox: Any) -> str:
    packages = ", ".join(sandbox.packages)
    lines = [
        f"Created sandbox {sandbox_id}.",
        f"  workspace: {SANDBOX_WORKSPACE} (writable; also the working directory)",
        f"  environment: {packages} — add packages with nix_add_dependency",
        f"  limits: {_limits_text()}",
        (
            "Each nix_exec runs in fresh namespaces: files under "
            f"{SANDBOX_WORKSPACE} survive between commands, but background processes do not."
        ),
        (
            f"The sandbox is released after {_IDLE_TEXT} without a call; check it with "
            "nix_sandbox_status, and destroy it early with nix_destroy_sandbox when done."
        ),
    ]
    warnings = list(getattr(sandbox, "warnings", ()) or ())
    if warnings:
        lines.append("  warnings: " + "; ".join(str(warning) for warning in warnings))
    return "\n".join(lines)


def _failure(what: str, exc: BaseException) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return f"Error: {what}: {detail}"


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _age(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    if seconds < 1:
        return "less than a second"
    if seconds < 60:
        return f"{seconds}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s" if rest else f"{minutes}m"
    hours, rest = divmod(minutes, 60)
    return f"{hours}h{rest:02d}m" if rest else f"{hours}h"


def _size_text(value: int) -> str:
    for unit, scale in (("G", 1024**3), ("M", 1024**2), ("K", 1024)):
        if value >= scale and value % scale == 0:
            return f"{value // scale}{unit}"
    return str(value)


def _limits_text() -> str:
    network = "host" if SANDBOX_NETWORK else "none"
    return (
        f"memory {_size_text(SANDBOX_MEMORY_BYTES)}, "
        f"disk {_size_text(SANDBOX_DISK_BYTES)}, "
        f"pids {SANDBOX_PIDS}, cpu {SANDBOX_CPU:.1f}, network {network}"
    )


# The registry the `nix_*` tools use. Importing this module starts no thread:
# the reaper is born with the first sandbox and dies with the process.
SANDBOXES = SandboxRegistry()
