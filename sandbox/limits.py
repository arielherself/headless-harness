"""How the sandbox is stopped from taking more than its share.

Three mechanisms, tried in order, because no single one is available everywhere:

* **cgroup v2** (`/sys/fs/cgroup`) is the real thing: `memory.max` and
  `memory.swap.max` bound the sandbox's memory and its swap, `cpu.max` bounds
  CPU time as quota per period, `pids.max` bounds fork, `io.max` bounds
  throughput on the sandbox's block device. It also gives the cleanest cleanup:
  one write to `cgroup.kill` and the whole tree is gone. It needs a delegated,
  writable cgroup — which a systemd user session provides and a container often
  does not.
* **systemd-run --user --scope** asks the user manager for a transient scope and
  passes the same knobs as unit properties. It is the same cgroup underneath,
  and it is the fallback when this process is not itself sitting in a delegated
  subtree.
* **rlimits** always work and are always a compromise: `RLIMIT_FSIZE` really does
  cap a single file, `RLIMIT_AS` caps *address space* rather than memory (which
  breaks runtimes that reserve address space they never touch), and there is no
  per-tree CPU or PID limit at all. When this is the engine, the report says so.

Whatever the engine, the limits have to cover the sandbox process from before it
forks anything. Every engine therefore works with the same handshake: bubblewrap
starts with `--info-fd` and `--block-fd`, reports the real PID of the process
that will become the sandbox, and holds it there until `place()` has put that PID
where it belongs. Nothing slips through a fork race, and a sandbox that could not
be limited reports that instead of pretending.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .spec import ResourceLimits

CGROUP_MOUNT = Path("/sys/fs/cgroup")
# cgroup file -> the limit it enforces. These names are the kernel's interface.
CGROUP_FILES = {
    "memory": "memory.max",
    "cpu": "cpu.max",
    "pids": "pids.max",
    "io": "io.max",
}
CPU_PERIOD_US = 100_000


@dataclass(frozen=True)
class LimitsReport:
    """What was actually enforced, in the kernel's own words."""

    engine: str
    applied: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def enforced(self) -> bool:
        return self.engine in ("cgroup", "systemd")

    def describe(self) -> str:
        if self.engine == "none":
            detail = "no resource limits"
        else:
            detail = f"{self.engine}: " + ", ".join(
                f"{key}={value}" for key, value in sorted(self.applied.items())
            )
        if self.warnings:
            detail += " | " + "; ".join(self.warnings)
        return detail


class Limiter:
    """The process-tree half of resource limiting.

    `spawn_prefix()` is prepended to the bubblewrap command, `place()` is handed
    the sandbox's real PID while it is still held at bubblewrap's block point,
    and `kill()`/`close()` tear down whatever was created.
    """

    def __init__(self, report: LimitsReport, prefix: tuple[str, ...] = ()) -> None:
        self.report = report
        self.prefix = prefix
        self.directory: Path | None = None

    @property
    def needs_placement(self) -> bool:
        """Whether a process must be handed to `place()` before it runs.

        Only the cgroup engine needs the handshake; the others work either
        through a wrapper process or through limits that are inherited by
        everything the sandbox forks.
        """
        return False

    def spawn_prefix(self) -> tuple[str, ...]:
        """Arguments that must precede bubblewrap for this spawn."""
        return self.prefix

    def place(self, pid: int) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def kill(self) -> None:
        """Kill everything the sandbox has running."""

    def close(self) -> None:
        """`kill`, then release whatever this limiter created."""


class NullLimiter(Limiter):
    """No limits: nothing created, nothing enforced, and the report says so."""

    def __init__(self, warnings: tuple[str, ...] = (), prefix: tuple[str, ...] = ()) -> None:
        super().__init__(LimitsReport(engine="none", warnings=warnings), prefix)

    def place(self, pid: int) -> None:
        del pid


class CgroupLimiter(Limiter):
    """A cgroup of the sandbox's own, under a delegated base.

    One cgroup is created per sandbox and reused by every command it runs, so
    an `io.max` or `memory.max` written once keeps applying to each new process
    tree. `place()` is what puts that tree there.
    """

    def __init__(
        self, directory: Path, report: LimitsReport, prefix: tuple[str, ...] = ()
    ) -> None:
        super().__init__(report, prefix)
        self.directory = directory

    @property
    def needs_placement(self) -> bool:
        return True

    def place(self, pid: int) -> None:
        if self.directory is None:
            return
        _write(self.directory / "cgroup.procs", str(pid))

    def kill(self) -> None:
        if self.directory is None:
            return
        kill_file = self.directory / "cgroup.kill"
        if kill_file.exists():
            try:
                _write(kill_file, "1")
                return
            except OSError:
                pass
        for pid in self._pids():
            try:
                os.kill(pid, 9)
            except OSError:
                pass

    def close(self) -> None:
        if self.directory is None:
            return
        self.kill()
        # the kernel needs a moment to reap what it just killed
        for _ in range(100):
            if not self._pids():
                break
            time.sleep(0.02)
        try:
            self.directory.rmdir()
        except OSError:
            pass  # a cgroup that outlives us is left to the next cleanup
        self.directory = None

    def _pids(self) -> list[int]:
        if self.directory is None:
            return []
        try:
            text = (self.directory / "cgroup.procs").read_text()
        except OSError:
            return []
        return [int(line) for line in text.split() if line.strip().isdigit()]

    def read(self, name: str) -> str | None:
        """One of the sandbox's accounting files, for the self-test."""
        if self.directory is None:
            return None
        try:
            return (self.directory / name).read_text().strip()
        except OSError:
            return None


class SystemdLimiter(Limiter):
    """A transient user scope per command, with limits as unit properties.

    The scope is created by systemd-run around bubblewrap, so the whole process
    tree is inside it from the first instruction: `place()` has nothing to do.
    """

    def __init__(
        self,
        sandbox_id: str,
        properties: tuple[str, ...],
        report: LimitsReport,
        prefix: tuple[str, ...] = (),
    ) -> None:
        super().__init__(report, prefix)
        self.sandbox_id = sandbox_id
        self.properties = properties
        self._spawns = 0

    def spawn_prefix(self) -> tuple[str, ...]:
        self._spawns += 1
        unit = f"hh-sandbox-{self.sandbox_id}-{self._spawns}"
        args = ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}"]
        for prop in self.properties:
            args += ["-p", prop]
        args.append("--")
        return tuple(args) + self.prefix

    def place(self, pid: int) -> None:
        del pid


class RlimitLimiter(Limiter):
    """`prlimit` in front of bubblewrap: the last resort, inside the process."""

    def place(self, pid: int) -> None:
        del pid


def _write(path: Path, value: str) -> None:
    with path.open("w") as handle:
        handle.write(value)


def _own_cgroup() -> str | None:
    """This process's own cgroup path, from the cgroup namespace's point of view."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            fields = line.split(":", 2)
            if len(fields) == 3 and fields[0] == "0":
                return "/" + fields[2].strip().lstrip("/")
    except OSError:
        return None
    return None


def _candidate_bases() -> list[Path]:
    """Where a sandbox cgroup could live, deepest first.

    The deepest candidate is this process's own cgroup, which is where a
    systemd-delegated unit wants its children. Enabling controllers for children
    is refused while a cgroup has processes of its own, so when that fails the
    search walks up — the user manager's own cgroup is usually the first
    ancestor that already has controllers delegated to it.
    """
    override = os.environ.get("HH_SANDBOX_CGROUP")
    if override:
        return [Path(override)]
    own = _own_cgroup()
    if own is None:
        return []
    parts = [part for part in own.split("/") if part]
    return [CGROUP_MOUNT.joinpath(*parts[:count]) for count in range(len(parts), -1, -1)]


def _controller_of(filename: str) -> str:
    return filename.split(".", 1)[0]


def _try_base(base: Path, name: str, needed: list[str]) -> Path | None:
    """Create `base/name` with the controllers we need, or give up quietly."""
    if not (base / "cgroup.controllers").exists():
        return None
    target = base / name
    try:
        target.mkdir()
    except OSError:
        return None
    missing = [filename for filename in needed if not (target / filename).exists()]
    if missing:
        enable = " ".join(f"+{_controller_of(filename)}" for filename in missing)
        try:
            _write(base / "cgroup.subtree_control", enable)
        except OSError:
            pass
        missing = [filename for filename in needed if not (target / filename).exists()]
    if missing:
        try:
            target.rmdir()
        except OSError:
            pass
        return None
    return target


def _mount_source(path: str) -> str | None:
    """The `/dev/...` a path's filesystem is mounted from.

    Needed because a number of filesystems — btrfs above all — report an
    anonymous device in `st_dev`, which is not the device `io.max` wants.
    """
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    best: tuple[int, str] | None = None
    for line in lines:
        fields = line.split(" ")
        if "-" not in fields:
            continue
        separator = fields.index("-")
        if len(fields) <= separator + 2:
            continue
        mount_point = fields[4].replace("\\040", " ")
        source = fields[separator + 2].split("[")[0]
        if not (path == mount_point or path.startswith(mount_point.rstrip("/") + "/")):
            continue
        if best is None or len(mount_point) > best[0]:
            best = (len(mount_point), source)
    if best is None or not best[1].startswith("/dev/"):
        return None
    return best[1]


def _io_device(state_dir: str) -> tuple[int, int] | None:
    """The block device the sandbox's state lives on, as `io.max` wants it."""
    source = _mount_source(state_dir)
    if source is not None:
        try:
            return os.major(os.stat(source).st_rdev), os.minor(os.stat(source).st_rdev)
        except OSError:
            pass
    try:
        stat = os.stat(state_dir)
    except OSError:
        return None
    if os.major(stat.st_dev) == 0:
        return None  # an anonymous device: tmpfs, overlay, a network filesystem
    return os.major(stat.st_dev), os.minor(stat.st_dev)


def _apply(
    directory: Path, limits: ResourceLimits, state_dir: str
) -> tuple[dict[str, str], list[str]]:
    """Write the cgroup interface files for the limits that were asked for."""
    applied: dict[str, str] = {}
    warnings: list[str] = []

    def put(filename: str, value: str) -> None:
        if not (directory / filename).exists():
            warnings.append(f"{filename} is not available on this kernel")
            return
        try:
            _write(directory / filename, value)
            applied[filename] = value
        except OSError as exc:
            warnings.append(f"could not set {filename}: {exc.strerror}")

    if limits.memory is not None:
        put("memory.max", str(limits.memory))
        put("memory.swap.max", "0")
        # an OOM inside a sandbox should take the sandbox down, not one
        # arbitrary process in the middle of it
        put("memory.oom.group", "1")
    if limits.cpu is not None:
        put("cpu.max", f"{max(1, int(round(limits.cpu * CPU_PERIOD_US)))} {CPU_PERIOD_US}")
    if limits.pids is not None:
        put("pids.max", str(limits.pids))
    if limits.io is not None and limits.io.any:
        device = _io_device(state_dir)
        if device is None:
            warnings.append("io limits skipped: the sandbox state is not on a block device")
        else:
            put("io.max", limits.io.cgroup_value(device))
    return applied, warnings


_SYSTEMD_PROBES: dict[tuple[str, ...], bool] = {}


def _systemd_run_usable(properties: tuple[str, ...] = ()) -> bool:
    """Whether a user manager will take this work, probed once per property set.

    The probe carries the real properties, because a property this systemd does
    not understand would make every later `exec` fail at spawn — better to find
    out while there is still an engine to fall back to.
    """
    if properties in _SYSTEMD_PROBES:
        return _SYSTEMD_PROBES[properties]
    probe_binary = next(
        (path for path in ("/bin/true", "/usr/bin/true") if os.path.exists(path)), None
    )
    usable = False
    if probe_binary is not None and shutil.which("systemd-run") is not None:
        argv = ["systemd-run", "--user", "--scope", "--quiet", "--collect"]
        for prop in properties:
            argv += ["-p", prop]
        argv += ["--", probe_binary]
        try:
            usable = subprocess.run(argv, capture_output=True, timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            usable = False
    _SYSTEMD_PROBES[properties] = usable
    return usable


def _systemd_properties(
    limits: ResourceLimits, state_dir: str
) -> tuple[list[str], list[str]]:
    properties: list[str] = []
    warnings: list[str] = []
    if limits.memory is not None:
        properties.append(f"MemoryMax={limits.memory}")
        properties.append("MemorySwapMax=0")
        properties.append("MemoryOOMGroup=yes")
    if limits.cpu is not None:
        # systemd reads 0% as "no quota", so the same floor the cgroup path uses
        properties.append(f"CPUQuota={max(1, int(round(limits.cpu * 100)))}%")
    if limits.pids is not None:
        properties.append(f"TasksMax={limits.pids}")
    if limits.io is not None and limits.io.any:
        source = _mount_source(state_dir)
        if source is None:
            warnings.append("io limits skipped: no block device for the sandbox state")
        else:
            if limits.io.read_bps is not None:
                properties.append(f"IOReadBandwidthMax={source} {limits.io.read_bps}")
            if limits.io.write_bps is not None:
                properties.append(f"IOWriteBandwidthMax={source} {limits.io.write_bps}")
            if limits.io.read_iops is not None:
                properties.append(f"IOReadIOPSMax={source} {limits.io.read_iops}")
            if limits.io.write_iops is not None:
                properties.append(f"IOWriteIOPSMax={source} {limits.io.write_iops}")
    return properties, warnings


def _prlimit_prefix(
    limits: ResourceLimits, *, memory: bool
) -> tuple[list[str], list[str]]:
    """`prlimit` arguments for what rlimits can genuinely enforce.

    `RLIMIT_FSIZE` is worth having in every engine — it is the only unprivileged
    mechanism that makes the kernel refuse a write — while `RLIMIT_AS` is only
    used when there is no cgroup, because it limits address space rather than
    memory and a runtime that reserves a lot of address space it never touches
    would fail for no good reason.
    """
    args: list[str] = []
    warnings: list[str] = []
    if limits.disk is not None:
        args.append(f"--fsize={limits.disk}")
    if memory and limits.memory is not None:
        args.append(f"--as={limits.memory}")
        warnings.append(
            "RLIMIT_AS is address space, not resident memory: a runtime that "
            "reserves more than it uses may fail"
        )
    if not args:
        return [], warnings
    if shutil.which("prlimit") is None:
        return [], warnings + ["prlimit not found: rlimits are not applied"]
    return ["prlimit", *args, "--"], warnings


def create_limiter(
    limits: ResourceLimits, *, state_dir: str, sandbox_id: str, engine: str = "auto"
) -> Limiter:
    """Build the strongest limiter this machine can give us for `limits`."""
    if engine not in ("auto", "cgroup", "systemd", "rlimit", "none"):
        raise ValueError(f"unknown engine {engine!r}")
    warnings: list[str] = []
    wanted = any(
        value is not None for value in (limits.memory, limits.cpu, limits.pids)
    ) or (limits.io is not None and limits.io.any)

    if engine == "none" or not wanted:
        if engine == "none" and wanted:
            warnings.append("limits were requested but the engine is 'none'")
        prefix, prefix_warnings = _prlimit_prefix(limits, memory=False)
        warnings.extend(prefix_warnings)
        return NullLimiter(tuple(warnings), tuple(prefix))

    if engine in ("auto", "cgroup"):
        needed = [CGROUP_FILES[key] for key in CGROUP_FILES if _wanted(key, limits)]
        # A base that can hold memory, cpu and pids but not io should still be
        # used: falling all the way to rlimits over one missing controller would
        # give up limits that were available, so io is retried without.
        attempts = [needed]
        if CGROUP_FILES["io"] in needed:
            attempts.append([name for name in needed if name != CGROUP_FILES["io"]])
        for attempt in attempts:
            target = None
            for base in _candidate_bases():
                target = _try_base(base, f"hh-sandbox-{sandbox_id}", attempt)
                if target is not None:
                    break
            if target is None:
                continue
            applied, apply_warnings = _apply(target, limits, state_dir)
            warnings.extend(f"{base}: {warning}" for warning in apply_warnings)
            if len(attempt) < len(needed):
                warnings.append(
                    "io.max is not available on a usable cgroup here; the other limits are applied"
                )
            if applied:
                prefix, prefix_warnings = _prlimit_prefix(limits, memory=False)
                warnings.extend(prefix_warnings)
                return CgroupLimiter(
                    target,
                    LimitsReport("cgroup", applied, tuple(warnings)),
                    tuple(prefix),
                )
            try:
                target.rmdir()
            except OSError:
                pass
        if engine == "cgroup":
            warnings.append("no writable delegated cgroup was found")
            prefix, prefix_warnings = _prlimit_prefix(limits, memory=False)
            warnings.extend(prefix_warnings)
            return NullLimiter(tuple(warnings), tuple(prefix))
        warnings.append("no delegated cgroup: falling back to systemd-run")

    if engine in ("auto", "systemd"):
        properties, systemd_warnings = _systemd_properties(limits, state_dir)
        if _systemd_run_usable(tuple(properties)) and properties:
            warnings.extend(systemd_warnings)
            prefix, prefix_warnings = _prlimit_prefix(limits, memory=False)
            warnings.extend(prefix_warnings)
            applied = {
                prop.split("=", 1)[0]: prop.split("=", 1)[1] for prop in properties
            }
            return SystemdLimiter(
                sandbox_id,
                tuple(properties),
                LimitsReport("systemd", applied, tuple(warnings)),
                tuple(prefix),
            )
        if engine == "systemd":
            warnings.append("systemd-run --user is not usable here (with these properties)")
            prefix, prefix_warnings = _prlimit_prefix(limits, memory=False)
            warnings.extend(prefix_warnings)
            return NullLimiter(tuple(warnings), tuple(prefix))
        warnings.append("systemd-run --user is not usable: falling back to rlimits")

    prefix, prefix_warnings = _prlimit_prefix(limits, memory=True)
    warnings.extend(prefix_warnings)
    if limits.cpu is not None or limits.pids is not None:
        warnings.append("CPU and PID limits are not enforced by the rlimit engine")
    limiter = RlimitLimiter(LimitsReport("rlimit", _applied_rlimits(prefix), tuple(warnings)))
    limiter.prefix = tuple(prefix)
    return limiter


def _applied_rlimits(prefix: list[str]) -> dict[str, str]:
    """Only the rlimits that made it into the command, in the command's words."""
    applied: dict[str, str] = {}
    for argument in prefix:
        name, separator, value = argument.lstrip("-").partition("=")
        if not separator or not name:
            continue  # the `--` separator, or prlimit's own name
        applied[f"RLIMIT_{name.upper()}"] = value
    return applied


def _wanted(key: str, limits: ResourceLimits) -> bool:
    if key == "io":
        return limits.io is not None and limits.io.any
    return getattr(limits, key) is not None


__all__ = [
    "CgroupLimiter",
    "LimitsReport",
    "Limiter",
    "NullLimiter",
    "RlimitLimiter",
    "SystemdLimiter",
    "create_limiter",
    "CGROUP_MOUNT",
]
