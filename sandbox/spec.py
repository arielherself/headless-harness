"""What a sandbox *is*: mounts, network, resources, syscalls.

Everything in this module is a frozen value object. Nothing here touches the
kernel, Nix or bubblewrap — a spec is a description, and `Sandbox.create` is
what turns one into a live sandbox. Keeping the two apart means a spec can be
built, compared, logged and sent between processes without side effects, and it
is the only part of the package that is worth unit-testing in isolation.

The shapes here follow the design this package implements: a spec names the Nix
packages to build an environment from, the files to place and where they may be
written, the network policy, the resource limits, and the syscall policy. The
sizes and durations are written the way a person writes them — `"512M"`,
`"1G"`, `"30s"` — and parsed once, here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

# Where a sandbox's own files live when the caller does not say otherwise.
DEFAULT_WORK = "/work"
DEFAULT_HOME = "/home/user"
DEFAULT_TMP = "/tmp"
# A scratch tmpfs is memory-backed and charged to the sandbox's cgroup, so it is
# sized well below the default memory budget instead of left unbounded.
DEFAULT_TMPFS_SIZE = 256 * 1024 * 1024
# A sandbox with no watchdog is a harness that hangs; callers can still ask for
# no limit with `timeout=None`.
DEFAULT_TIMEOUT = 60.0


class SpecError(ValueError):
    """A spec that cannot be turned into a sandbox."""


_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "tib": 1024**4,
}
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", re.IGNORECASE)


def parse_size(value: int | str) -> int:
    """Bytes from `512`, `"512M"`, `"1.5GiB"`. Units are binary, always."""
    if isinstance(value, bool):
        raise SpecError(f"not a size: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise SpecError(f"size cannot be negative: {value}")
        return value
    if isinstance(value, float):
        return parse_size(int(value))
    if not isinstance(value, str):
        raise SpecError(f"not a size: {value!r}")
    match = _SIZE_RE.match(value)
    if not match:
        raise SpecError(f"cannot read size {value!r} (try '512M', '1G', '1024')")
    unit = _SIZE_UNITS.get(match.group(2).lower())
    if unit is None:
        raise SpecError(f"unknown size unit in {value!r}")
    return int(float(match.group(1)) * unit)


def parse_duration(value: float | int | str) -> float:
    """Seconds from `30`, `"30s"`, `"1m30s"`, `"500ms"`."""
    if isinstance(value, bool):
        raise SpecError(f"not a duration: {value!r}")
    if isinstance(value, (int, float)):
        if value < 0:
            raise SpecError(f"duration cannot be negative: {value}")
        return float(value)
    if not isinstance(value, str):
        raise SpecError(f"not a duration: {value!r}")
    text = value.strip()
    if not text:
        raise SpecError("empty duration")
    if _SIZE_RE.match(text) and not _DURATION_PART_RE.match(text):
        # a bare number means seconds, as in `timeout: 30`
        return float(text)
    total = 0.0
    position = 0
    for match in _DURATION_PART_RE.finditer(text):
        if match.start() != position:
            raise SpecError(f"cannot read duration {value!r} (try '30', '1m30s')")
        total += float(match.group(1)) * _DURATION_UNITS[match.group(2).lower()]
        position = match.end()
    if position != len(text) or position == 0:
        raise SpecError(f"cannot read duration {value!r} (try '30', '1m30s')")
    return total


def check_sandbox_path(path: str, what: str) -> str:
    """Validate a path that names something inside the sandbox, normalized.

    Absolute, and free of `.` and `..`: a harness that hands an agent-supplied
    path to `put_file` must not be talked into walking out of the mount it
    landed in, so the check is here rather than in the caller.
    """
    if not path.startswith("/"):
        raise SpecError(f"{what} must be an absolute path, got {path!r}")
    parts = [part for part in path.split("/") if part]
    if any(part in (".", "..") for part in parts):
        raise SpecError(f"{what} must not contain '.' or '..': {path!r}")
    return "/" + "/".join(parts)


def _prefix_of(path: str, outer: str) -> bool:
    """Whether `path` is `outer` or lives below it."""
    return path == outer or path.startswith(outer.rstrip("/") + "/")


@dataclass(frozen=True)
class Mount:
    """One thing visible in the sandbox's filesystem tree.

    `ro-bind` and `rw-bind` expose a host path; `tmpfs` conjures a memory-backed
    filesystem that never touches the host disk and disappears with the sandbox.

    A `tmpfs` mount is the only way to get a *hard* size cap on writes: the
    kernel refuses the write, where a host directory can only be measured after
    the fact (see `ResourceLimits.disk`).
    """

    path: str
    kind: Literal["ro-bind", "rw-bind", "tmpfs"]
    source: str | None = None
    size: int | None = None

    def __post_init__(self) -> None:
        # paths are normalized here so `/work/` and `/work` are the same mount
        object.__setattr__(self, "path", check_sandbox_path(self.path, "mount path"))
        if self.kind not in ("ro-bind", "rw-bind", "tmpfs"):
            raise SpecError(f"unknown mount kind {self.kind!r}")
        if self.kind == "tmpfs":
            if self.source is not None:
                raise SpecError(f"tmpfs mount {self.path} cannot have a source")
            if self.size is not None and self.size <= 0:
                raise SpecError(f"tmpfs mount {self.path} needs a positive size")
        else:
            # a read-only bind with no source is always a mistake; a writable
            # one with no source is the layout's job, which gives it a directory
            # under the sandbox's own state
            if self.kind == "ro-bind" and not self.source:
                raise SpecError(f"ro-bind mount {self.path} needs a source")
            if self.source is not None and not self.source.startswith("/"):
                raise SpecError(f"mount source must be absolute: {self.source!r}")
            if self.size is not None:
                raise SpecError(f"a bind mount cannot have a size: {self.path}")

    @property
    def writable(self) -> bool:
        return self.kind in ("rw-bind", "tmpfs")

    @property
    def ephemeral(self) -> bool:
        """Whether anything written here is unreachable from the host.

        The harness cannot hand files in or out of an ephemeral mount by path:
        `put_file` and `get_file` refuse it rather than hand back a file that
        will not be there. See the lifetime notes in the package README.
        """
        return self.kind == "tmpfs"

    @classmethod
    def ro_bind(cls, source: str, path: str | None = None) -> "Mount":
        return cls(path=path or source, kind="ro-bind", source=source)

    @classmethod
    def rw_bind(cls, source: str, path: str | None = None) -> "Mount":
        return cls(path=path or source, kind="rw-bind", source=source)

    @classmethod
    def tmpfs(cls, path: str, size: int | str | None = None) -> "Mount":
        return cls(path=path, kind="tmpfs", size=parse_size(size) if size is not None else None)


@dataclass(frozen=True)
class NetworkPolicy:
    """What the sandbox may reach.

    `none` gets its own network namespace: a fresh interface list holding only
    `lo`, so there is nothing to route to and nothing to resolve with. `host`
    shares the harness's namespace and is the escape hatch for sandboxes that
    are not a boundary.

    `allow` is the shape a domain allow-list will take — it needs a CONNECT
    proxy outside the namespace to mean anything, and until that exists
    `Sandbox.create` refuses it rather than pretending.
    """

    mode: Literal["none", "host", "allow"] = "none"
    allow: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("none", "host", "allow"):
            raise SpecError(f"unknown network mode {self.mode!r}")
        if self.mode == "allow" and not self.allow:
            raise SpecError("network mode 'allow' needs at least one domain")
        if self.mode != "allow" and self.allow:
            raise SpecError("network.allowed is only meaningful with mode 'allow'")

    @property
    def isolated(self) -> bool:
        """Whether the sandbox gets a network namespace of its own."""
        return self.mode != "host"

    @classmethod
    def none(cls) -> "NetworkPolicy":
        return cls("none")

    @classmethod
    def host(cls) -> "NetworkPolicy":
        return cls("host")

    @classmethod
    def allowing(cls, *domains: str) -> "NetworkPolicy":
        return cls("allow", tuple(domains))


@dataclass(frozen=True)
class IoLimits:
    """Throughput caps for the sandbox's own block device.

    Applied through cgroup v2's `io.max`, which speaks per-device numbers, so
    the device is resolved from the filesystem the sandbox's state lives on.
    """

    read_bps: int | None = None
    write_bps: int | None = None
    read_iops: int | None = None
    write_iops: int | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "IoLimits":
        known = {"read_bps", "write_bps", "read_iops", "write_iops"}
        unknown = set(values) - known
        if unknown:
            raise SpecError(f"unknown io limit(s): {', '.join(sorted(unknown))}")
        return cls(
            read_bps=parse_size(values["read_bps"]) if "read_bps" in values else None,
            write_bps=parse_size(values["write_bps"]) if "write_bps" in values else None,
            read_iops=int(values["read_iops"]) if "read_iops" in values else None,
            write_iops=int(values["write_iops"]) if "write_iops" in values else None,
        )

    def cgroup_value(self, device: tuple[int, int]) -> str:
        """The `io.max` line for this device, e.g. `8:0 rbps=1048576 wbps=..`."""
        parts = []
        if self.read_bps is not None:
            parts.append(f"rbps={self.read_bps}")
        if self.write_bps is not None:
            parts.append(f"wbps={self.write_bps}")
        if self.read_iops is not None:
            parts.append(f"riops={self.read_iops}")
        if self.write_iops is not None:
            parts.append(f"wiops={self.write_iops}")
        return f"{device[0]}:{device[1]} " + " ".join(parts)

    @property
    def any(self) -> bool:
        return any(
            value is not None
            for value in (self.read_bps, self.write_bps, self.read_iops, self.write_iops)
        )


@dataclass(frozen=True)
class ResourceLimits:
    """What the sandbox may consume.

    `memory`, `cpu` and `pids` are cgroup v2 hard limits: the kernel refuses
    the allocation, the fork, the extra cycle. `cpu` is in cores, so `1.0` is
    one core's worth of quota over a 100 ms period.

    `disk` is different in kind, and the difference is worth knowing. It bounds
    the sandbox's writable state on the host — the work directory, home and any
    other host-backed mount — but no portable unprivileged mechanism caps a
    directory's total size, so it is enforced by measuring after each command
    and refusing to run the next one when the budget is spent. `RLIMIT_FSIZE`
    is set alongside it, which does cap any single file at the kernel level.
    Declare the hot directory as a `tmpfs` mount when a hard cap matters.
    """

    memory: int | None = None
    cpu: float | None = None
    pids: int | None = None
    disk: int | None = None
    io: IoLimits | None = None
    timeout: float | None = DEFAULT_TIMEOUT
    tmpfs_size: int = DEFAULT_TMPFS_SIZE

    def __post_init__(self) -> None:
        for name in ("memory", "disk", "tmpfs_size"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise SpecError(f"{name} must be positive, got {value}")
        if self.cpu is not None and self.cpu <= 0:
            raise SpecError(f"cpu must be positive, got {self.cpu}")
        if self.pids is not None and self.pids < 1:
            raise SpecError(f"pids must be at least 1, got {self.pids}")
        if self.timeout is not None and self.timeout <= 0:
            raise SpecError(f"timeout must be positive, got {self.timeout}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ResourceLimits":
        """Build from the dict shape the design uses for `resources={...}`."""
        values = dict(values)
        known = {"memory", "cpu", "pids", "disk", "io", "timeout", "tmpfs_size"}
        unknown = set(values) - known
        if unknown:
            raise SpecError(f"unknown resource(s): {', '.join(sorted(unknown))}")
        if "io" in values and isinstance(values["io"], Mapping):
            values["io"] = IoLimits.from_mapping(values["io"])
        for key in ("memory", "disk", "tmpfs_size"):
            if key in values:
                values[key] = parse_size(values[key])
        if "timeout" in values:
            values["timeout"] = None if values["timeout"] is None else parse_duration(values["timeout"])
        if "cpu" in values:
            values["cpu"] = float(values["cpu"])
        if "pids" in values:
            values["pids"] = int(values["pids"])
        return cls(**values)


@dataclass(frozen=True)
class SyscallPolicy:
    """Which syscalls the sandbox may make, and which user namespaces it may make.

    The denylist is installed by bubblewrap as a seccomp filter with
    `SECCOMP_RET_ERRNO`, so a denied call fails with `EPERM` instead of killing
    the process — a runtime that probes a syscall and falls back keeps working.

    The default list is deliberately short and every entry is there for a
    reason: syscalls that would edit the mount table, load kernel code, reach
    another process, or change the clock. It is not a "sounds dangerous" list,
    because compilers, Node and Python all use syscalls that look dangerous in
    isolation. `allow` removes an entry when a workload needs it.
    """

    enabled: bool = True
    deny: tuple[str | int, ...] = ()
    allow: tuple[str, ...] = ()
    errno: Literal["EPERM", "ENOSYS"] = "EPERM"
    # bubblewrap's --disable-userns: makes `unshare(CLONE_NEWUSER)` fail inside,
    # which is what stops a sandboxed process from building a fresh user
    # namespace and re-obtaining capabilities within it.
    disable_userns: bool = True


@dataclass(frozen=True)
class SandboxSpec:
    """A whole sandbox description.

    `packages` are Nix attribute names (`"python312"`, `"ripgrep"`) or absolute
    store paths. `bash` and `coreutils` are always added, because `/bin/sh` and
    `/usr/bin/env` have to resolve for other people's scripts to run.

    `files` places content inside: a path under a writable mount is written
    there and is writable, anything else is staged read-only and bind-mounted.
    `writable` adds writable paths, each either a plain path (backed by a
    directory under the sandbox's state, so the harness can hand files in and
    out) or a `Mount` (which is how a scratch `tmpfs` with a hard size cap is
    declared).
    """

    packages: tuple[str, ...] = ()
    files: Mapping[str, bytes | str] = field(default_factory=dict)
    writable: tuple[str | Mount, ...] = (DEFAULT_WORK,)
    mounts: tuple[Mount, ...] = ()
    # both of these accept the value objects above or the shapes the design
    # writes in prose — `network=False`, `network={"allow": [...]}`,
    # `resources={"memory": "512M"}` — which `__post_init__` normalizes
    network: NetworkPolicy | bool | str | Mapping[str, Any] | Sequence[str] = field(
        default_factory=NetworkPolicy.none
    )
    resources: ResourceLimits | Mapping[str, Any] = field(default_factory=ResourceLimits)
    syscalls: SyscallPolicy = field(default_factory=SyscallPolicy)
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str = DEFAULT_WORK
    # A private root is enough to keep the host out; making it read-only as well
    # costs nothing and closes the one unbounded write target left (the root
    # tmpfs has no size of its own). Anything a sandbox is meant to write to is
    # a mount, so turn this off only for a workload that writes to `/` itself.
    read_only_root: bool = True
    # Where the environment comes from. `nixpkgs` overrides `<nixpkgs>` (a path
    # or any Nix fetcher argument); `env_dir` skips Nix entirely and points at an
    # existing directory with a `bin/`, which is what the tests and `--host-tools`
    # use. Setting both is a contradiction. An `env_dir` is only a `bin/`: whatever
    # its binaries need mounted alongside (a host-tools environment needs the
    # host's libraries) is the caller's business — `toolchain.host_toolchain()`
    # returns the mounts it needs, and `_mount_args` adds the toolchain's own
    # mounts but cannot know about an arbitrary directory's libraries.
    nixpkgs: str | None = None
    env_dir: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "packages", tuple(self.packages))
        object.__setattr__(self, "writable", tuple(self.writable))
        object.__setattr__(self, "mounts", tuple(self.mounts))
        # the shapes the design uses in prose are accepted here, so `network=False`
        # and `resources={"memory": "512M"}` mean what they look like
        object.__setattr__(self, "network", network_from(self.network))
        object.__setattr__(self, "resources", resources_from(self.resources))
        object.__setattr__(self, "cwd", check_sandbox_path(self.cwd, "cwd"))
        object.__setattr__(
            self,
            "files",
            {
                check_sandbox_path(str(path), "file path"): content
                for path, content in self.files.items()
            },
        )
        object.__setattr__(
            self,
            "writable",
            tuple(
                entry
                if isinstance(entry, Mount)
                else check_sandbox_path(entry, "writable path")
                for entry in self.writable
            ),
        )
        object.__setattr__(self, "env", {str(k): str(v) for k, v in self.env.items()})
        if self.env_dir:
            object.__setattr__(self, "env_dir", check_sandbox_path(self.env_dir, "env_dir"))
        if self.packages and self.env_dir:
            raise SpecError("pass either packages= (Nix) or env_dir=, not both")
        for name, value in self.env.items():
            if "\x00" in name or "\x00" in value or "=" in name:
                raise SpecError(f"environment entry {name!r} cannot be passed to a process")
        self._validate_tree()

    def _validate_tree(self) -> None:
        """Refuse specs whose mounts contradict each other.

        Whether the cwd exists is checked later, against the full mount list
        including the ones the layout adds (`/tmp`, the home directory, the Nix
        store) — this only catches two mounts claiming one path.
        """
        seen: dict[str, Mount] = {}
        for mount in self.all_mounts():
            if mount.path in seen:
                raise SpecError(f"two mounts for {mount.path}")
            seen[mount.path] = mount

    def all_mounts(self) -> tuple[Mount, ...]:
        """The declared mounts plus the writable paths, in mount order."""
        declared = list(self.mounts)
        for entry in self.writable:
            if isinstance(entry, Mount):
                declared.append(entry)
            else:
                declared.append(Mount(path=entry, kind="rw-bind"))
        # deepest first, so an inner mount is not shadowed by an outer one
        declared.sort(key=lambda mount: mount.path.count("/"), reverse=True)
        return tuple(declared)

    def mount_for(self, path: str) -> Mount | None:
        """The deepest mount covering `path`, if any."""
        best: Mount | None = None
        for mount in self.all_mounts():
            if _prefix_of(path, mount.path):
                if best is None or len(mount.path) > len(best.path):
                    best = mount
        return best

    def writable_mount_for(self, path: str) -> Mount | None:
        mount = self.mount_for(path)
        if mount is not None and mount.writable:
            return mount
        return None

    def with_changes(self, **changes: Any) -> "SandboxSpec":
        return replace(self, **changes)


def network_from(value: Any) -> NetworkPolicy:
    """`False`/`True`/`"none"`/`"host"`/a domain list/a policy, as one policy."""
    if isinstance(value, NetworkPolicy):
        return value
    if value is False or value is None:
        return NetworkPolicy.none()
    if value is True:
        return NetworkPolicy.host()
    if isinstance(value, str):
        if value in ("none", "host"):
            return NetworkPolicy(mode=value)  # type: ignore[arg-type]
        return NetworkPolicy.allowing(value)
    if isinstance(value, Mapping):
        return NetworkPolicy(
            mode=value.get("mode", "allow"),
            allow=tuple(value.get("allow", ())),
        )
    if isinstance(value, Sequence):
        return NetworkPolicy.allowing(*value)
    raise SpecError(f"cannot read network policy from {value!r}")


def resources_from(value: Any) -> ResourceLimits:
    """A `ResourceLimits`, or the dict shape the design uses."""
    if isinstance(value, ResourceLimits):
        return value
    if value is None:
        return ResourceLimits()
    if isinstance(value, Mapping):
        return ResourceLimits.from_mapping(value)
    raise SpecError(f"cannot read resources from {value!r}")
