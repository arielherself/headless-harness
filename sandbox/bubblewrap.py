"""Turning a plan into a bubblewrap command line.

bubblewrap is where the isolation actually comes from, and its argument list is
the whole specification of what the sandbox can see. This module builds that
list, in the order bubblewrap applies it, from three inputs: the mounts the
sandbox should have, the environment the toolchain provides, and the namespaces
the spec asked for.

The order is not cosmetic. Mounts are applied in the order given and a later
mount lands on top of an earlier one, so read-only roots come before writable
ones; and a *file* can only be placed inside a directory that is still writable
at the time, because bubblewrap creates the mount point before mounting over it.
That is why `/etc` is never mounted as a directory: the sandbox's own `/etc`
files are bound one by one onto a `/etc` that bubblewrap creates in the root
tmpfs, and the host's certificate and resolver files are layered on top of those
same paths.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .spec import NetworkPolicy, SandboxSpec
from .toolchain import Toolchain

# Not every bubblewrap in the wild is new enough to have all of these; checking
# the help text once is more honest than parsing versions.
REQUIRED_FLAGS = (
    "--unshare-all",
    "--unshare-user",
    "--share-net",
    "--disable-userns",
    "--die-with-parent",
    "--new-session",
    "--cap-drop",
    "--info-fd",
    "--block-fd",
    "--seccomp",
    "--size",
    "--ro-bind-try",
    "--remount-ro",
    "--clearenv",
    "--setenv",
    "--symlink",
    "--chdir",
    "--hostname",
)


class BubblewrapError(RuntimeError):
    """bubblewrap is missing, too old, or refused the command."""


@dataclass(frozen=True)
class ResolvedMount:
    """A mount with its source settled: a host path, or nothing for tmpfs."""

    path: str
    kind: str
    source: str | None = None
    size: int | None = None

    @property
    def writable(self) -> bool:
        return self.kind in ("rw-bind", "tmpfs")

    @property
    def ephemeral(self) -> bool:
        return self.kind == "tmpfs"

    @property
    def flag(self) -> str:
        """The bubblewrap option that creates this mount."""
        return "--bind" if self.kind == "rw-bind" else f"--{self.kind}"


_bwrap_help: str | None = None


def check() -> str:
    """Verify bubblewrap is present and has the options this package uses."""
    global _bwrap_help
    binary = shutil.which("bwrap")
    if binary is None:
        raise BubblewrapError(
            "bwrap is not on PATH; install bubblewrap (the package is what provides isolation)"
        )
    if _bwrap_help is None:
        try:
            probe = subprocess.run([binary, "--help"], capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as exc:
            raise BubblewrapError(f"could not run {binary} --help: {exc}") from exc
        if probe.returncode != 0:
            detail = probe.stderr.decode("utf-8", "replace").strip()
            raise BubblewrapError(
                f"{binary} --help failed ({probe.returncode}): {detail or 'no output'}"
            )
        help_text = probe.stdout.decode("utf-8", "replace")
        missing = [flag for flag in REQUIRED_FLAGS if flag not in help_text]
        if missing:
            version = subprocess.run([binary, "--version"], capture_output=True, timeout=15)
            raise BubblewrapError(
                f"this bubblewrap ({version.stdout.decode().strip()}) does not support: "
                + ", ".join(missing)
            )
        _bwrap_help = help_text
    return binary


def _namespace_args(network: NetworkPolicy, disable_userns: bool) -> list[str]:
    args = ["--unshare-all"]
    if not network.isolated:
        # --share-net is only meaningful next to --unshare-all, which is exactly
        # where it is needed: this is the "not a boundary" case
        args.append("--share-net")
    # explicit, because --disable-userns refuses to be inferred
    args.append("--unshare-user")
    if disable_userns:
        args.append("--disable-userns")
    args += ["--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    return args


def _mount_args(
    mounts: Iterable[ResolvedMount],
    file_binds: Iterable[tuple[str, str]],
    toolchain: Toolchain,
    *,
    network: NetworkPolicy,
    tmpfs_size: int,
    read_only_root: bool,
) -> list[str]:
    mounts = list(mounts)
    declared = {mount.path for mount in mounts}

    # These come first so that a mount the spec declared for the same path lands
    # on top of them: bubblewrap applies mounts in order and the last one wins.
    # The store is the exception — it is the toolchain's own closure, and a spec
    # that mounts over it knows what it is doing.
    args: list[str] = ["--ro-bind-try", "/nix/store", "/nix/store"]
    args += ["--proc", "/proc"]
    args += ["--dev", "/dev"]
    if "/dev/shm" not in declared:
        # a writable tmpfs like any other, so it gets the same size cap
        args += ["--size", str(tmpfs_size), "--tmpfs", "/dev/shm"]

    # Everything else is collected and applied shallowest-first, because a bind
    # of `/tmp/host-tools` has to come after the tmpfs on `/tmp` or the tmpfs
    # covers it and every binary in it disappears.
    tree: list[ResolvedMount] = [
        ResolvedMount(m.path, "ro-bind", m.source, None) for m in toolchain.mounts
    ]
    if not toolchain.path.startswith("/nix/store/"):
        tree.append(ResolvedMount(toolchain.path, "ro-bind", toolchain.path, None))
    tree += mounts
    tree.sort(key=lambda mount: mount.path.count("/"))

    # Whatever the toolchain brought, `/bin/sh` and `/usr/bin/env` have to
    # resolve or other people's scripts do not run. A link is skipped where a
    # mount already covers the path in either direction, because bubblewrap
    # refuses to create a node on top of one that already exists.
    occupied = [mount.path for mount in tree] + ["/proc", "/dev"]
    for link in toolchain.link_paths:
        if any(_overlaps(link, path) for path in occupied):
            continue
        args += ["--symlink", toolchain.bin_dir, link]

    for mount in tree:
        if mount.kind == "tmpfs":
            args += ["--size", str(mount.size or tmpfs_size), "--tmpfs", mount.path]
        else:
            args += [mount.flag, mount.source or "", mount.path]

    for source, destination in file_binds:
        args += ["--ro-bind", source, destination]
    if not network.isolated:
        # with no network namespace there is a network to resolve names on and
        # certificates to check, so lend the host's
        for path in ("/etc/resolv.conf", "/etc/ssl", "/etc/ca-certificates"):
            args += ["--ro-bind-try", path, path]
    if read_only_root:
        # last, so the file binds above can still create their mount points in
        # the root tmpfs; the root itself has no size of its own, and nothing a
        # sandbox is meant to write to lives there
        args += ["--remount-ro", "/"]
    return args


def _overlaps(one: str, other: str) -> bool:
    """Whether two sandbox paths are equal or one contains the other."""
    return (
        one == other
        or one.startswith(other.rstrip("/") + "/")
        or other.startswith(one.rstrip("/") + "/")
    )


def build_argv(
    *,
    spec: SandboxSpec,
    mounts: Sequence[ResolvedMount],
    file_binds: Sequence[tuple[str, str]],
    toolchain: Toolchain,
    env: Mapping[str, str],
    cwd: str,
    command: Sequence[str],
    seccomp_fd: int | None,
    info_fd: int | None,
    block_fd: int | None,
    hostname: str = "sandbox",
) -> list[str]:
    """The full `bwrap ... -- command` argument list."""
    if not command:
        raise BubblewrapError("no command to run")
    argv = [check()]
    argv += _namespace_args(spec.network, spec.syscalls.disable_userns)
    argv += ["--hostname", hostname]
    if seccomp_fd is not None:
        argv += ["--seccomp", str(seccomp_fd)]
    if info_fd is not None:
        argv += ["--info-fd", str(info_fd)]
    if block_fd is not None:
        argv += ["--block-fd", str(block_fd)]

    argv += ["--clearenv"]
    for name, value in env.items():
        argv += ["--setenv", name, value]

    argv += _mount_args(
        mounts,
        file_binds,
        toolchain,
        network=spec.network,
        tmpfs_size=spec.resources.tmpfs_size,
        read_only_root=spec.read_only_root,
    )
    argv += ["--chdir", cwd]
    argv += ["--", *command]
    return argv


def default_environment(
    spec: SandboxSpec,
    toolchain: Toolchain,
    home: str,
    *,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment inside, before per-command overrides.

    Small on purpose: `--clearenv` means nothing leaks in from the harness, so
    everything a program may need has to be named here.
    """
    shell = f"{toolchain.bin_dir}/bash"
    if not os.path.exists(shell):
        shell = "/bin/sh"
    env = {
        "PATH": f"{toolchain.bin_dir}:/bin:/usr/bin",
        "HOME": home,
        "USER": "user",
        "LOGNAME": "user",
        "SHELL": shell,
        "TMPDIR": "/tmp",
        "TERM": "dumb",
        # a sandbox has no locale archive of its own, and C.UTF-8 is built into
        # glibc, so text handling stays sane without shipping locales
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PAGER": "cat",
        "NO_COLOR": "1",
    }
    if not spec.network.isolated:
        for variable, path in (
            ("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt"),
            ("NIX_SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt"),
        ):
            if os.path.exists(path):
                env[variable] = path
    env.update(spec.env)
    if extra:
        env.update(extra)
    return env



