"""A sandbox for running untrusted commands: Nix + bubblewrap + cgroup v2 + seccomp.

The model, layer by layer, from the outside in:

    cgroup v2        memory.max, cpu.max, pids.max, io.max — hard limits the
                     kernel enforces, applied before the sandbox can fork
    bubblewrap       user, pid, mount, network, ipc, uts namespaces; a private
                     root; read-only bind mounts of the Nix store; a size-capped
                     tmpfs for scratch space
    seccomp          a denylist of syscalls that would let a process change the
                     mount table, load kernel code, reach another process, or
                     move the clock
    Nix              what is *inside* — an immutable store path with exactly the
                     packages that were asked for, instead of an image to unpack

Nothing here uses `chroot` on its own, because `chroot` is a path-resolution
change and not an isolation boundary. bubblewrap does the namespace work and
`chroot` never appears in the implementation; the root filesystem is a private
tmpfs with things bind-mounted into it.

    from sandbox import Sandbox

    sandbox = Sandbox.create(
        packages=["python312", "git", "ripgrep"],
        files={"/work/main.py": code, "/input/data.json": data},
        writable=["/work"],
        network=False,
        resources={"memory": "512M", "cpu": 1.0, "pids": 64, "disk": "1G",
                   "timeout": 30},
    )
    result = sandbox.exec(["python", "/work/main.py"])
    sandbox.destroy()

What each part of a spec buys, and what it does not:

| You asked for              | Enforced by                          | Hard? |
|----------------------------|--------------------------------------|-------|
| files invisible/unreadable | mount namespace, no host binds       | yes   |
| a path read-only           | `--ro-bind`                          | yes   |
| a path writable            | `--bind` to a state directory        | yes   |
| no network                 | network namespace (only `lo`)        | yes   |
| memory                     | `memory.max`, `memory.swap.max`      | yes   |
| cpu                        | `cpu.max`                            | yes   |
| process count              | `pids.max`                           | yes   |
| disk I/O                   | `io.max`                             | yes   |
| scratch space size         | tmpfs `--size`; `/` is read-only too | yes   |
| the sandbox's own files    | measured between commands            | no    |
| syscalls                   | seccomp filter                       | yes   |
| wall-clock time            | harness watchdog, then SIGKILL       | yes   |
| software environment       | Nix store path                       | yes   |

The one soft limit is `resources.disk` applied to the writable mounts the
sandbox owns: no portable unprivileged mechanism caps a directory's total size,
so it is measured after each command and the sandbox refuses to run once the
budget is spent. An `RLIMIT_FSIZE` is set alongside it, which does make the
kernel refuse a single file larger than the budget. A host directory supplied by
the caller is not measured (it may have been full already), and a warning says
so; declare the hot path as a `Mount.tmpfs(...)` when a hard cap matters more
than seeing the files from outside.

Every command gets fresh namespaces, which is what makes the model simple to
reason about and has one visible consequence: host-backed writable paths persist
between commands (they are directories the harness can also reach), while a
tmpfs mount is created, used and discarded within the command that asked for it.
`put_file` and `get_file` refuse a tmpfs path instead of returning a file that
will not be there. See `sandbox/README.md` for the session-scoped alternative and
why a seccomp filter cannot be inherited by a process that joins later.

Not implemented yet: `NetworkPolicy.allowing(...)`, which needs a CONNECT proxy
outside the network namespace to have any meaning. `Sandbox.create` refuses it
explicitly rather than quietly treating it as `none`.

Run `python -m sandbox --self-test` to check, on this machine, that each row of
that table actually holds.
"""

from . import limits, seccomp, toolchain
from .limits import LimitsReport
from .sandbox import ExecResult, Layout, Sandbox, SandboxDiskExceeded, SandboxError
from .spec import (
    DEFAULT_HOME,
    DEFAULT_TIMEOUT,
    DEFAULT_TMP,
    DEFAULT_TMPFS_SIZE,
    DEFAULT_WORK,
    IoLimits,
    Mount,
    NetworkPolicy,
    ResourceLimits,
    SandboxSpec,
    SpecError,
    SyscallPolicy,
)
from .toolchain import Toolchain, ToolchainError

__all__ = [
    "DEFAULT_HOME",
    "DEFAULT_TIMEOUT",
    "DEFAULT_TMP",
    "DEFAULT_TMPFS_SIZE",
    "DEFAULT_WORK",
    "ExecResult",
    "IoLimits",
    "Layout",
    "LimitsReport",
    "Mount",
    "NetworkPolicy",
    "ResourceLimits",
    "Sandbox",
    "SandboxDiskExceeded",
    "SandboxError",
    "SandboxSpec",
    "SpecError",
    "SyscallPolicy",
    "Toolchain",
    "ToolchainError",
    "limits",
    "seccomp",
    "toolchain",
]
