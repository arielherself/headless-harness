# `sandbox` — Nix + bubblewrap + cgroup v2 + seccomp

A sandbox for running commands the harness does not trust, with the shape the
design calls for:

```python
from sandbox import Sandbox

sandbox = Sandbox.create(
    packages=["python312", "git", "ripgrep"],
    files={"/work/main.py": code, "/input/data.json": data},
    writable=["/work"],
    network=False,
    resources={"memory": "512M", "cpu": 1.0, "pids": 64,
               "disk": "1G", "timeout": 30},
)
result = sandbox.exec(["python", "/work/main.py"])
sandbox.destroy()
```

It is not wired into the harness yet, and nothing in it imports the harness.

## Layers

```
Harness
   │  Sandbox.create(spec)                     one spec → one sandbox
   ▼
cgroup v2            memory.max, memory.swap.max, cpu.max, pids.max, io.max
   │                 applied before the sandbox can fork, via a PID handshake
   ▼
bubblewrap           user, pid, mount, network, ipc, uts, cgroup namespaces
   │                 a private root tmpfs; the Nix store read-only; scratch
   │                 tmpfs sized; --disable-userns; all capabilities dropped
   ▼
seccomp              cBPF denylist, SECCOMP_RET_ERRNO(EPERM), arch-checked
   ▼
Nix environment      one buildEnv store path with exactly the packages asked for
                     (plus bash and coreutils, so `#!/bin/sh` resolves)
```

There is no `chroot` anywhere in the implementation. `chroot` only changes path
resolution, so it is not an isolation boundary; the root filesystem here is a
private tmpfs with things bind-mounted into it by bubblewrap.

## Running it by hand

```bash
# talk to a sandbox directly: Nix builds the environment, then the command runs
python -m sandbox -p python3 -p git --work ./scratch -- python3 /work/main.py

# an interactive bash inside the sandbox: its own pty, so line editing, Ctrl-C
# and job control work, and the shell's state lives as long as the session does
python sandbox/test.py
python sandbox/test.py --host-tools --dir ~/scratch

# no Nix on this machine? build the environment from the host's own binaries
python -m sandbox --host-tools --work ./scratch -- sh -c 'id; ls /'

# ask the machine what it actually enforces
python -m sandbox --self-test
```

`--self-test` builds one sandbox and probes it from the inside: the host home
directory, `/etc/shadow`, writes to read-only mounts, the interface list, a TCP
dial, a fork storm against `pids.max`, an oversized allocation against
`memory.max`, a tmpfs that fills up, a raw syscall against the filter, a command
that overruns its watchdog, and whether anything survives its own command. Each
line is `ok`, `FAIL` or `skip`, and the skips say why (an engine that cannot
enforce something does not get to claim it).

Two of those probes have a subtlety worth knowing, because getting them wrong is
how a self-test lies. The memory probe *writes to* the memory it allocates: a
cgroup charges pages when they are first touched, so merely allocating — which
is all `bytearray(1 << 30)` may do — can succeed under a cap it should never fit
in. The fork storm forks from Python rather than a shell, because `bash` retries
a fork that fails with `EAGAIN`, which turns "the limit is working" into "the
command hangs". Both report the kernel's own counters (`memory.events`,
`pids.events`) when a cgroup is in use. Probes also stick to the environment's
guaranteed tools — shell builtins, coreutils, `python3` — and parse what they
read on the harness side, since `awk` and `grep` are packages of their own and a
minimal environment does not have them.

Flags worth knowing: `--input SRC[:DEST]` (read-only), `--scratch PATH[:SIZE]`
(size-capped tmpfs), `--network` (share the host's namespace), `--disk`,
`--memory`, `--cpu`, `--pids`, `--timeout`, `--limits-engine`, `--no-seccomp`,
`--keep` (keep the state directory for inspection), `-v` (print the whole plan).

## What is enforced, and how

| You asked for               | Mechanism                              | Hard? |
|-----------------------------|----------------------------------------|-------|
| host files invisible        | mount namespace, no host binds         | yes   |
| a path read-only            | `--ro-bind`                            | yes   |
| a path writable             | `--bind` to a directory under the state | yes  |
| no network                  | network namespace; only `lo` exists    | yes   |
| memory                      | `memory.max` (+ `memory.swap.max=0`)   | yes   |
| cpu                         | `cpu.max` (quota per 100 ms period)    | yes   |
| process count               | `pids.max`                             | yes   |
| disk throughput             | `io.max`                               | yes   |
| scratch space               | tmpfs `--size`; `/` itself is read-only | yes  |
| the sandbox's own files     | measured between commands              | no    |
| syscalls                    | seccomp denylist, `EPERM`              | yes   |
| wall-clock time             | watchdog, then SIGKILL of the tree     | yes   |
| the software environment    | Nix store path                         | yes   |

One row is soft. `resources.disk` bounds the *tree* of writable mounts that the
sandbox owns, and no portable unprivileged mechanism caps a directory's size, so
it is measured after each command and the sandbox refuses to run once the budget
is spent. `RLIMIT_FSIZE` is set to the same number, which does make the kernel
refuse any single file past it. A host directory *you* supplied is not measured —
it may have been full before the sandbox existed — and `Sandbox.warnings` says so
when one is used with a budget. When a hard cap matters more than seeing the
files from outside, declare the hot path as a `tmpfs` mount.

## Lifetime: commands get fresh namespaces

Every `exec` starts a new bubblewrap, so it gets new namespaces: a command cannot
see processes from an earlier command, and background work from one command is
gone before the next one starts. That is deliberate — it is the simplest model
that cannot leak state — and it has two consequences worth knowing:

* **host-backed writable paths persist** (`/work` in the example), because they
  are directories under the sandbox's state root, reachable by both sides;
* **`tmpfs` mounts do not.** A scratch tmpfs is created for a command, used, and
  thrown away with its namespaces. `put_file` and `get_file` refuse a tmpfs path
  rather than hand back a file that will not be there.

### Sessions: one process that keeps the namespaces

`Sandbox.open_session(command)` is the other model, for the one case where a
single long-lived process is the point — an interactive shell, a REPL, a
language server:

```python
with sandbox.open_session(["bash", "-i"]) as session:
    session.interact()          # or write() / read() / wait()
```

One bubblewrap process holds the namespaces for as long as the command runs, and
the command is the parent of everything done in it, so the working directory,
the environment, background jobs and `tmpfs` mounts persist from one line to the
next. The command runs on a pty, for the same reason a shell wants one: line
editing, Ctrl-C, `fg`/`bg` and window resizing need a controlling terminal. The
pty is the sandbox's own — none of the harness's terminal is passed in — which
is why a session's argv leaves out `--new-session`: that flag detaches a sandbox
from its caller's terminal, and here the terminal is the sandbox's to keep.

Two differences from `exec` are worth stating. A session has no watchdog —
`resources.timeout` applies to commands, and a session is as long as the process
is — and `exec` refuses while one is open, because the namespaces belong to the
session's process until it ends. `Session.close()` ends it (SIGHUP, then SIGKILL
and `cgroup.kill` if it does not take the hint), and `destroy()` ends it too.

A command outside the session cannot be run in those namespaces either: a
seccomp filter is inherited from the process that installed it, not by a process
that `setns`-es in later, so anything that should share the sandbox has to be a
child of the sandboxed process. That is exactly what typing into the shell does,
and why `interact()` moves bytes between two terminals rather than handing over
a file descriptor.

## Resource limits: which engine

`Sandbox.create(limits_engine=...)` or `HH_SANDBOX_CGROUP` picks where the
limits come from. In `auto` (the default) the ladder is:

1. **cgroup v2**, a group of the sandbox's own under a delegated base — found by
   walking up from this process's own cgroup until a writable group appears that
   can hold the controllers the spec asked for. This is the only engine that gets
   a hard `pids.max`, `io.max` and an atomic `cgroup.kill` on destroy.
2. **systemd-run --user --scope**, when there is a user manager to ask. The same
   cgroup underneath, with the limits passed as unit properties.
3. **rlimits**, through `prlimit`: `RLIMIT_FSIZE` for the disk budget, and
   `RLIMIT_AS` for memory (address space, not RSS — a runtime that reserves more
   than it uses may fail, which is why it is only used when there is no cgroup).
   CPU and PID limits are **not** enforced by this engine, and the report says so.

`sandbox.describe()["limits"]` holds the engine and the values that were
actually written, `sandbox.limiter.directory` is the cgroup when there is one,
and `sandbox.warnings` lists what could not be applied.

The exact enforcement is worth as much as the ladder:

* every engine is applied through the same handshake — bubblewrap starts with
  `--info-fd` and `--block-fd`, reports the real PID of the process that will
  become the sandbox, and holds it there until that PID has been placed in its
  cgroup. Nothing can escape by forking first, and if the handshake fails the
  command is refused rather than run unlimited;
* on destroy, the cgroup is killed (`cgroup.kill`) and removed. Commands already
  kill the whole process group, and `--die-with-parent` covers whatever left it.

## The filesystem inside

```
/            private tmpfs, remounted read-only once everything is mounted
/nix/store   read-only (the toolchain's closure lives here)
/bin /usr/bin  symlinks to the environment's bin (or the host's, with --host-tools)
/etc/passwd /etc/group /etc/hosts   generated; no host identity
/etc/resolv.conf /etc/ssl ...        the host's, but only when the network is shared
/proc        a private instance for the PID namespace
/dev         bubblewrap's minimal device set; /dev/shm is a sized tmpfs
/tmp         sized tmpfs (per command; per session, in a session)
/work /home/user / whatever `writable` names   host directories, read-write
read-only files from `files=`   bound in one file at a time
```

The root is read-only because nothing a sandbox is meant to write to lives there,
and because the root tmpfs has no size of its own. `read_only_root=False` in the
spec turns that off for a workload that writes to `/` itself.

The built-in mounts come first in the argument list, so a mount the spec declares
for the same path (`/dev/shm`, `/proc`) lands on top of it instead of being
silently overridden.

`files={...}` decides by where the path is: under a writable mount it is written
there and the sandbox may change it; anywhere else it is staged on the host and
bound read-only. A file whose content starts with `#!` is made executable.

`/etc` is never mounted as a directory. bubblewrap creates a mount point before
it mounts over it, so a `/etc` that arrived as a read-only bind would refuse
every later file bind with `Read-only file system`; the sandbox's own files are
bound one path at a time instead, and the host's `/etc/ssl` lands on top.

## Syscalls

A denylist with `SECCOMP_RET_ALLOW` as the default, `SECCOMP_RET_ERRNO(EPERM)`
for what is denied, an architecture check that kills the process if the filter
does not match the running ABI, and the x32 guard on x86_64. The list is in
`sandbox/seccomp.py` and covers four groups: editing the mount table, reaching
into another process, kernel surfaces that have produced escapes (modules,
`bpf`, `perf_event_open`, `userfaultfd`), and machine-level state (reboot,
swap, the clock, the keyring, the kernel log).

It is deliberately short, because compilers, Node and browser engines all use
syscalls that sound dangerous in isolation; `unshare`, `setns`, `clone` and
`io_uring` are all still allowed, and the reason they do not matter is
`--disable-userns` plus a dropped capability set. `SyscallPolicy(allow=(...))`
removes an entry, `deny=(...)` adds one by name or number, and names that do not
exist on the running architecture are reported instead of guessed. An unknown
architecture refuses to build a filter at all (a filter with the wrong ABI's
numbers is worse than no filter).

## Network

`NetworkPolicy.none()` (the default) is a network namespace with only loopback:
nothing to route to, nothing to resolve with, and abstract sockets are namespaced
along with it. `NetworkPolicy.host()` shares the harness's namespace and is the
"this is not a boundary" escape hatch — when it is used, the host's resolver and
CA store are mounted in too, and `SSL_CERT_FILE`/`NIX_SSL_CERT_FILE` are set.

`NetworkPolicy.allowing("pypi.org", ...)` is the shape the design wants for
dynamic authorization, and it is **not implemented**: a domain allow-list needs
a CONNECT proxy outside the network namespace to mean anything, and
`Sandbox.create` raises `NotImplementedError` rather than quietly treating it as
`none`. The policy type exists so callers can declare intent now.

## Requirements

* Linux with user namespaces enabled (`kernel.unprivileged_userns_clone=1`, or a
  kernel that does not need it) and a kernel with cgroup v2 for the hard limits.
* `bwrap` — checked at create time against the options actually used, so an old
  bubblewrap fails with a list of what it lacks rather than a confusing error.
* Nix with a `<nixpkgs>` in `NIX_PATH` (or `nixpkgs=` in the spec), unless the
  caller supplies `env_dir=` or `--host-tools`.
* For the cgroup engine: a delegated, writable cgroup — a normal systemd user
  session provides one. `HH_SANDBOX_CGROUP` overrides where to look, which is
  also how the tests exercise that code path.

The environment variables the package reads, all optional:

| Variable | Meaning |
|---|---|
| `HH_SANDBOX_ROOT` | where sandbox state directories are created (default `/tmp/headless-harness-sandboxes`) |
| `HH_SANDBOX_KEEP` | keep the state directory on destroy, as `destroy(keep=True)` does |
| `HH_SANDBOX_CGROUP` | the delegated cgroup base to create sandboxes under |

No daemon, no images, no root: creating a sandbox is a directory, a Nix store
path and a cgroup; running a command is one `bwrap` process.

## Testing

```bash
python -m unittest tests.test_sandbox -v      # 89 tests
python -m sandbox --self-test                 # what this machine really enforces
```

The unit tests need nothing (spec parsing, the BPF program, the argv, the cgroup
bookkeeping, the toolchain expression). The integration tests build real
sandboxes from the host's own binaries — so they need no Nix and no network — and
are skipped only where bubblewrap or user namespaces are missing. The cgroup
engine's *enforcement* is the one thing they cannot prove on a machine whose
cgroup tree is read-only; `--self-test` on a real session is what proves that.
