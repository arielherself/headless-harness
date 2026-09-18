"""Run one command in a sandbox, or prove the sandbox is what it claims to be.

Two modes:

    python -m sandbox [options] -- command [args...]

creates a sandbox, runs the command, and exits with the command's status. It is
the shortest path to poking at the thing by hand — and the same call the
harness would make.

    python -m sandbox --self-test

runs a series of probes *inside* a fresh sandbox and reports, one line each,
which isolation properties actually hold on this machine. The kernel, the
cgroup delegation and the available bubblewrap differ enough between machines
that this is worth being able to check rather than assume; a probe that cannot
run is reported as skipped, and a probe that fails is reported with what
happened instead of a stack trace.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
from dataclasses import dataclass

from . import limits as limits_module
from . import seccomp as seccomp_module
from . import toolchain as toolchain_module
from .spec import Mount, NetworkPolicy, SpecError, SyscallPolicy
from .sandbox import Sandbox, SandboxError

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    critical: bool = True


def _check(name: str, status: str, detail: str, critical: bool = True) -> Check:
    return Check(name, status, detail, critical)


def _probe(sandbox: Sandbox, script: str, *, timeout: float | None = None):
    """Run a shell snippet, letting the caller read exit code and output.

    Probes stick to what the environment is guaranteed to have — the shell's own
    builtins and coreutils — and parse what they read in Python rather than
    reaching for `awk` or `grep`, which are their own packages and are often not
    in a minimal environment.
    """
    return sandbox.exec(["/bin/sh", "-c", script], timeout=timeout)


def _python_probe(sandbox: Sandbox, script: str, *, timeout: float | None = None):
    """Run a Python snippet. Used where a probe has to touch kernel state."""
    return sandbox.exec(["python3", "-c", script], timeout=timeout)


def _has_python(sandbox: Sandbox) -> bool:
    return _probe(sandbox, "command -v python3 >/dev/null && echo yes").stdout.strip() == "yes"


def _counter(text: str | None, key: str) -> int | None:
    """One number out of a cgroup accounting file (`oom_kill 3`, `max 5`)."""
    if not text:
        return None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == key:
            try:
                return int(fields[1])
            except ValueError:
                return None
    return None


# -- self-test --------------------------------------------------------------


def self_test(args: argparse.Namespace) -> int:
    env_dir, env_mounts, packages = _environment_choice(args, ("bash", "coreutils", "python3"))
    try:
        sandbox = Sandbox.create(
            packages=packages,
            files={"/input/data.json": '{"hello": "world"}\n'},
            writable=["/work"],
            mounts=env_mounts,
            network=False,
            resources={
                "memory": "256M",
                "cpu": 1.0,
                "pids": 32,
                "disk": "64M",
                "tmpfs_size": "16M",
                "timeout": 60,
            },
            env_dir=env_dir,
            limits_engine=args.limits_engine,
        )
    except (SandboxError, toolchain_module.ToolchainError) as exc:
        print(f"could not create a sandbox: {exc}", file=sys.stderr)
        return 2

    checks: list[Check] = []
    try:
        print(f"sandbox {sandbox.identity} at {sandbox.layout.root}")
        print(f"  {sandbox.toolchain.describe()}")
        print(f"  {sandbox.limiter.report.describe()}")
        print(f"  {seccomp_module.describe(sandbox.program)}")
        print()
        for probe in (
            _check_environment,
            _check_filesystem,
            _check_readonly,
            _check_workdir,
            _check_network,
            _check_pids,
            _check_memory,
            _check_cpu,
            _check_scratch_size,
            _check_syscalls,
            _check_timeout,
            _check_no_survivors,
        ):
            try:
                checks.append(probe(sandbox, args))
            except Exception as exc:  # a broken probe must not hide the others
                checks.append(_check(probe.__name__[7:], FAIL, f"probe raised {exc!r}", False))
    finally:
        sandbox.destroy()
    checks.append(_check_cleanup(sandbox))

    width = max(len(check.name) for check in checks)
    for check in checks:
        marker = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[check.status]
        print(f"  {marker}  {check.name:<{width}}  {check.detail}")
    failed = [check for check in checks if check.status == FAIL and check.critical]
    skipped = [check for check in checks if check.status == SKIP]
    passed = [check for check in checks if check.status == PASS]
    print()
    print(f"{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("critical checks failed; this machine's isolation is not what the model assumes")
    return 1 if failed else 0


def _check_environment(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    result = _probe(sandbox, "echo $PATH; command -v sh; command -v python3 || true")
    if result.exit_code != 0:
        return _check("environment", FAIL, result.summary())
    return _check("environment", PASS, f"PATH={result.stdout.splitlines()[0]}")


def _check_filesystem(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    home = os.path.expanduser("~")
    result = _probe(
        sandbox,
        f"test -e {home} && echo HOME_VISIBLE; test -e /etc/shadow && echo SHADOW_VISIBLE; "
        "ls -A / | tr '\\n' ' '; echo; ls -d /proc/[0-9]* | wc -l",
    )
    lines = result.stdout.splitlines()
    problems = [word for word in ("HOME_VISIBLE", "SHADOW_VISIBLE") if word in result.stdout]
    if problems:
        return _check("filesystem", FAIL, "the host is visible: " + ", ".join(problems))
    root = lines[0] if lines else "?"
    pids = lines[-1].strip() if len(lines) > 1 else "?"
    return _check("filesystem", PASS, f"/ holds [{root.strip()}], {pids} processes visible")


def _check_readonly(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    result = _probe(
        sandbox,
        "cat /input/data.json; echo x >> /input/data.json 2>&1 || echo WRITE_REFUSED",
    )
    if "hello" not in result.stdout:
        return _check("input read-only", FAIL, f"the input file is not readable: {result.summary()}")
    if "WRITE_REFUSED" not in result.stdout:
        return _check("input read-only", FAIL, "/input/data.json accepted a write")
    return _check("input read-only", PASS, "an input file reads but cannot be written")


def _check_workdir(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    result = _probe(sandbox, "cd /work && echo written > probe.txt && cat probe.txt")
    if result.stdout.strip() != "written":
        return _check("work writable", FAIL, result.summary())
    sandbox.put_file("/work/from_harness.txt", "handed in\n")
    result = _probe(sandbox, "cat /work/from_harness.txt")
    if result.stdout.strip() != "handed in":
        return _check("work writable", FAIL, "put_file did not reach the sandbox")
    sandbox.get_file("/work/probe.txt")
    return _check("work writable", PASS, "the sandbox writes and the harness reads")


def _check_network(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    if not sandbox.spec.network.isolated:
        return _check("network isolated", SKIP, "the spec asked for the host network")
    listing = _probe(sandbox, "cat /proc/net/dev")
    # the first two lines are headings; an interface line is `name: numbers...`
    names = [
        line.split(":")[0].strip()
        for line in listing.stdout.splitlines()[2:]
        if ":" in line
    ]
    names = [name for name in names if name]
    if not names:
        return _check("network isolated", SKIP, f"no interface list to read: {listing.summary(120)}")
    extra = [name for name in names if name != "lo"]
    if extra:
        return _check("network isolated", FAIL, f"interfaces beyond lo: {', '.join(extra)}")
    dial = _probe(
        sandbox,
        "bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null && echo REACHED || echo UNREACHABLE",
    )
    if "REACHED" in dial.stdout:
        return _check("network isolated", FAIL, "a TCP connection to 1.1.1.1:443 succeeded")
    return _check("network isolated", PASS, "only lo exists; a TCP dial is refused")


FORK_STORM = """import os, sys, time
count = 0
try:
    while count < 128:
        pid = os.fork()
        if pid == 0:
            time.sleep(30)
            os._exit(0)
        count += 1
    print(count, "none")
except OSError as exc:
    print(count, exc.errno)
sys.stdout.flush()
"""

# far more forks than any sane pids.max for one harness command; if the limit
# is missing the probe stops here and reports that instead of forking forever
FORK_STORM_CAP = 128


def _check_pids(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    """Fork until the kernel says no, and ask the kernel how often it said no.

    A shell loop is the wrong tool: bash retries a fork that fails with EAGAIN,
    so a sandbox that is working correctly looks like a sandbox that hangs.
    Python's `os.fork` raises, which turns the limit into a number.
    """
    limit = sandbox.spec.resources.pids or 0
    if not sandbox.limiter.report.enforced:
        return _check(
            "pids limit",
            SKIP,
            f"the {sandbox.limiter.report.engine} engine cannot enforce pids.max ({limit})",
        )
    if not _has_python(sandbox):
        return _check("pids limit", SKIP, "no python3 in the environment to fork from")
    limiter = sandbox.limiter
    before = limiter.read("pids.events") if isinstance(limiter, limits_module.CgroupLimiter) else None
    result = _python_probe(sandbox, FORK_STORM, timeout=30)
    fields = result.stdout.split()
    if result.timed_out:
        return _check("pids limit", FAIL, "the fork storm never finished")
    if len(fields) < 2 or not fields[0].isdigit():
        return _check("pids limit", SKIP, f"could not count forks: {result.summary(120)}")
    count, errno_text = int(fields[0]), fields[1]
    after = limiter.read("pids.events") if isinstance(limiter, limits_module.CgroupLimiter) else None
    refusals = None
    if before is not None and after is not None:
        delta = _counter(after, "max")
        first = _counter(before, "max")
        if delta is not None and first is not None:
            refusals = delta - first
    evidence = f", the kernel refused {refusals} more forks" if refusals else ""
    if count >= FORK_STORM_CAP:
        return _check(
            "pids limit", FAIL, f"{count} forks succeeded under a pids.max of {limit}{evidence}"
        )
    if count == 0:
        return _check(
            "pids limit",
            FAIL,
            f"not one fork succeeded under a pids.max of {limit}; nothing was measured",
        )
    if errno_text == "none":
        return _check(
            "pids limit", FAIL, f"the fork storm was never refused under a pids.max of {limit}"
        )
    return _check(
        "pids limit",
        PASS,
        f"{count} forks before EAGAIN under a pids.max of {limit}{evidence}",
    )


MEMORY_TOUCH = """import ctypes
n = {bytes}
buf = ctypes.create_string_buffer(n)
ctypes.memset(buf, 65, n)
print("touched", n)
"""


def _check_memory(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    """Ask for more memory than the cap and *write to it*.

    Reserving address space proves nothing to a memory cgroup: pages are charged
    when they are first touched, so a probe that only allocates can succeed under
    a cap it should never fit in. `memset` faults every page in.
    """
    limit = sandbox.spec.resources.memory or 0
    mib = limit // (1024 * 1024)
    engine = sandbox.limiter.report.engine
    if not _has_python(sandbox):
        return _check("memory limit", SKIP, "no python3 in the environment to touch memory with")
    limiter = sandbox.limiter
    before = limiter.read("memory.events") if isinstance(limiter, limits_module.CgroupLimiter) else None
    result = _python_probe(sandbox, MEMORY_TOUCH.format(bytes=mib * 4 * 1024 * 1024), timeout=60)
    after = limiter.read("memory.events") if isinstance(limiter, limits_module.CgroupLimiter) else None
    evidence = ""
    if after:
        killed = _counter(after, "oom_kill")
        hit = _counter(after, "max")
        if killed:
            evidence = f", the kernel OOM-killed it ({killed} in this cgroup)"
        elif hit:
            evidence = f", the kernel refused the charge ({hit} times in this cgroup)"
    if result.timed_out:
        return _check("memory limit", FAIL, "the allocation hung rather than failing")
    if result.exit_code == 0:
        return _check(
            "memory limit",
            FAIL,
            f"writing {mib * 4} MiB succeeded under a {mib} MiB cap ({engine})",
        )
    return _check(
        "memory limit",
        PASS,
        f"{mib * 4} MiB refused under a {mib} MiB cap "
        f"({engine}: {_last_line(result.stderr or result.stdout)}{evidence})",
    )


def _last_line(text: str) -> str:
    """The most informative line of a failure: an error, or what came last."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if "error" in line.lower() or "refused" in line.lower():
            return line[:90]
    return lines[-1][:90] if lines else "no output"


def _check_cpu(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    quota = sandbox.spec.resources.cpu
    limiter = sandbox.limiter
    if not isinstance(limiter, limits_module.CgroupLimiter):
        return _check("cpu limit", SKIP, "no cgroup to read cpu accounting from")
    before = limiter.read("cpu.stat") or ""
    result = _probe(
        sandbox,
        "for i in 1 2 3 4; do (while :; do :; done) & done; sleep 3",
        timeout=30,
    )
    if result.timed_out or result.exit_code != 0:
        return _check("cpu limit", FAIL, result.summary())
    usage = _usage_usec(limiter.read("cpu.stat") or "") - _usage_usec(before)
    burned = usage / 1_000_000
    budget = 3.0 * (quota or 1.0)
    if burned > budget * 2:
        return _check(
            "cpu limit", FAIL, f"4 busy loops burned {burned:.1f}s of cpu in 3s at quota {quota}"
        )
    if burned < budget * 0.5:
        # four loops over three seconds cannot use *less* than the quota unless
        # they were never in this cgroup, in which case nothing was measured
        return _check(
            "cpu limit",
            FAIL,
            f"only {burned:.1f}s of cpu was accounted to this cgroup; the load did not run in it",
        )
    return _check("cpu limit", PASS, f"4 busy loops burned {burned:.1f}s of cpu in 3s at quota {quota}")


def _usage_usec(text: str) -> int:
    for line in text.splitlines():
        if line.startswith("usage_usec"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return 0
    return 0


def _check_scratch_size(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    size = sandbox.spec.resources.tmpfs_size
    mib = max(1, size // (1024 * 1024))
    result = _probe(
        sandbox,
        f"dd if=/dev/zero of=/tmp/big bs=1M count={mib + 8} 2>&1",
        timeout=60,
    )
    if result.exit_code == 0:
        return _check("scratch size", FAIL, f"wrote {mib + 8} MiB into a {mib} MiB tmpfs")
    return _check(
        "scratch size",
        PASS,
        f"a {mib} MiB tmpfs refused a {mib + 8} MiB write ({_last_line(result.stderr or result.stdout)})",
    )


def _check_syscalls(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    if sandbox.program is None:
        return _check("seccomp", SKIP, "the syscall filter is off")
    if not _has_python(sandbox):
        return _check("seccomp", SKIP, "no python3 in the environment to make a raw syscall")
    try:
        arch = seccomp_module.machine_arch()
    except Exception as exc:
        return _check("seccomp", SKIP, str(exc))
    # keyctl is callable by anyone and is on the denylist, so it makes a clean
    # probe: with the filter it is EPERM, without it the call gets further
    number = seccomp_module.syscall_number("keyctl", arch)
    if number is None:
        return _check("seccomp", SKIP, "no keyctl on this architecture")
    script = (
        "import ctypes; libc = ctypes.CDLL(None, use_errno=True); "
        f"libc.syscall({number}, 0, -2, 1, 0, 0); "
        "print(ctypes.get_errno())"
    )
    filtered = _python_probe(sandbox, script)
    errno_text = filtered.stdout.strip().splitlines()[-1] if filtered.stdout.strip() else "?"
    if errno_text != "1":
        return _check("seccomp", FAIL, f"a denied syscall returned errno {errno_text}, not EPERM(1)")
    control = Sandbox.create(
        env_dir=sandbox.spec.env_dir,
        packages=sandbox.spec.packages,
        mounts=sandbox.spec.mounts,
        writable=["/work"],
        resources={"timeout": 30},
        syscalls=SyscallPolicy(enabled=False),
        nixpkgs=sandbox.spec.nixpkgs,
        limits_engine=args.limits_engine,
    )
    try:
        unfiltered = _python_probe(control, script)
        control_errno = (
            unfiltered.stdout.strip().splitlines()[-1] if unfiltered.stdout.strip() else "?"
        )
    finally:
        control.destroy()
    if control_errno == "1":
        return _check(
            "seccomp", SKIP, "keyctl returns EPERM here without a filter too; not decisive"
        )
    return _check(
        "seccomp", PASS, f"a denied syscall is EPERM under the filter, errno {control_errno} without it"
    )


def _check_timeout(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    result = sandbox.exec(["/bin/sh", "-c", "sleep 30"], timeout=1.5)
    if not result.timed_out:
        return _check("timeout", FAIL, f"a 30s sleep returned in {result.duration:.1f}s: {result.summary()}")
    if result.duration > 10:
        return _check("timeout", FAIL, f"the watchdog took {result.duration:.1f}s to fire")
    return _check("timeout", PASS, f"a 30s sleep was killed after {result.duration:.1f}s")


def _check_no_survivors(sandbox: Sandbox, args: argparse.Namespace) -> Check:
    """Background work from an earlier command must not be here now.

    The count of processes is not the question — a fresh command legitimately
    has a handful — so this asks specifically about the processes an earlier
    probe left behind, and parses them here rather than in the sandbox.
    """
    listing = _probe(sandbox, "cat /proc/[0-9]*/comm 2>/dev/null")
    survivors = [line.strip() for line in listing.stdout.splitlines() if line.strip() == "sleep"]
    if survivors:
        return _check(
            "no survivors", FAIL, f"{len(survivors)} background processes outlived their command"
        )
    return _check("no survivors", PASS, "background work from an earlier command is gone")


def _check_cleanup(sandbox: Sandbox) -> Check:
    if os.environ.get("HH_SANDBOX_KEEP"):
        return _check("cleanup", SKIP, "HH_SANDBOX_KEEP is set, so the state is kept on purpose")
    if sandbox.layout.root.exists():
        return _check("cleanup", FAIL, f"{sandbox.layout.root} survived destroy()")
    if isinstance(sandbox.limiter, limits_module.CgroupLimiter):
        return _check("cleanup", PASS, "state directory and cgroup both released")
    return _check("cleanup", PASS, "state directory released")


# -- one command ------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    if not args.command:
        print("no command given; see --help", file=sys.stderr)
        return 2
    env_dir, env_mounts, packages = _environment_choice(args, ("bash", "coreutils"))
    spec_kwargs = {
        "packages": packages,
        "files": {},
        "writable": [],
        "mounts": list(env_mounts),
        "network": NetworkPolicy.host() if args.network else NetworkPolicy.none(),
        "resources": {},
        "env_dir": env_dir,
        "limits_engine": args.limits_engine,
    }
    if args.no_seccomp:
        spec_kwargs["syscalls"] = SyscallPolicy(enabled=False)
    resources = spec_kwargs["resources"]
    for key in ("memory", "cpu", "pids", "disk", "tmpfs_size"):
        value = getattr(args, key)
        if value is not None:
            resources[key] = value
    mounts = spec_kwargs["mounts"]
    writable = spec_kwargs["writable"]
    if args.work:
        mounts.append(Mount.rw_bind(os.path.abspath(args.work), "/work"))
    else:
        writable.append("/work")
    for entry in args.input or []:
        source, _, destination = entry.partition(":")
        source = os.path.abspath(source)
        if not os.path.exists(source):
            print(f"no such input: {source}", file=sys.stderr)
            return 2
        if not destination:
            destination = os.path.join("/input", os.path.basename(source)) if os.path.isfile(source) else "/input"
        mounts.append(Mount.ro_bind(source, destination))
    for entry in args.scratch or []:
        path, _, size = entry.partition(":")
        mounts.append(Mount.tmpfs(path, size or "64M"))

    try:
        sandbox = Sandbox.create(**spec_kwargs)
    except (SandboxError, toolchain_module.ToolchainError, NotImplementedError, SpecError) as exc:
        print(f"could not create a sandbox: {exc}", file=sys.stderr)
        return 2
    if args.verbose:
        print(json.dumps(sandbox.describe(), indent=2), file=sys.stderr)
    try:
        result = sandbox.exec(args.command, timeout=args.timeout if args.timeout is not None else 60)
    finally:
        # `--keep` keeps the state directory, not the cgroup: whatever the
        # command left running is killed either way
        sandbox.destroy(keep=args.keep)
        if args.keep:
            print(f"kept {sandbox.layout.root}", file=sys.stderr)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.truncated:
        print("[output truncated]", file=sys.stderr)
    if result.timed_out:
        print(f"[timed out after {result.duration:.1f}s]", file=sys.stderr)
    for warning in sandbox.warnings:
        print(f"[sandbox] {warning}", file=sys.stderr)
    return result.exit_code


def _environment_choice(
    args: argparse.Namespace, default_packages: tuple[str, ...]
) -> tuple[str | None, list[Mount], tuple[str, ...]]:
    """`(env_dir, extra_mounts, packages)` for the environment the flags asked for.

    `--host-tools` exists so this package can be exercised on a machine without
    Nix. It is weaker than a store closure by construction: the binaries come
    from the host, and so do the mounts the environment needs to run them. Its
    temporary directory is removed when the process exits (unless `--keep`).
    """
    if args.host_tools:
        toolchain = toolchain_module.host_toolchain()
        if not args.keep:
            atexit.register(shutil.rmtree, toolchain.path, ignore_errors=True)
        return toolchain.path, list(toolchain.mounts), ()
    if args.env_dir:
        # an environment directory is the whole environment: asking Nix for
        # packages as well would be a contradiction, not a merge
        return args.env_dir, [], ()
    return None, [], tuple(args.packages or default_packages)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sandbox",
        description="Run a command in a Nix + bubblewrap + cgroup v2 sandbox.",
    )
    parser.add_argument("--self-test", action="store_true", help="probe the isolation this machine provides")
    parser.add_argument("-p", "--packages", action="append", help="Nix package (repeatable)")
    parser.add_argument("--env-dir", help="use an existing environment directory instead of Nix")
    parser.add_argument(
        "--host-tools",
        action="store_true",
        help="build the environment from the host's binaries (debugging, weaker)",
    )
    parser.add_argument("--work", help="host directory to expose read-write at /work")
    parser.add_argument("--input", action="append", metavar="SRC[:DEST]", help="read-only input (repeatable)")
    parser.add_argument("--scratch", action="append", metavar="PATH[:SIZE]", help="size-capped tmpfs (repeatable)")
    parser.add_argument("--network", action="store_true", help="share the host network namespace")
    parser.add_argument("--memory", help="memory cap, e.g. 512M")
    parser.add_argument("--cpu", type=float, help="cpu cap in cores, e.g. 1.0")
    parser.add_argument("--pids", type=int, help="process cap")
    parser.add_argument("--disk", help="cap on the sandbox's writable state, e.g. 1G")
    parser.add_argument("--tmpfs-size", dest="tmpfs_size", help="size of scratch tmpfs mounts")
    parser.add_argument("--timeout", type=float, help="wall-clock seconds per command")
    parser.add_argument(
        "--limits-engine",
        choices=("auto", "cgroup", "systemd", "rlimit", "none"),
        default="auto",
        help="which mechanism enforces the resource limits",
    )
    parser.add_argument("--no-seccomp", action="store_true", help="run without a syscall filter")
    parser.add_argument("--keep", action="store_true", help="keep the state directory after the run")
    parser.add_argument("-v", "--verbose", action="store_true", help="describe the sandbox on stderr")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if args.self_test:
        return self_test(args)
    if not args.command:
        parser.print_help()
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
