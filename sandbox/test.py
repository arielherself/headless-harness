#!/usr/bin/env python3
"""An interactive bash inside a sandbox.

The shell runs in the sandbox's namespaces: the host's files are not visible,
the network namespace has only loopback, and memory, CPU, processes and disk
are capped by the spec. Everything done in the shell — `cd`, exported
variables, background jobs, files under `/work` and `/tmp` — lives as long as
the shell does, because the session is one bubblewrap process holding its
namespaces, with the shell as its child.

    python sandbox/test.py                      # bash, no network
    python sandbox/test.py --network            # the host's network namespace
    python sandbox/test.py -p git -p ripgrep    # extra Nix packages
    python sandbox/test.py --dir ~/src          # a host directory at /work
    python sandbox/test.py --host-tools         # no Nix: the host's binaries

`exit` (or Ctrl-D) ends the shell and destroys the sandbox. A session has no
watchdog: the spec's `resources.timeout` applies to `exec` commands, and this
shell is as long as it needs to be.
"""

from __future__ import annotations

import argparse
import atexit
import shutil
import sys
from pathlib import Path

# running this file directly (`python sandbox/test.py`) puts `sandbox/` itself on
# sys.path, where there is no `sandbox` package to import, so the parent — the
# directory that holds the package — goes first. `python -m sandbox.test` needs
# none of this.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sandbox import Mount, Sandbox, SandboxError, SpecError, ToolchainError
from sandbox import toolchain as toolchain_module

# The shell has no host dotfiles to read, so give it a prompt that says where it
# is; `files=` stages this into the sandbox's own home directory.
BASH_RC = """\
# the sandbox's bashrc: the host's dotfiles are not in here
PS1='(sandbox) \\w \\$ '
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python sandbox/test.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p", "--package", action="append", default=[], metavar="NAME",
        help="a Nix package to add to the environment (repeatable)",
    )
    parser.add_argument(
        "--host-tools", action="store_true",
        help="build the environment from the host's own binaries instead of Nix",
    )
    parser.add_argument(
        "--dir", metavar="PATH",
        help="bind this host directory at /work and start there instead of a private one",
    )
    parser.add_argument(
        "--network", action="store_true",
        help="share the host's network namespace instead of an isolated one",
    )
    parser.add_argument("--memory", default="512M", metavar="SIZE", help="memory cap (512M)")
    parser.add_argument("--cpu", type=float, default=1.0, metavar="N", help="CPU quota (1.0)")
    parser.add_argument("--pids", type=int, default=256, metavar="N", help="process cap (256)")
    parser.add_argument("--disk", default="512M", metavar="SIZE", help="disk budget (512M)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.host_tools and args.package:
        parser.error("--host-tools builds the environment from the host, so --package does not apply")

    tools = None
    if args.host_tools:
        tools = toolchain_module.host_toolchain()
        # the temporary environment is a debugging aid, not something to keep
        atexit.register(shutil.rmtree, tools.path, ignore_errors=True)

    writable: list[str | Mount] = []
    if args.dir:
        source = Path(args.dir).expanduser()
        if not source.is_dir():
            parser.error(f"--dir is not a directory: {source}")
        writable.append(Mount.rw_bind(str(source.resolve()), "/work"))
    else:
        writable.append("/work")

    spec: dict[str, object] = {
        "files": {"/home/user/.bashrc": BASH_RC},
        "writable": writable,
        "cwd": "/work",
        "network": "host" if args.network else False,
        "resources": {
            "memory": args.memory,
            "cpu": args.cpu,
            "pids": args.pids,
            "disk": args.disk,
        },
    }
    if tools is not None:
        spec["env_dir"] = tools.path
        spec["mounts"] = list(tools.mounts)
    else:
        spec["packages"] = tuple(args.package)

    try:
        sandbox = Sandbox.create(**spec)  # type: ignore[arg-type]
    except (SandboxError, ToolchainError, SpecError, NotImplementedError) as exc:
        print(f"could not create a sandbox: {exc}", file=sys.stderr)
        return 2

    described = sandbox.describe()
    print(f"sandbox {described['id']}: {described['limits']}", file=sys.stderr)
    if described["network"] == "none":
        print("network: loopback only (--network shares the host's)", file=sys.stderr)
    for warning in sandbox.warnings:
        print(f"sandbox: {warning}", file=sys.stderr)

    try:
        session = sandbox.open_session(["bash", "-i"])
    except SandboxError as exc:
        sandbox.destroy()
        print(f"could not start a shell: {exc}", file=sys.stderr)
        return 2
    try:
        code = session.interact()
    except KeyboardInterrupt:
        code = 130
    finally:
        sandbox.destroy()
    if code:
        print(f"sandbox: the shell exited with status {code}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
