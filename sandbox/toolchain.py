"""The software environment inside the sandbox, built by Nix.

The point of using Nix here is that the environment is a *store path*: an
immutable directory of symlinks into a closure Nix already built, shared between
every sandbox that asks for the same packages. There is no image to unpack and
no root filesystem to assemble — `/nix/store` is mounted read-only and `PATH`
points at one `bin/` directory inside it.

`buildEnv` is asked for the requested packages plus `bash` and `coreutils`,
because a sandbox whose `/bin/sh` does not resolve is not a sandbox anyone can
do anything in. The store path for a given package list never changes, so
resolution is cached in-process and, best effort, in a JSON file under the
user's cache directory; `resolve_packages` is the entry point a created sandbox
uses to ask for a different list, which is how packages are added or removed
after `Sandbox.create`.

Nothing here runs sandboxed: `nix-build` is invoked the way the operator's Nix
is configured, which is also how the closure lands in the store.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .spec import Mount, SandboxSpec

# A first build may have to download a closure; later ones are instant.
BUILD_TIMEOUT = 900.0
# Anything a caller may name as a package: a Nix attribute path, or an absolute
# store path used as a literal. The check exists so a name cannot smuggle Nix
# code into the generated expression.
_ATTRIBUTE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'-]*(?:\.[A-Za-z_][A-Za-z0-9_'-]*)*$")

# Every environment gets these, whether or not they were asked for: without a
# shell at `/bin/sh` and an `env` at `/usr/bin/env`, other people's scripts do
# not run. Both are already in any closure that has a compiler.
BASE_PACKAGES = ("bash", "coreutils")

# Tools the host-toolchain fallback links in. It exists so the package can be
# exercised on a machine where Nix is unavailable; it is not an isolation
# feature, and the binaries it exposes come from the host, which is exactly what
# a Nix environment is meant to avoid.
HOST_TOOLS = (
    "sh", "bash", "cat", "ls", "pwd", "mkdir", "rm", "cp", "mv", "ln",
    "chmod", "echo", "env", "sed", "grep", "head", "tail", "wc", "seq",
    "sleep", "dd", "sync", "touch", "true", "false", "id", "uname",
    "date", "du", "df", "find", "sort", "cut", "tr", "test", "timeout",
    "python3",
)


class ToolchainError(RuntimeError):
    """Nix could not produce the requested environment."""


@dataclass(frozen=True)
class Toolchain:
    """A directory with a `bin/` that the sandbox will have on `PATH`."""

    path: str
    packages: tuple[str, ...] = ()
    origin: str = "nix"
    # extra read-only mounts the environment needs in order to work at all: a
    # Nix closure needs none (the store is always mounted), the host-tools
    # fallback needs the host's libraries
    mounts: tuple[Mount, ...] = ()
    # the paths a script expects to find a shell at; both are pointed at the
    # environment's `bin/`, which is what makes `#!/bin/sh` work
    link_paths: tuple[str, ...] = ("/bin", "/usr/bin")
    warnings: tuple[str, ...] = ()

    @property
    def bin_dir(self) -> str:
        return f"{self.path}/bin"

    def describe(self) -> str:
        what = {
            "nix": "Nix environment",
            "given": "given environment",
            "host": "host-tools environment",
        }.get(self.origin, self.origin)
        if self.origin == "nix" and self.packages:
            return f"{what}: {self.path} ({', '.join(self.packages)})"
        return f"{what}: {self.path}"


def _inspect(path: str) -> tuple[str, ...]:
    """Warnings about an environment that will not behave.

    A missing `/bin/sh` is the one mistake worth naming out loud: everything
    else fails clearly at the first command, but this one fails as `execvp:
    No such file or directory` from inside a namespace, which is a bad place to
    be debugging.
    """
    problems = []
    if not os.path.isdir(os.path.join(path, "bin")):
        problems.append(f"{path} has no bin/ directory")
    elif not os.path.exists(os.path.join(path, "bin", "sh")):
        problems.append(
            f"{path}/bin/sh does not exist, so `#!/bin/sh` and shelling out will fail; "
            "include bash or coreutils in the environment"
        )
    return tuple(problems)


def cache_file() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "headless-harness" / "sandbox-envs.json"


_env_cache: dict[str, str] = {}
_cache_loaded = False


def _load_cache() -> dict[str, str]:
    global _cache_loaded
    if not _cache_loaded:
        _cache_loaded = True
        try:
            with cache_file().open() as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                _env_cache.update({str(key): str(value) for key, value in stored.items()})
        except (OSError, ValueError):
            pass
    return _env_cache


def _save_cache() -> None:
    try:
        path = cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(_env_cache, handle, indent=0, sort_keys=True)
    except OSError:
        pass  # a cache that cannot be written is not a failure


def expression(packages: tuple[str, ...]) -> str:
    """The `buildEnv` expression for a package list.

    Attribute names become `pkgs.<name>`; anything starting with `/` is used as
    a literal store path, so a caller who already has a closure can name it
    without Nix evaluating anything about it.
    """
    if not packages:
        raise ToolchainError("an environment needs at least one package")
    paths = []
    for package in packages:
        if package.startswith("/"):
            if not os.path.exists(package):
                raise ToolchainError(f"no such store path: {package}")
            paths.append(package)
        elif _ATTRIBUTE_RE.match(package):
            paths.append(f"pkgs.{package}")
        else:
            raise ToolchainError(f"not a package name: {package!r}")
    joined = "\n      ".join(paths)
    return (
        "let\n"
        "  pkgs = import <nixpkgs> { };\n"
        "in\n"
        "pkgs.buildEnv {\n"
        '  name = "sandbox-env";\n'
        "  paths = [\n"
        f"      {joined}\n"
        "  ];\n"
        "  ignoreCollisions = true;\n"
        "}\n"
    )


def with_base_packages(packages: tuple[str, ...]) -> tuple[str, ...]:
    """`packages`, with the two attributes that make `/bin/sh` resolve.

    A caller who named only store paths is taken at their word: they either
    included a shell or they meant not to, and `_inspect` says so.
    """
    if packages and all(name.startswith("/") for name in packages):
        return packages
    names = list(packages)
    for base in BASE_PACKAGES:
        if base not in names:
            names.append(base)
    return tuple(names)


def resolve(spec: SandboxSpec, *, runner=None, use_cache: bool = True) -> Toolchain:
    """The toolchain for a spec: the given directory, or a Nix environment."""
    if spec.env_dir:
        if not os.path.isdir(spec.env_dir):
            raise ToolchainError(f"env_dir is not a directory: {spec.env_dir}")
        return Toolchain(
            path=spec.env_dir,
            packages=(),
            origin="given",
            warnings=_inspect(spec.env_dir),
        )
    return resolve_packages(
        spec.packages, nixpkgs=spec.nixpkgs, runner=runner, use_cache=use_cache
    )


def resolve_packages(
    packages: Sequence[str],
    *,
    nixpkgs: str | None = None,
    runner=None,
    use_cache: bool = True,
) -> Toolchain:
    """A Nix environment holding `packages`, plus `bash` and `coreutils`.

    A package list maps to one `buildEnv` store path and never to another, so
    resolution is cached in-process and, best effort, on disk. This is what
    `resolve` ends up calling for a spec, and it is also what lets a sandbox
    change its packages after creation: a new list is just another store path,
    and every command already mounts `/nix/store` read-only.
    """
    packages = with_base_packages(tuple(packages))
    key = f"{nixpkgs or '<nixpkgs>'}|{' '.join(packages)}"
    if use_cache:
        cached = _load_cache().get(key)
        if cached and os.path.isdir(cached):
            return Toolchain(
                path=cached, packages=packages, origin="nix", warnings=_inspect(cached)
            )
    store_path = build(packages, nixpkgs=nixpkgs, runner=runner)
    if use_cache:
        _env_cache[key] = store_path
        _save_cache()
    return Toolchain(
        path=store_path, packages=packages, origin="nix", warnings=_inspect(store_path)
    )


def build(packages: tuple[str, ...], *, nixpkgs: str | None = None, runner=None) -> str:
    """Run `nix-build` and return the resulting store path."""
    runner = runner or _run
    argv = ["nix-build", "--no-out-link"]
    if nixpkgs:
        argv += ["-I", f"nixpkgs={nixpkgs}"]
    argv += ["--expr", expression(packages)]
    try:
        result = runner(argv, BUILD_TIMEOUT)
    except FileNotFoundError as exc:
        raise ToolchainError("nix-build is not on PATH; install Nix or pass env_dir=") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(f"nix-build did not finish within {BUILD_TIMEOUT:.0f}s") from exc
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        tail = "\n".join(detail.splitlines()[-20:])
        raise ToolchainError(
            f"nix-build failed for {' '.join(packages)}:\n{tail or '<no output>'}"
        )
    lines = [line.strip() for line in (result.stdout or b"").decode().splitlines()]
    # normally one `/nix/store/...` line; any existing absolute path is accepted
    # too, so a relocated store or a test's temporary environment works
    paths = [line for line in lines if line.startswith("/") and os.path.isdir(line)]
    if not paths:
        raise ToolchainError(
            "nix-build printed no store path: " + (lines[-1] if lines else "<no output>")
        )
    return paths[-1]


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, capture_output=True, timeout=timeout)


def host_toolchain(tools: tuple[str, ...] = HOST_TOOLS) -> Toolchain:
    """An environment assembled from the host's own binaries.

    A debugging aid for machines without Nix (`--host-tools`): it links whatever
    is already installed into a temporary `bin/` and asks for the host's
    libraries to be mounted read-only so those binaries can run. It makes the
    sandbox weaker, not stronger — the binaries are the host's, and so are the
    mount points they need.
    """
    directory = tempfile.mkdtemp(prefix="hh-host-tools-")
    bin_dir = os.path.join(directory, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    linked = []
    for tool in tools:
        found = shutil.which(tool)
        if found is None:
            continue
        target = os.path.join(bin_dir, tool)
        if not os.path.lexists(target):
            os.symlink(found, target)
            linked.append(tool)
    # Binding a path whose host form is a symlink (`/lib64 -> usr/lib` on Arch)
    # would fail: bubblewrap refuses to mount on a symlink destination, and the
    # loader's `/lib64/ld-linux-*.so` still has to resolve inside. So the source
    # is the resolved target, mounted at the path the loader will look for.
    # Candidates already covered by another mount are skipped: under a bound
    # `/usr`, `/usr/lib64` is a symlink and mounting on it is what fails.
    mounts = []
    covered: list[str] = []
    for path in ("/usr", "/lib", "/lib64"):
        if not os.path.exists(path):
            continue
        if any(path.startswith(seen.rstrip("/") + "/") for seen in covered):
            continue
        covered.append(path)
        mounts.append(Mount.ro_bind(os.path.realpath(path), path))
    mounts = tuple(mounts)
    return Toolchain(
        path=directory,
        packages=tuple(linked),
        origin="host",
        mounts=mounts,
        # /usr/bin already exists inside the mounted host /usr, so only /bin is
        # linked; the environment's own bin holds the symlinks into /usr/bin
        link_paths=("/bin",),
        warnings=_inspect(directory),
    )
