"""A long-lived command in a sandbox, on a pty.

`Sandbox.exec` runs one command and waits for it: each call starts its own
bubblewrap, so it gets fresh namespaces, and anything the command changed —
working directory, environment, background jobs — is gone by the time the next
one starts. That is what makes the sandbox easy to reason about. A session is
the other model. One bubblewrap process holds the namespaces for as long as the
command runs, and the command (a shell, usually) is the parent of everything
typed into it, so state persists from one line to the next. It is the model an
interactive shell needs.

The command is given a pty rather than pipes, and that is the whole reason this
module exists: a shell wants a *controlling terminal* for line editing, the
signal characters, the window size and job control, and `forkpty` is what
provides one. The pty is the sandbox's own — none of the harness's terminal is
passed in — which is why the bubblewrap command behind a session leaves out
`--new-session`: that flag detaches a sandbox from its caller's terminal, and
here the terminal is the sandbox's to keep.

`Sandbox.open_session` does the bubblewrap setup and the resource-limit
handshake; this module is the pty side of it: writing, reading, resizing,
waiting and ending. A session has no watchdog — the spec's `resources.timeout`
applies to `exec` commands, not to a command whose length is the point.
"""

from __future__ import annotations

import fcntl
import os
import select
import signal
import struct
import sys
import termios
import time
import tty
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sandbox import Sandbox, SandboxError

# A close() hangs the pty up and gives the command this long to exit on SIGHUP
# before SIGKILL and the cgroup take over: long enough for a shell to leave on
# its own, short enough that `destroy()` is not noticeably a wait.
HANGUP_GRACE = 2.0
# One read's ceiling. `interact` loops, so this bounds memory, not throughput.
READ_SIZE = 65536
# A pty starts at 0x0; a shell handles that badly, so give it the conventional
# size before the real terminal size is known.
FALLBACK_SIZE = (24, 80)


class Session:
    """A running command that holds a sandbox's namespaces open.

    Created by `Sandbox.open_session`, which is where the bubblewrap process,
    the pty and the resource limits are set up. The command's output is a byte
    stream and a pty is not a pipe — it echoes, translates newlines and, with a
    terminal in raw mode, hands over every keystroke — so the methods are the
    terminal kind:

        session.write("pwd\\n")
        time.sleep(0.2)
        print(session.read().decode())

    `interact()` is that loop with the harness's own terminal attached, for a
    program that is itself interactive.

    The exit status follows `ExecResult`'s convention: the command's code, or
    128+N when a signal killed it. `poll()` and `wait()` return `None` while it
    is still running.
    """

    def __init__(
        self, sandbox: Sandbox, pid: int, master_fd: int, command: Sequence[str]
    ) -> None:
        self._sandbox = sandbox
        self._pid = pid
        self._master: int | None = master_fd
        self._command = tuple(command)
        self._returncode: int | None = None
        self._eof = False
        self.resize()

    # -- observing ----------------------------------------------------------

    @property
    def pid(self) -> int:
        """The bubblewrap process, which is the session leader."""
        return self._pid

    @property
    def command(self) -> tuple[str, ...]:
        return self._command

    @property
    def returncode(self) -> int | None:
        return self.poll()

    @property
    def finished(self) -> bool:
        """Whether the command has ended (on its own, or because it was closed)."""
        return self.poll() is not None

    def fileno(self) -> int:
        """The pty master, for a caller that wants to select on it itself."""
        if self._master is None:
            raise ValueError("this session is closed")
        return self._master

    def poll(self) -> int | None:
        """The exit status once there is one, else None. Never blocks."""
        if self._returncode is None:
            try:
                reaped, status = os.waitpid(self._pid, os.WNOHANG)
            except ChildProcessError:
                # something else in this process reaped it (a SIGCHLD handler,
                # say); there is no status left to report
                self._returncode = 127
            else:
                if reaped:
                    self._returncode = _exit_status(status)
        return self._returncode

    def wait(self, timeout: float | None = None) -> int | None:
        """Wait for the command to end, or `timeout` seconds for it to do so.

        Returns the exit status, or None if the timeout ran out first.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            code = self.poll()
            if code is not None:
                return code
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.02)

    # -- the pty ------------------------------------------------------------

    def write(self, data: bytes | str) -> None:
        """Send bytes (a string is encoded as UTF-8) to the command's stdin."""
        payload = data.encode() if isinstance(data, str) else bytes(data)
        if not payload:
            return
        master = self._master
        if master is None or self.poll() is not None:
            raise _error("the session has ended; there is nothing to write to")
        try:
            _write_all(master, payload)
        except OSError as exc:
            raise _error(f"could not write to the session: {exc}") from exc

    def read(self, *, timeout: float = 0.0, size: int = READ_SIZE) -> bytes:
        """Whatever the command has written, waiting up to `timeout` seconds.

        b"" means nothing arrived in time, and keeps meaning it once the pty has
        reached end of file — which is what a command that has ended looks like
        from here.
        """
        if self._master is None or self._eof:
            return b""
        try:
            ready, _, _ = select.select([self._master], [], [], timeout)
        except InterruptedError:
            return b""
        if not ready:
            return b""
        return self._read_master(size)

    def resize(self, rows: int | None = None, cols: int | None = None) -> None:
        """Match the pty to this process's terminal, or to an explicit size."""
        if self._master is None:
            return
        if rows is None or cols is None:
            try:
                terminal = os.get_terminal_size(sys.stdin.fileno())
            except (OSError, ValueError, AttributeError):
                terminal = os.terminal_size((FALLBACK_SIZE[1], FALLBACK_SIZE[0]))
            rows = rows if rows is not None else terminal.lines
            cols = cols if cols is not None else terminal.columns
        try:
            fcntl.ioctl(
                self._master,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", max(rows, 1), max(cols, 1), 0, 0),
            )
        except OSError:
            pass

    def interrupt(self) -> None:
        """Send the terminal's interrupt character — what Ctrl-C sends."""
        if self.poll() is None:
            self._write_quietly(b"\x03")

    # -- ending -------------------------------------------------------------

    def close(self, *, force: bool = False) -> int | None:
        """End the session, and everything it started.

        The pty is hung up — SIGHUP to the session, then the master closed,
        which is what a terminal does when it goes away — and the command gets
        a moment to leave on its own before SIGKILL and the cgroup take over.
        `force=True` skips the grace period. Returns the exit status, or None
        if it could not be reaped.
        """
        master, self._master = self._master, None
        if master is None:
            return self._returncode
        if self.poll() is None:
            _signal_group(self._pid, signal.SIGHUP)
            _close_fd(master)  # the hangup reaches the foreground process group
            if not force:
                self.wait(timeout=HANGUP_GRACE)
            if self.poll() is None:
                _signal_group(self._pid, signal.SIGKILL)
                self._sandbox.limiter.kill()
        else:
            _close_fd(master)
        self.wait(timeout=10)
        self._sandbox._session_done(self)
        return self._returncode

    def interact(self) -> int:
        """Connect this process's terminal to the session until it ends.

        The terminal is put in raw mode so that every keystroke — Ctrl-C,
        Ctrl-Z, Ctrl-D — is the sandbox shell's to interpret, the pty is given
        the terminal's size and follows SIGWINCH, and output is written through
        as it arrives. Input that is not a terminal is forwarded as it comes and
        the shell is asked to exit at end of file, so a session can also be
        driven from a pipe.
        """
        if self._master is None:
            return self._returncode if self._returncode is not None else 0
        if sys.stdin is None or sys.stdout is None:
            raise _error("interact() needs a standard input and output to carry the session")
        stdin_fd = sys.stdin.fileno()
        stdout_fd = sys.stdout.fileno()
        try:
            sys.stdout.flush()
        except (OSError, ValueError):
            pass
        terminal = os.isatty(stdin_fd)
        saved = None
        previous_winch = None
        if terminal:
            saved = termios.tcgetattr(stdin_fd)
            tty.setraw(stdin_fd)
            self.resize()
            try:
                previous_winch = signal.signal(signal.SIGWINCH, self._on_winch)
            except ValueError:
                previous_winch = None  # not the main thread, where handlers live
        try:
            return self._pump(stdin_fd, stdout_fd, terminal=terminal)
        finally:
            if saved is not None:
                termios.tcsetattr(stdin_fd, termios.TCSADRAIN, saved)
            if previous_winch is not None:
                signal.signal(signal.SIGWINCH, previous_winch)

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "running" if self.poll() is None else f"exited {self._returncode}"
        return f"<Session {' '.join(self._command)} ({state})>"

    # -- internals ----------------------------------------------------------

    def _pump(self, stdin_fd: int, stdout_fd: int, *, terminal: bool) -> int:
        master = self._master
        if master is None:
            return self._returncode if self._returncode is not None else 0
        watching_input = True
        while True:
            watch = [master] if not watching_input else [master, stdin_fd]
            try:
                ready, _, _ = select.select(watch, [], [], 0.5)
            except InterruptedError:
                ready = []
            if master in ready:
                chunk = self._read_master()
                if chunk:
                    _write_all(stdout_fd, chunk)
            if watching_input and stdin_fd in ready:
                try:
                    typed = os.read(stdin_fd, READ_SIZE)
                except OSError:
                    typed = b""
                if typed:
                    self._write_quietly(typed)
                else:
                    watching_input = False
                    # end of input: a terminal's EOF byte is Ctrl-D, but a pipe
                    # has simply run out, so ask the shell to finish rather than
                    # leaving it at a prompt nobody can answer
                    self._write_quietly(b"\x04" if terminal else b"exit\n")
            if self.poll() is not None:
                break
        # the command is gone, but what it wrote last is still on the pty and
        # belongs to the caller; a plain read would block if a process the
        # command started still holds the slave end open, so the drain is
        # bounded by a short timeout instead
        while True:
            chunk = self.read(timeout=0.05)
            if not chunk:
                break
            _write_all(stdout_fd, chunk)
        return self._returncode if self._returncode is not None else 0

    def _read_master(self, size: int = READ_SIZE) -> bytes:
        """Read what is there; b"" marks the end of the pty.

        A pty whose last slave file descriptor has closed raises EIO here — the
        kernel's way of saying end of file for a terminal — so that counts as
        the end rather than as an error.
        """
        if self._master is None:
            return b""
        try:
            chunk = os.read(self._master, size)
        except OSError:
            chunk = b""
        if not chunk:
            self._eof = True
        return chunk

    def _write_quietly(self, payload: bytes) -> None:
        if self._master is None:
            return
        try:
            _write_all(self._master, payload)
        except OSError:
            pass

    def _on_winch(self, *_: object) -> None:
        self.resize()


def _exit_status(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 127


def _signal_group(pid: int, number: int) -> None:
    try:
        os.killpg(pid, number)
    except OSError:
        pass


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _write_all(fd: int, data: bytes) -> None:
    while data:
        written = os.write(fd, data)
        if written <= 0:
            raise OSError("the terminal accepted no data")
        data = data[written:]


def _error(message: str) -> SandboxError:
    # imported here because sandbox.py imports this module on its way to
    # defining Sandbox; a top-level import would be a cycle
    from .sandbox import SandboxError

    return SandboxError(message)
