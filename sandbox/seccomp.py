"""The syscall filter, as a classic-BPF program bubblewrap can load.

bubblewrap takes `--seccomp FD` and reads the fd as a bare array of
`struct sock_filter` (8 bytes each, `{u16 code; u8 jt; u8 jf; u32 k}`); the
instruction count is the file length divided by eight. It is *not* a
`sock_fprog` header followed by code, which is what the man page's mention of
`seccomp_export_bpf` suggests — `bubblewrap.c`'s `seccomp_program_new` sets
`program.len = len / 8` and points `program.filter` at the data itself. So the
serialisation below is the whole interface, and `decode` reads it back so tests
can assert on what will actually run.

The program is a denylist with a default of `SECCOMP_RET_ALLOW`. Denied calls
return `EPERM` (or `ENOSYS`, for the die-hards who want a runtime to think the
kernel is older than it is) rather than killing the process: a process that
probes a syscall and falls back should keep working.

Two structural checks come first, both standard:

* the architecture word must match, or the process is killed — otherwise a
  filter written for x86_64 numbers would be applied to a different ABI's
  numbers;
* on x86_64, any syscall number with the x32 bit set is refused, because the
  x32 ABI has its own numbering that would otherwise slip past a table built
  for the 64-bit ABI.

Every entry in the default list is there because it would let a process escape
its namespaces, attack the kernel through a historically fragile surface, or
reach a resource the sandbox deliberately does not have. The list is
deliberately short: compilers, Node and browser engines all use syscalls that
sound dangerous in isolation, and a filter that breaks them is a filter that
gets turned off. `SyscallPolicy.allow` removes an entry when a workload needs
it, and names that do not exist on the running architecture are skipped and
reported instead of failing the build.
"""

from __future__ import annotations

import platform
import struct
from dataclasses import dataclass
from typing import Iterable

from .spec import SpecError, SyscallPolicy

# --- classic BPF ------------------------------------------------------------

_BPF_LD = 0x00
_BPF_W = 0x00
_BPF_ABS = 0x20
_BPF_JMP = 0x05
_BPF_JEQ = 0x10
_BPF_JGE = 0x30
_BPF_K = 0x00
_BPF_RET = 0x06

# seccomp_data: int nr (offset 0), __u32 arch (offset 4), then ip and args
_OFFSET_NR = 0
_OFFSET_ARCH = 4

SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000

# linux/audit.h
AUDIT_ARCH_X86_64 = 0xC000003E
AUDIT_ARCH_AARCH64 = 0xC00000B7

# syscall numbers are only defined relative to an architecture, so the tables
# are explicit and the package refuses to guess. x86_64 numbers follow
# arch/x86/entry/syscalls/syscall_64.tbl; aarch64 follows the generic table in
# include/uapi/asm-generic/unistd.h, whose newer entries (424 and up) share
# their numbers with x86_64.
_X86_64 = {
    "read": 0, "write": 1, "open": 2, "close": 3, "stat": 4, "fstat": 5,
    "lstat": 6, "poll": 7, "lseek": 8, "mmap": 9, "mprotect": 10, "munmap": 11,
    "brk": 12, "rt_sigaction": 13, "rt_sigprocmask": 14, "rt_sigreturn": 15,
    "ioctl": 16, "pread64": 17, "pwrite64": 18, "readv": 19, "writev": 20,
    "access": 21, "pipe": 22, "select": 23, "sched_yield": 24, "mremap": 25,
    "msync": 26, "mincore": 27, "madvise": 28, "shmget": 29, "shmat": 30,
    "shmctl": 31, "dup": 32, "dup2": 33, "pause": 34, "nanosleep": 35,
    "getitimer": 36, "alarm": 37, "setitimer": 38, "getpid": 39, "sendfile": 40,
    "socket": 41, "connect": 42, "accept": 43, "sendto": 44, "recvfrom": 45,
    "sendmsg": 46, "recvmsg": 47, "shutdown": 48, "bind": 49, "listen": 50,
    "getsockname": 51, "getpeername": 52, "socketpair": 53, "setsockopt": 54,
    "getsockopt": 55, "clone": 56, "fork": 57, "vfork": 58, "execve": 59,
    "exit": 60, "wait4": 61, "kill": 62, "uname": 63, "semget": 64, "semop": 65,
    "semctl": 66, "shmdt": 67, "msgget": 68, "msgsnd": 69, "msgrcv": 70,
    "msgctl": 71, "fcntl": 72, "flock": 73, "fsync": 74, "fdatasync": 75,
    "truncate": 76, "ftruncate": 77, "getdents": 78, "getcwd": 79, "chdir": 80,
    "fchdir": 81, "rename": 82, "mkdir": 83, "rmdir": 84, "creat": 85,
    "link": 86, "unlink": 87, "symlink": 88, "readlink": 89, "chmod": 90,
    "fchmod": 91, "chown": 92, "fchown": 93, "lchown": 94, "umask": 95,
    "gettimeofday": 96, "getrlimit": 97, "getrusage": 98, "sysinfo": 99,
    "times": 100, "ptrace": 101, "getuid": 102, "syslog": 103, "getgid": 104,
    "setuid": 105, "setgid": 106, "geteuid": 107, "getegid": 108, "setpgid": 109,
    "getppid": 110, "getpgrp": 111, "setsid": 112, "setreuid": 113,
    "setregid": 114, "getgroups": 115, "setgroups": 116, "setresuid": 117,
    "getresuid": 118, "setresgid": 119, "getresgid": 120, "getpgid": 121,
    "setfsuid": 122, "setfsgid": 123, "getsid": 124, "capget": 125, "capset": 126,
    "rt_sigpending": 127, "rt_sigtimedwait": 128, "rt_sigqueueinfo": 129,
    "rt_sigsuspend": 130, "sigaltstack": 131, "utime": 132, "mknod": 133,
    "uselib": 134, "personality": 135, "ustat": 136, "statfs": 137,
    "fstatfs": 138, "sysfs": 139, "getpriority": 140, "setpriority": 141,
    "sched_setparam": 142, "sched_getparam": 143, "sched_setscheduler": 144,
    "sched_getscheduler": 145, "sched_get_priority_max": 146,
    "sched_get_priority_min": 147, "sched_rr_get_interval": 148, "mlock": 149,
    "munlock": 150, "mlockall": 151, "munlockall": 152, "vhangup": 153,
    "modify_ldt": 154, "pivot_root": 155, "_sysctl": 156, "prctl": 157,
    "arch_prctl": 158, "adjtimex": 159, "setrlimit": 160, "chroot": 161,
    "sync": 162, "acct": 163, "settimeofday": 164, "mount": 165, "umount2": 166,
    "swapon": 167, "swapoff": 168, "reboot": 169, "sethostname": 170,
    "setdomainname": 171, "iopl": 172, "ioperm": 173, "create_module": 174,
    "init_module": 175, "delete_module": 176, "get_kernel_syms": 177,
    "query_module": 178, "quotactl": 179, "nfsservctl": 180, "getpmsg": 181,
    "putpmsg": 182, "afs_syscall": 183, "tuxcall": 184, "security": 185,
    "gettid": 186, "readahead": 187, "setxattr": 188, "lsetxattr": 189,
    "fsetxattr": 190, "getxattr": 191, "lgetxattr": 192, "fgetxattr": 193,
    "listxattr": 194, "llistxattr": 195, "flistxattr": 196, "removexattr": 197,
    "lremovexattr": 198, "fremovexattr": 199, "tkill": 200, "time": 201,
    "futex": 202, "sched_setaffinity": 203, "sched_getaffinity": 204,
    "set_thread_area": 205, "io_setup": 206, "io_destroy": 207,
    "io_getevents": 208, "io_submit": 209, "io_cancel": 210,
    "get_thread_area": 211, "lookup_dcookie": 212, "epoll_create": 213,
    "remap_file_pages": 216, "getdents64": 217, "set_tid_address": 218,
    "restart_syscall": 219, "semtimedop": 220, "fadvise64": 221,
    "timer_create": 222, "timer_settime": 223, "timer_gettime": 224,
    "timer_getoverrun": 225, "timer_delete": 226, "clock_settime": 227,
    "clock_gettime": 228, "clock_getres": 229, "clock_nanosleep": 230,
    "exit_group": 231, "epoll_wait": 232, "epoll_ctl": 233, "tgkill": 234,
    "utimes": 235, "mbind": 237, "set_mempolicy": 238, "get_mempolicy": 239,
    "mq_open": 240, "mq_unlink": 241, "mq_timedsend": 242, "mq_timedreceive": 243,
    "mq_notify": 244, "mq_getsetattr": 245, "kexec_load": 246, "waitid": 247,
    "add_key": 248, "request_key": 249, "keyctl": 250, "ioprio_set": 251,
    "ioprio_get": 252, "inotify_init": 253, "inotify_add_watch": 254,
    "inotify_rm_watch": 255, "migrate_pages": 256, "openat": 257, "mkdirat": 258,
    "mknodat": 259, "fchownat": 260, "futimesat": 261, "newfstatat": 262,
    "unlinkat": 263, "renameat": 264, "linkat": 265, "symlinkat": 266,
    "readlinkat": 267, "fchmodat": 268, "faccessat": 269, "pselect6": 270,
    "ppoll": 271, "unshare": 272, "set_robust_list": 273, "get_robust_list": 274,
    "splice": 275, "tee": 276, "sync_file_range": 277, "vmsplice": 278,
    "move_pages": 279, "utimensat": 280, "epoll_pwait": 281, "signalfd": 282,
    "timerfd_create": 283, "eventfd": 284, "fallocate": 285,
    "timerfd_settime": 286, "timerfd_gettime": 287, "accept4": 288,
    "signalfd4": 289, "eventfd2": 290, "epoll_create1": 291, "dup3": 292,
    "pipe2": 293, "inotify_init1": 294, "preadv": 295, "pwritev": 296,
    "rt_tgsigqueueinfo": 297, "perf_event_open": 298, "recvmmsg": 299,
    "fanotify_init": 300, "fanotify_mark": 301, "prlimit64": 302,
    "name_to_handle_at": 303, "open_by_handle_at": 304, "clock_adjtime": 305,
    "syncfs": 306, "sendmmsg": 307, "setns": 308, "getcpu": 309,
    "process_vm_readv": 310, "process_vm_writev": 311, "kcmp": 312,
    "finit_module": 313, "sched_setattr": 314, "sched_getattr": 315,
    "renameat2": 316, "seccomp": 317, "getrandom": 318, "memfd_create": 319,
    "kexec_file_load": 320, "bpf": 321, "execveat": 322, "userfaultfd": 323,
    "membarrier": 324, "mlock2": 325, "copy_file_range": 326, "preadv2": 327,
    "pwritev2": 328, "pkey_mprotect": 329, "pkey_alloc": 330, "pkey_free": 331,
    "statx": 332, "io_pgetevents": 333, "rseq": 334, "pidfd_send_signal": 424,
    "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    "open_tree": 428, "move_mount": 429, "fsopen": 430, "fsconfig": 431,
    "fsmount": 432, "fspick": 433, "pidfd_open": 434, "clone3": 435,
    "close_range": 436, "openat2": 437, "pidfd_getfd": 438, "faccessat2": 439,
    "process_madvise": 440, "epoll_pwait2": 441, "mount_setattr": 442,
    "quotactl_fd": 443, "landlock_create_ruleset": 444, "landlock_add_rule": 445,
    "landlock_restrict_self": 446, "memfd_secret": 447, "process_mrelease": 448,
    "futex_waitv": 449, "set_mempolicy_home_node": 450,
}

# Only what this package needs to be able to name on aarch64; entries a
# missing name is skipped rather than guessed, and the omission is reported.
_AARCH64 = {
    "io_setup": 0, "setxattr": 5, "getxattr": 8, "listxattr": 11,
    "removexattr": 14, "getcwd": 17, "lookup_dcookie": 18, "eventfd2": 19,
    "epoll_create1": 20, "epoll_ctl": 21, "epoll_pwait": 22, "dup": 23,
    "dup3": 24, "fcntl": 25, "inotify_init1": 26, "inotify_add_watch": 27,
    "inotify_rm_watch": 28, "ioctl": 29, "ioprio_set": 30, "ioprio_get": 31,
    "flock": 32, "mknodat": 33, "mkdirat": 34, "unlinkat": 35, "symlinkat": 36,
    "linkat": 37, "renameat": 38, "umount2": 39, "mount": 40, "pivot_root": 41,
    "nfsservctl": 42, "statfs": 43, "fstatfs": 44, "truncate": 45,
    "ftruncate": 46, "fallocate": 47, "faccessat": 48, "chdir": 49,
    "fchdir": 50, "chroot": 51, "fchmod": 52, "fchmodat": 53, "fchownat": 54,
    "fchown": 55, "openat": 56, "close": 57, "vhangup": 58, "pipe2": 59,
    "quotactl": 60, "getdents64": 61, "lseek": 62, "read": 63, "write": 64,
    "readv": 65, "writev": 66, "pread64": 67, "pwrite64": 68, "preadv": 69,
    "pwritev": 70, "sendfile": 71, "pselect6": 72, "ppoll": 73,
    "signalfd4": 74, "vmsplice": 75, "splice": 76, "tee": 77, "readlinkat": 78,
    "newfstatat": 79, "fstat": 80, "sync": 81, "fsync": 82, "fdatasync": 83,
    "sync_file_range": 84, "timerfd_create": 85, "timerfd_settime": 86,
    "timerfd_gettime": 87, "utimensat": 88, "acct": 89, "capget": 90,
    "capset": 91, "personality": 92, "exit": 93, "exit_group": 94, "waitid": 95,
    "set_tid_address": 96, "unshare": 97, "futex": 98, "set_robust_list": 99,
    "get_robust_list": 100, "nanosleep": 101, "getitimer": 102,
    "setitimer": 103, "kexec_load": 104, "init_module": 105,
    "delete_module": 106, "timer_create": 107, "timer_gettime": 108,
    "timer_getoverrun": 109, "timer_settime": 110, "timer_delete": 111,
    "clock_settime": 112, "clock_gettime": 113, "clock_getres": 114,
    "clock_nanosleep": 115, "syslog": 116, "ptrace": 117, "sched_setparam": 118,
    "sched_setscheduler": 119, "sched_getscheduler": 120, "sched_getparam": 121,
    "sched_setaffinity": 122, "sched_getaffinity": 123, "sched_yield": 124,
    "sched_get_priority_max": 125, "sched_get_priority_min": 126,
    "sched_rr_get_interval": 127, "restart_syscall": 128, "kill": 129,
    "tkill": 130, "tgkill": 131, "sigaltstack": 132, "rt_sigsuspend": 133,
    "rt_sigaction": 134, "rt_sigprocmask": 135, "rt_sigpending": 136,
    "rt_sigtimedwait": 137, "rt_sigqueueinfo": 138, "rt_sigreturn": 139,
    "setpriority": 140, "getpriority": 141, "reboot": 142, "setregid": 143,
    "setgid": 144, "setreuid": 145, "setuid": 146, "setresuid": 147,
    "getresuid": 148, "setresgid": 149, "getresgid": 150, "setfsuid": 151,
    "setfsgid": 152, "times": 153, "setpgid": 154, "getpgid": 155,
    "getsid": 156, "setsid": 157, "getgroups": 158, "setgroups": 159,
    "uname": 160, "sethostname": 161, "setdomainname": 162, "getrlimit": 163,
    "setrlimit": 164, "getrusage": 165, "umask": 166, "prctl": 167,
    "getcpu": 168, "gettimeofday": 169, "settimeofday": 170, "adjtimex": 171,
    "getpid": 172, "getppid": 173, "getuid": 174, "geteuid": 175, "getgid": 176,
    "getegid": 177, "gettid": 178, "sysinfo": 179, "mq_open": 180,
    "mq_unlink": 181, "mq_timedsend": 182, "mq_timedreceive": 183,
    "mq_notify": 184, "mq_getsetattr": 185, "msgget": 186, "msgctl": 187,
    "msgrcv": 188, "msgsnd": 189, "semget": 190, "semctl": 191,
    "semtimedop": 192, "semop": 193, "shmget": 194, "shmctl": 195,
    "shmat": 196, "shmdt": 197, "socket": 198, "socketpair": 199, "bind": 200,
    "listen": 201, "accept": 202, "connect": 203, "getsockname": 204,
    "getpeername": 205, "sendto": 206, "recvfrom": 207, "setsockopt": 208,
    "getsockopt": 209, "shutdown": 210, "sendmsg": 211, "recvmsg": 212,
    "readahead": 213, "brk": 214, "munmap": 215, "mremap": 216, "add_key": 217,
    "request_key": 218, "keyctl": 219, "clone": 220, "execve": 221, "mmap": 222,
    "fadvise64": 223, "swapon": 224, "swapoff": 225, "mprotect": 226,
    "msync": 227, "mlock": 228, "munlock": 229, "mlockall": 230,
    "munlockall": 231, "mincore": 232, "madvise": 233, "remap_file_pages": 234,
    "mbind": 235, "get_mempolicy": 236, "set_mempolicy": 237,
    "migrate_pages": 238, "move_pages": 239, "rt_tgsigqueueinfo": 240,
    "perf_event_open": 241, "accept4": 242, "recvmmsg": 243, "wait4": 260,
    "prlimit64": 261, "fanotify_init": 262, "fanotify_mark": 263,
    "name_to_handle_at": 264, "open_by_handle_at": 265, "clock_adjtime": 266,
    "syncfs": 267, "setns": 268, "sendmmsg": 269, "process_vm_readv": 270,
    "process_vm_writev": 271, "kcmp": 272, "finit_module": 273,
    "sched_setattr": 274, "sched_getattr": 275, "renameat2": 276,
    "seccomp": 277, "getrandom": 278, "memfd_create": 279, "bpf": 280,
    "execveat": 281, "userfaultfd": 282, "membarrier": 283, "mlock2": 284,
    "copy_file_range": 285, "preadv2": 286, "pwritev2": 287,
    "pkey_mprotect": 288, "pkey_alloc": 289, "pkey_free": 290, "statx": 291,
    "io_pgetevents": 292, "rseq": 293, "kexec_file_load": 294,
    "pidfd_send_signal": 424, "io_uring_setup": 425, "io_uring_enter": 426,
    "io_uring_register": 427, "open_tree": 428, "move_mount": 429,
    "fsopen": 430, "fsconfig": 431, "fsmount": 432, "fspick": 433,
    "pidfd_open": 434, "clone3": 435, "close_range": 436, "openat2": 437,
    "pidfd_getfd": 438, "faccessat2": 439, "process_madvise": 440,
    "epoll_pwait2": 441, "mount_setattr": 442, "quotactl_fd": 443,
    "landlock_create_ruleset": 444, "landlock_add_rule": 445,
    "landlock_restrict_self": 446, "memfd_secret": 447,
    "process_mrelease": 448, "futex_waitv": 449,
}

_TABLES: dict[str, tuple[int, dict[str, int]]] = {
    "x86_64": (AUDIT_ARCH_X86_64, _X86_64),
    "aarch64": (AUDIT_ARCH_AARCH64, _AARCH64),
}

_MACHINE_ALIASES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "x64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}

# Denied by default. Each group, and why it is here:
#
#   mount, umount2, pivot_root, chroot, mount_setattr, open_tree, move_mount,
#   fsopen, fsconfig, fsmount, fspick
#       editing the mount table; the sandbox's whole filesystem story rests on
#       these being out of reach
#   ptrace, process_vm_readv, process_vm_writev, kcmp
#       reaching into another process
#   init_module, finit_module, delete_module, create_module, kexec_load,
#   kexec_file_load, bpf, perf_event_open, userfaultfd, open_by_handle_at
#       kernel attack surface that has produced escapes and privilege
#       escalations; almost all of it is privileged anyway
#   reboot, swapon, swapoff, acct, quotactl, quotactl_fd, nfsservctl,
#   iopl, ioperm, vhangup, syslog
#       machine-level state the sandbox has no business touching, plus the
#       kernel log
#   settimeofday, clock_settime, adjtimex, clock_adjtime
#       the clock: nothing inside should be able to move a shared resource
#   add_key, request_key, keyctl
#       the kernel keyring, which is not namespaced
#   _sysctl, get_kernel_syms, query_module, ustat, modify_ldt, afs_syscall,
#   tuxcall, security, getpmsg, putpmsg
#       removed, unimplemented or legacy-only; denying them costs nothing and
#       keeps the list honest about what it covers
DEFAULT_DENY: tuple[str, ...] = (
    "mount", "umount2", "pivot_root", "chroot", "mount_setattr", "open_tree",
    "move_mount", "fsopen", "fsconfig", "fsmount", "fspick",
    "ptrace", "process_vm_readv", "process_vm_writev", "kcmp",
    "init_module", "finit_module", "delete_module", "create_module",
    "kexec_load", "kexec_file_load", "bpf", "perf_event_open", "userfaultfd",
    "open_by_handle_at",
    "reboot", "swapon", "swapoff", "acct", "quotactl", "quotactl_fd",
    "nfsservctl", "iopl", "ioperm", "vhangup", "syslog",
    "settimeofday", "clock_settime", "adjtimex", "clock_adjtime",
    "add_key", "request_key", "keyctl",
    "_sysctl", "get_kernel_syms", "query_module", "ustat", "modify_ldt",
    "afs_syscall", "tuxcall", "security", "getpmsg", "putpmsg",
)

_ERRNO_VALUES = {"EPERM": 1, "ENOSYS": 38}

# The value a jump uses to mean "keep going" when a syscall is not in the list;
# only a handful of instructions are ever needed, so 8-bit offsets are plenty.
assert len(DEFAULT_DENY) * 2 + 8 < 4096


def syscall_number(name: str, arch: str | None = None) -> int | None:
    """The number for a syscall name on an architecture, or None if absent."""
    _audit, table = _TABLES[machine_arch(arch)]
    return table.get(name)


def machine_arch(machine: str | None = None) -> str:
    """The kernel's architecture name, or a `SpecError` if it is unknown.

    Failing closed matters here: a filter written with another architecture's
    numbers would deny the wrong syscalls, so an unknown machine means no
    seccomp rather than a wrong seccomp.
    """
    name = _MACHINE_ALIASES.get((machine or platform.machine()).lower())
    if name is None:
        raise SpecError(
            f"no seccomp syscall table for this architecture ({machine or platform.machine()}); "
            "pass SyscallPolicy(enabled=False) to run without a syscall filter"
        )
    return name


@dataclass(frozen=True)
class Program:
    """A filter ready for `bwrap --seccomp`, plus what went into it."""

    data: bytes
    arch: str
    audit_arch: int
    denied: tuple[str, ...]
    numbers: tuple[int, ...]
    skipped: tuple[str, ...]

    @property
    def instruction_count(self) -> int:
        return len(self.data) // 8


def resolve_names(policy: SyscallPolicy, arch: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The names that will be denied on `arch`, and those that do not exist there."""
    _audit, table = _TABLES[arch]
    wanted: list[str] = []
    for name in DEFAULT_DENY:
        if name not in policy.allow:
            wanted.append(name)
    for extra in policy.deny:
        if isinstance(extra, int) or extra in policy.allow or extra in wanted:
            continue
        wanted.append(extra)
    denied = tuple(name for name in wanted if name in table)
    skipped = tuple(name for name in wanted if name not in table)
    return denied, skipped


def build(policy: SyscallPolicy, arch: str | None = None) -> Program | None:
    """The seccomp program for a policy, or None when the filter is off."""
    if not policy.enabled:
        return None
    name = machine_arch(arch)
    audit_arch, table = _TABLES[name]
    denied, skipped = resolve_names(policy, name)
    raw = _raw_numbers(policy)
    numbers = [table[entry] for entry in denied] + raw
    errno_value = _ERRNO_VALUES.get(policy.errno)
    if errno_value is None:
        raise SpecError(f"unknown errno for denied syscalls: {policy.errno!r}")
    data = _assemble(audit_arch, numbers, errno_value)
    return Program(
        data=data,
        arch=name,
        audit_arch=audit_arch,
        # numbers are reported alongside names so the count and the summary stay
        # honest about what is in the program
        denied=denied + tuple(f"#{number}" for number in raw),
        numbers=tuple(numbers),
        skipped=skipped,
    )


def _raw_numbers(policy: SyscallPolicy) -> list[int]:
    """The syscalls the policy names by number, validated."""
    numbers = []
    for entry in policy.deny:
        if isinstance(entry, bool) or not isinstance(entry, int):
            continue
        if entry < 0:
            raise SpecError(f"syscall numbers cannot be negative: {entry}")
        if entry >= 0x40000000:
            raise SpecError(
                f"syscall number {entry} is in the x32 range, which is already refused"
            )
        if entry not in numbers:
            numbers.append(entry)
    return numbers


def _instruction(code: int, jt: int, jf: int, k: int) -> bytes:
    return struct.pack("<HBBI", code, jt, jf, k)


def _load(field_offset: int) -> bytes:
    return _instruction(_BPF_LD | _BPF_W | _BPF_ABS, 0, 0, field_offset)


def _jump(op: int, value: int, jt: int, jf: int) -> bytes:
    return _instruction(_BPF_JMP | op | _BPF_K, jt, jf, value & 0xFFFFFFFF)


def _return(action: int) -> bytes:
    return _instruction(_BPF_RET | _BPF_K, 0, 0, action)


def _assemble(audit_arch: int, numbers: Iterable[int], errno_value: int) -> bytes:
    """Deny `numbers`, allow everything else, kill anything from another ABI."""
    deny = _return(SECCOMP_RET_ERRNO | errno_value)
    parts = [
        _load(_OFFSET_ARCH),
        # arch == ours -> skip the kill, otherwise fall into it
        _jump(_BPF_JEQ, audit_arch, 1, 0),
        _return(SECCOMP_RET_KILL_PROCESS),
        _load(_OFFSET_NR),
        # the x32 ABI numbers its syscalls above 0x40000000 and would otherwise
        # match nothing in a table built for the 64-bit ABI
        _jump(_BPF_JGE, 0x40000000, 0, 1),
        deny,
    ]
    for number in numbers:
        parts.append(_jump(_BPF_JEQ, number, 0, 1))
        parts.append(deny)
    parts.append(_return(SECCOMP_RET_ALLOW))
    return b"".join(parts)


def decode(data: bytes) -> list[tuple[int, int, int, int]]:
    """Read a program back as `(code, jt, jf, k)` tuples.

    Used by the tests to assert on the program that will actually be loaded,
    and by the self-test to print the policy it is running under.
    """
    if len(data) % 8:
        raise SpecError("seccomp data must be a whole number of 8-byte instructions")
    return [struct.unpack_from("<HBBI", data, offset) for offset in range(0, len(data), 8)]


def describe(program: Program | None) -> str:
    """A one-line summary for logs and `--self-test`."""
    if program is None:
        return "seccomp: off"
    text = (
        f"seccomp: {len(program.denied)} syscalls denied on {program.arch} "
        f"({program.instruction_count} BPF instructions)"
    )
    if program.skipped:
        text += f"; not present on this architecture: {', '.join(program.skipped)}"
    return text
