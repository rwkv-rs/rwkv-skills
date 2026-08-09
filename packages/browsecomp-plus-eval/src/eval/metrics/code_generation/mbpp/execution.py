from __future__ import annotations

import contextlib
import faulthandler
import io
import linecache
import multiprocessing
import os
import platform
import signal
import tempfile
import traceback
from typing import Dict, Optional


def unsafe_execute(problem: Dict, completion: str, timeout: float, result):
    with create_tempdir():
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        reliability_guard()

        test_sections: list[str] = []
        for key in ("test_list", "test_cases", "tests", "assertion"):
            section = problem.get(key)
            if not section:
                continue
            if isinstance(section, str):
                test_sections.append(section)
            else:
                test_sections.append("\n".join(section))
        tests_str = "\n".join(test_sections)

        check_program = completion
        if tests_str:
            if not check_program.endswith("\n"):
                check_program += "\n"
            if not tests_str.endswith("\n"):
                tests_str += "\n"
            check_program += tests_str

        try:
            exec_globals: dict = {}
            with swallow_io():
                with time_limit(timeout):
                    filename = "<rwkv_skills_mbpp>"
                    linecache.cache[filename] = (
                        len(check_program),
                        None,
                        check_program.splitlines(keepends=True),
                        filename,
                    )
                    exec(compile(check_program, filename, "exec"), exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as exc:  # noqa: BLE001
            exc_type = type(exc).__name__
            exc_msg = str(exc).strip()
            header = f"failed: {exc_type}: {exc_msg}" if exc_msg else f"failed: {exc_type}"
            tb = traceback.format_exc().rstrip()
            result.append(f"{header}\n{tb}" if tb else header)

        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def check_correctness(problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None) -> Dict:
    """Execute MBPP tests for a completion."""
    manager = multiprocessing.Manager()
    result = manager.list()

    proc = multiprocessing.Process(target=unsafe_execute, args=(problem, completion, timeout, result))
    proc.start()
    proc.join(timeout=timeout + 1)
    if proc.is_alive():
        proc.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem.get("task_id"),
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):  # noqa: ANN001, ARG001
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):  # noqa: ANN002, ARG002
        raise IOError

    def readline(self, *args, **kwargs):  # noqa: ANN002, ARG002
        raise IOError

    def readlines(self, *args, **kwargs):  # noqa: ANN002, ARG002
        raise IOError

    def readable(self, *args, **kwargs):  # noqa: ANN002, ARG002
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """Disable destructive functions to reduce risk (not a true sandbox)."""
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None  # type: ignore

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
