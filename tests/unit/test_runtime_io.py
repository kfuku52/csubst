import io
import os
import subprocess
import sys

import pytest

from csubst import runtime


@pytest.mark.parametrize("alias_kind", ["directory_symlink", "file_symlink", "hardlink"])
def test_replace_file_preserves_same_file_aliases(tmp_path, alias_kind):
    source = tmp_path / "source.txt"
    source.write_text("simulation output")
    destination = tmp_path / "alias.txt"
    if alias_kind == "directory_symlink":
        alias_dir = tmp_path / "alias"
        alias_dir.symlink_to(tmp_path, target_is_directory=True)
        destination = alias_dir / source.name
    elif alias_kind == "file_symlink":
        destination.symlink_to(source)
    else:
        os.link(source, destination)
    assert runtime.replace_file_cross_device(source, destination) == str(destination)
    assert source.read_text() == "simulation output"
    assert destination.read_text() == "simulation output"


def test_replace_file_replaces_distinct_destination(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("new")
    destination.write_text("old")
    runtime.replace_file_cross_device(source, destination)
    assert not source.exists()
    assert destination.read_text() == "new"


@pytest.mark.parametrize("error_type", [BrokenPipeError, KeyboardInterrupt])
@pytest.mark.parametrize("requires_kill", [False, True])
def test_subprocess_tee_reaps_child_on_output_failure(monkeypatch, error_type, requires_kill):
    class Child:
        stdout = io.StringIO("child output\n")
        terminated = False
        killed = False
        reaped = False

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            if requires_kill and not self.killed:
                raise subprocess.TimeoutExpired("child", timeout)
            self.reaped = True
            return -9 if self.killed else -15

    child = Child()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: child)

    def fail_output(*args, **kwargs):
        raise error_type("output interrupted")

    monkeypatch.setattr("builtins.print", fail_output)
    with pytest.raises(error_type, match="output interrupted"):
        runtime.run_subprocess_tee(["child"])
    assert child.terminated
    assert child.killed == requires_kill
    assert child.reaped
    assert child.stdout.closed


def test_subprocess_tee_reaps_real_child_on_output_failure(monkeypatch):
    popen = subprocess.Popen
    children = []

    def start(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    def fail_output(*args, **kwargs):
        raise BrokenPipeError("closed output")

    monkeypatch.setattr(runtime.subprocess, "Popen", start)
    monkeypatch.setattr("builtins.print", fail_output)
    try:
        with pytest.raises(BrokenPipeError, match="closed output"):
            runtime.run_subprocess_tee([
                sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)",
            ])
        assert children[0].poll() is not None
        assert children[0].stdout.closed
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
            child.stdout.close()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_subprocess_tee_forwards_output_and_exit_status(exit_code, capsys):
    result = runtime.run_subprocess_tee([
        sys.executable, "-c",
        "import os,sys; os.write(1,b'out\\n'); os.write(2,b'err\\xff\\n'); sys.exit({})".format(exit_code),
    ])
    assert result == exit_code
    assert capsys.readouterr().out == "out\nerr\ufffd\n"
