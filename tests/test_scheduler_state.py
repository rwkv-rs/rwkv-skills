from pathlib import Path

from src.eval.scheduler import state


def test_load_running_tolerates_pid_file_removed_during_scan(tmp_path, monkeypatch) -> None:
    pid_file = tmp_path / "vanished.pid"
    pid_file.write_text("123\n")
    original_read_text = Path.read_text

    def remove_then_raise(path: Path, *args, **kwargs):
        if path == pid_file:
            pid_file.unlink()
            raise FileNotFoundError(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", remove_then_raise)

    assert state.load_running(tmp_path) == {}
