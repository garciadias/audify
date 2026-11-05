"""A spinner left running must not crash the interpreter at exit."""

import subprocess
import sys

from audify.utils import progress
from audify.utils.progress import ProgressIndicator


def test_stop_all_stops_running_spinners():
    indicator = ProgressIndicator(update_interval=0.01)
    indicator.start()
    assert indicator in progress._RUNNING
    progress._stop_all()
    assert indicator._running is False
    assert indicator._thread is not None
    assert not indicator._thread.is_alive()


def test_stopped_spinner_is_forgotten():
    indicator = ProgressIndicator(update_interval=0.01)
    indicator.start()
    indicator.stop()
    assert indicator not in progress._RUNNING


def test_exit_with_a_running_spinner_is_clean():
    code = (
        "from audify.utils.progress import ProgressIndicator\n"
        "for _ in range(20):\n"
        "    ProgressIndicator(update_interval=0.0005).start()\n"
    )
    for _ in range(5):
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stderr[-500:]
        assert "Fatal Python error" not in result.stderr
