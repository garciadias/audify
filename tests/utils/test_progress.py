"""Tests for the ProgressIndicator spinner phase/counter state machine."""

from audify.utils.progress import ProgressIndicator


class TestProgressIndicatorCounter:
    """Counter lifecycle: set per chapter, cleared on phase transition."""

    def test_set_counter_stores_values(self):
        progress = ProgressIndicator()
        progress.set_counter(3, 20)
        assert progress._current_counter == 3
        assert progress._total_counter == 20

    def test_final_counter_stays_visible(self):
        """The final [N/N] must persist until the phase actually changes."""
        progress = ProgressIndicator()
        progress.set_counter(20, 20)
        assert progress._current_counter == 20
        assert progress._total_counter == 20

    def test_set_phase_clears_counter(self):
        """A phase transition resets the counter so it is never stale."""
        progress = ProgressIndicator()
        progress.set_counter(20, 20)
        progress.set_phase("Assembling")
        assert progress._current_phase == "Assembling"
        assert progress._current_counter is None
        assert progress._total_counter is None

    def test_counter_none_by_default(self):
        progress = ProgressIndicator()
        assert progress._current_counter is None
        assert progress._total_counter is None


class TestProgressIndicatorLifecycle:
    """Start/stop idempotency."""

    def test_start_when_already_running_is_noop(self):
        progress = ProgressIndicator()
        progress._running = True
        progress.start()
        # No thread spawned by the second start call.
        assert progress._thread is None
        progress._running = False

    def test_stop_when_not_running_is_noop(self):
        progress = ProgressIndicator()
        progress.stop()
        assert not progress._running

    def test_start_stop_cycle(self):
        progress = ProgressIndicator(update_interval=0.01)
        progress.start()
        assert progress._running
        assert progress._thread is not None
        progress.stop()
        assert not progress._running
