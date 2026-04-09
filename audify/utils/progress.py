"""Progress indicator for long-running tasks."""

import sys
import threading
import time
from itertools import cycle
from typing import Optional

# Braille spinner frames for smooth animation
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class ProgressIndicator:
    """
    Thread-safe progress indicator with rotating spinner and phase description.

    Displays: ⠋ Reading... ⠙ Translating... etc.

    Example:
        >>> progress = ProgressIndicator()
        >>> progress.start()
        >>> progress.set_phase("Reading")
        >>> # do work
        >>> progress.set_phase("Translating")
        >>> # do more work
        >>> progress.stop()
    """

    def __init__(self, update_interval: float = 0.1):
        """Initialize progress indicator.

        Args:
            update_interval: How often to update the spinner (in seconds).
        """
        self.update_interval = update_interval
        self._current_phase = "Processing"
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._spinner = cycle(SPINNER_FRAMES)
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the progress indicator in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the progress indicator and clear the line."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        # Clear the line
        sys.stderr.write("\r" + " " * 60 + "\r")
        sys.stderr.flush()

    def set_phase(self, phase: str) -> None:
        """Update the current phase description.

        Args:
            phase: Short description of current phase (e.g., "Reading", "Synthesizing").
        """
        with self._lock:
            self._current_phase = phase

    def _run(self) -> None:
        """Main loop for the progress indicator."""
        while self._running:
            with self._lock:
                frame = next(self._spinner)
                phase = self._current_phase

            # Write to stderr so it doesn't interfere with stdout
            message = f"{frame} {phase}..."
            sys.stderr.write(f"\r{message:<60}")
            sys.stderr.flush()

            time.sleep(self.update_interval)
