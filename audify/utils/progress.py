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

    def print_table_of_contents(self, chapters: list[str]) -> None:
        """Display table of contents for all chapters.

        Args:
            chapters: List of chapter titles.
        """
        sys.stdout.write("\n")
        sys.stdout.write("=" * 70 + "\n")
        sys.stdout.write("TABLE OF CONTENTS\n")
        sys.stdout.write("=" * 70 + "\n")
        for i, chapter in enumerate(chapters, 1):
            # Convert to string and handle edge cases
            chapter_str = str(chapter) if chapter else f"Chapter {i}"
            # Truncate long titles to fit terminal width
            max_width = 65
            display_title = (
                chapter_str[: max_width - 3] + "..."
                if len(chapter_str) > max_width
                else chapter_str
            )
            sys.stdout.write(f"{i:3d}. {display_title}\n")
        sys.stdout.write("=" * 70 + "\n\n")
        sys.stdout.flush()

    def print_chapter_start(
        self, chapter_number: int, chapter_title: str, text_snippet: str = ""
    ) -> None:
        """Print chapter processing start information.

        Args:
            chapter_number: Chapter number being processed.
            chapter_title: Title of the chapter.
            text_snippet: First few words from the chapter (optional).
        """
        self.stop()  # Stop spinner to show clear output
        sys.stdout.write("\n")
        sys.stdout.write("-" * 70 + "\n")
        sys.stdout.write(f"Chapter {chapter_number}: {chapter_title}\n")
        if text_snippet:
            # Convert to string and show first 60 characters of text
            snippet_str = str(text_snippet) if text_snippet else ""
            display_snippet = (
                snippet_str[: 60] + "..."
                if len(snippet_str) > 60
                else snippet_str
            )
            if display_snippet:
                sys.stdout.write(f"Preview: {display_snippet}\n")
        sys.stdout.write("-" * 70 + "\n")
        sys.stdout.flush()
        self.start()  # Restart spinner for phase updates

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
