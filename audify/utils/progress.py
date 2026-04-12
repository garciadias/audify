"""Progress indicator for long-running tasks."""

import sys
import threading
import time
from itertools import cycle
from typing import Optional

# Braille spinner frames for smooth animation
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# ANSI color codes for enhanced visual feedback
class Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    BRIGHT_BLACK = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


# Phase emoji mapping for visual feedback
PHASE_EMOJIS = {
    "Reading": "📖",
    "Processing": "⚙️",
    "Generating": "✨",
    "Synthesizing": "🔊",
    "Translating": "🌐",
    "Converting": "🔄",
    "Assembling": "🔗",
}


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

        # Clear the line with reset codes
        sys.stderr.write("\r" + " " * 80 + "\r")
        sys.stderr.flush()

    def set_phase(self, phase: str) -> None:
        """Update the current phase description.

        Args:
            phase: Short description of current phase (e.g., "Reading", "Synthesizing").
        """
        with self._lock:
            self._current_phase = phase

    def print_table_of_contents(self, chapters: list[str]) -> None:
        """Display table of contents for all chapters with modern styling.

        Args:
            chapters: List of chapter titles.
        """
        sys.stdout.write("\n")
        # Box border width (68 = 2 for borders + 66 content)
        border = "═" * 68
        sys.stdout.write(f"{Colors.CYAN}╔{border}╗{Colors.RESET}\n")

        # Header title - "📚 TABLE OF CONTENTS" is 21 visible chars
        header_text = "📚 TABLE OF CONTENTS"
        padding = 66 - len(header_text) - 2  # -2 for internal padding
        sys.stdout.write(
            f"{Colors.CYAN}║ {Colors.BOLD}{header_text}{Colors.RESET}{Colors.CYAN}"
            f"{' ' * padding}║{Colors.RESET}\n"
        )
        sys.stdout.write(f"{Colors.CYAN}╠{border}╣{Colors.RESET}\n")

        for i, chapter in enumerate(chapters, 1):
            # Convert to string and handle edge cases
            chapter_str = str(chapter) if chapter else f"Chapter {i}"
            # Truncate long titles to fit in the box (66 - 2 for "N. " - 1 for space)
            max_width = 60
            display_title = (
                chapter_str[: max_width - 3] + "..."
                if len(chapter_str) > max_width
                else chapter_str
            )

            # Alternate row colors for better readability
            color = Colors.BLUE if i % 2 == 0 else Colors.CYAN
            # Proper padding: 66 chars total - number width - dot - space - title
            title_padding = 66 - len(str(i)) - 3 - len(display_title) - 1
            sys.stdout.write(
                f"{Colors.CYAN}║{Colors.RESET} {color}{i:2d}. {display_title}"
                f"{' ' * title_padding}{Colors.RESET}{Colors.CYAN}║{Colors.RESET}\n"
            )

        sys.stdout.write(f"{Colors.CYAN}╚{border}╝{Colors.RESET}\n\n")
        sys.stdout.flush()

    def print_chapter_start(
        self, chapter_number: int, chapter_title: str, text_snippet: str = ""
    ) -> None:
        """Print chapter processing start information with dynamic styling.

        Args:
            chapter_number: Chapter number being processed.
            chapter_title: Title of the chapter.
            text_snippet: First few words from the chapter (optional).
        """
        self.stop()  # Stop spinner to show clear output
        sys.stdout.write("\n")

        # Dynamic chapter header with book emoji
        sys.stdout.write(
            f"{Colors.MAGENTA}📖 {Colors.BOLD}CHAPTER {chapter_number}{Colors.RESET} "
            f"{Colors.CYAN}{chapter_title}{Colors.RESET}\n"
        )

        if text_snippet:
            # Convert to string and show first 65 characters of text
            snippet_str = str(text_snippet) if text_snippet else ""
            max_preview = 65
            display_snippet = (
                snippet_str[:max_preview] + "..."
                if len(snippet_str) > max_preview
                else snippet_str
            )
            if display_snippet:
                # Preview with speech bubble emoji and subtle styling
                sys.stdout.write(
                    f"{Colors.BRIGHT_BLACK}💬 Preview: {display_snippet}{Colors.RESET}\n"
                )

        # Progress indicator line with lightning emoji
        sys.stdout.write(f"{Colors.YELLOW}⚡ Processing...{Colors.RESET}\n")
        # Add blank line to separate from tqdm output
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.start()  # Restart spinner for phase updates

    def _run(self) -> None:
        """Main loop for the progress indicator with dynamic styling."""
        while self._running:
            with self._lock:
                frame = next(self._spinner)
                phase = self._current_phase

            # Get emoji for current phase
            emoji = PHASE_EMOJIS.get(phase, "⏳")

            # Write to stderr so it doesn't interfere with stdout
            message = f"{Colors.GREEN}{frame}{Colors.RESET} {emoji} {Colors.CYAN}{phase}...{Colors.RESET}"
            sys.stderr.write(f"\r{message:<70}")
            sys.stderr.flush()

            time.sleep(self.update_interval)
