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
        # Header with gradient effect
        sys.stdout.write(f"{Colors.CYAN}╔{'═' * 68}╗\n")
        sys.stdout.write(f"║ {Colors.BOLD}📚 TABLE OF CONTENTS{Colors.RESET}{Colors.CYAN}" + 
                        " " * 47 + "║\n")
        sys.stdout.write(f"╠{'═' * 68}╣\n{Colors.RESET}")
        
        for i, chapter in enumerate(chapters, 1):
            # Convert to string and handle edge cases
            chapter_str = str(chapter) if chapter else f"Chapter {i}"
            # Truncate long titles to fit terminal width
            max_width = 61
            display_title = (
                chapter_str[: max_width - 3] + "..."
                if len(chapter_str) > max_width
                else chapter_str
            )
            
            # Alternate row colors for better readability
            color = Colors.BLUE if i % 2 == 0 else Colors.CYAN
            sys.stdout.write(
                f"{color}║ {i:2d}. {display_title:<61}{Colors.CYAN} ║\n"
            )
        
        sys.stdout.write(f"╚{'═' * 68}╝{Colors.RESET}\n\n")
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
        
        # Dynamic chapter header with gradient
        sys.stdout.write(f"{Colors.MAGENTA}▸ {Colors.BOLD}CHAPTER {chapter_number}{Colors.RESET} ")
        sys.stdout.write(f"{Colors.CYAN}{chapter_title}{Colors.RESET}\n")
        
        if text_snippet:
            # Convert to string and show first 70 characters of text
            snippet_str = str(text_snippet) if text_snippet else ""
            display_snippet = (
                snippet_str[:70] + "..." if len(snippet_str) > 70 else snippet_str
            )
            if display_snippet:
                # Preview with subtle styling
                sys.stdout.write(f"{Colors.BRIGHT_BLACK}├─ Preview: {display_snippet}{Colors.RESET}\n")
        
        # Progress indicator line
        sys.stdout.write(f"{Colors.YELLOW}└─ Processing...{Colors.RESET}\n")
        sys.stdout.flush()
        self.start()  # Restart spinner for phase updates

    def _run(self) -> None:
        """Main loop for the progress indicator with dynamic styling."""
        while self._running:
            with self._lock:
                frame = next(self._spinner)
                phase = self._current_phase

            # Write to stderr so it doesn't interfere with stdout
            message = f"{Colors.GREEN}{frame}{Colors.RESET} {Colors.CYAN}{phase}...{Colors.RESET}"
            sys.stderr.write(f"\r{message:<60}")
            sys.stderr.flush()

            time.sleep(self.update_interval)
