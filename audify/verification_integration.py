#!/usr/bin/env python3
"""
Verification integration for audiobook processing pipeline.

Provides functions to:
1. Check individual chapter duration against expected length
2. Verify complete audiobook after generation
3. Prompt user for actions if anomalies detected
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from audify.utils.audio import AudioProcessor
from audify.verify import AudiobookVerifier

logger = logging.getLogger(__name__)


class ChapterDurationChecker:
    """Check individual chapter duration against expected length."""

    # Expected words per minute for audiobook narration
    WORDS_PER_MINUTE = 75

    @staticmethod
    def estimate_duration(word_count: int) -> float:
        """Estimate duration in seconds based on word count.

        Args:
            word_count: Number of words in the script

        Returns:
            Estimated duration in seconds
        """
        minutes = max(word_count / ChapterDurationChecker.WORDS_PER_MINUTE, 0.5)
        return minutes * 60

    @staticmethod
    def get_actual_duration(audio_path: Path) -> float:
        """Get actual duration of audio file in seconds.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)

        Returns:
            Duration in seconds
        """
        try:
            return AudioProcessor.get_duration(str(audio_path))
        except Exception as e:
            logger.warning(f"Could not get duration for {audio_path}: {e}")
            return 0.0

    @staticmethod
    def check_chapter(
        chapter_number: int,
        chapter_title: str,
        audio_path: Path,
        script_word_count: int,
        threshold: float = 0.7,
    ) -> Tuple[bool, str, float]:
        """Check if chapter duration is reasonable.

        Args:
            chapter_number: Episode number
            chapter_title: Chapter title for reporting
            audio_path: Path to generated audio file
            script_word_count: Word count of the script
            threshold: Minimum acceptable ratio (actual/expected). Default 0.7.

        Returns:
            Tuple of (is_ok, message, ratio)
            - is_ok: True if duration is acceptable
            - message: Human-readable status message
            - ratio: Actual/expected duration ratio
        """
        if not audio_path.exists():
            return False, f"Audio file not found: {audio_path}", 0.0

        expected_s = ChapterDurationChecker.estimate_duration(script_word_count)
        actual_s = ChapterDurationChecker.get_actual_duration(audio_path)

        if expected_s == 0:
            return True, f"Episode {chapter_number}: Unable to estimate duration", 0.0

        ratio = actual_s / expected_s if expected_s > 0 else 0.0

        if ratio >= threshold:
            status = "✅"
            message = (
                f"{status} Episode {chapter_number} ({chapter_title}): "
                f"{actual_s:.0f}s (expected {expected_s:.0f}s, ratio: {ratio:.2f})"
            )
            return True, message, ratio
        else:
            status = "⚠️"
            message = (
                f"{status} Episode {chapter_number} ({chapter_title}): "
                f"{actual_s:.0f}s (expected {expected_s:.0f}s, ratio: {ratio:.2f}) "
                f"- SHORTER than expected"
            )
            return False, message, ratio


class AudiobookVerificationCheck:
    """Verify complete audiobook after generation."""

    @staticmethod
    def verify_audiobook(
        source_path: Path,
        audiobook_path: Path,
        duration_ratio_threshold: float = 0.6,
    ) -> Tuple[bool, dict]:
        """Verify complete audiobook against source.

        Args:
            source_path: Path to source EPUB/PDF
            audiobook_path: Path to generated M4B/MP3
            duration_ratio_threshold: Minimum acceptable duration ratio

        Returns:
            Tuple of (verification_passed, report_dict)
        """
        try:
            verifier = AudiobookVerifier(source_path, audiobook_path)
            report = verifier.verify()
            report_dict = verifier.generate_report()

            # Determine if verification passed
            passed = True
            issues = []

            # Check for missing chapters
            if report.has_missing_chapters():
                passed = False
                issues.append(
                    f"Missing {len(report.missing)} chapter(s): "
                    f"{', '.join(m.title for m in report.missing)}"
                )

            # Check for extra chapters
            if report.has_extra_chapters():
                passed = False
                issues.append(
                    f"Extra {len(report.extra)} chapter(s): "
                    f"{', '.join(e.title for e in report.extra)}"
                )

            # Check for order violations
            if report.has_order_issues():
                passed = False
                issues.append(
                    f"Chapter order issues: "
                    f"{', '.join(f'{v.title} (pos {v.actual_position + 1})' for v in report.order_violations)}"
                )

            # Check duration ratio
            if report.duration_hint:
                ratio = report.duration_hint.ratio
                if ratio < duration_ratio_threshold:
                    passed = False
                    issues.append(
                        f"Audio duration {ratio:.2f}x expected "
                        f"({report.duration_hint.actual_duration_s:.0f}s "
                        f"vs {report.duration_hint.expected_duration_s:.0f}s expected)"
                    )

            report_dict["verification_passed"] = passed
            report_dict["issues"] = issues

            return passed, report_dict

        except Exception as e:
            logger.error(f"Error verifying audiobook: {e}", exc_info=True)
            return False, {"error": str(e)}


class VerificationPrompts:
    """User prompts for verification issues."""

    @staticmethod
    def prompt_short_chapter(
        chapter_number: int,
        chapter_title: str,
        ratio: float,
        threshold: float = 0.7,
    ) -> bool:
        """Prompt user about short chapter.

        Args:
            chapter_number: Episode number
            chapter_title: Chapter title
            ratio: Actual/expected duration ratio
            threshold: Minimum acceptable ratio

        Returns:
            True if user wants to continue, False to abort
        """
        if ratio >= threshold:
            return True  # No prompt needed

        message = (
            f"\n⚠️  WARNING: Episode {chapter_number} ({chapter_title}) is shorter than expected\n"
            f"   Expected duration: ~{ChapterDurationChecker.WORDS_PER_MINUTE} words/min\n"
            f"   Actual ratio: {ratio:.1%}\n"
            f"\n"
            f"   Continue anyway? (y/N): "
        )

        response = input(message)
        return response.lower() in ["y", "yes"]

    @staticmethod
    def prompt_audiobook_verification(
        source_path: Path,
        audiobook_path: Path,
        report: dict,
    ) -> bool:
        """Prompt user about audiobook verification issues.

        Args:
            source_path: Path to source file
            audiobook_path: Path to audiobook file
            report: Verification report dict

        Returns:
            True if user accepts, False to retry
        """
        passed = report.get("verification_passed", False)

        if passed:
            logger.info("✅ Audiobook verification passed!")
            return True

        issues = report.get("issues", [])
        if not issues:
            logger.info("✅ Audiobook verification passed!")
            return True

        print("\n" + "=" * 70)
        print("⚠️  AUDIOBOOK VERIFICATION WARNINGS")
        print("=" * 70)
        print(f"\nSource: {source_path.name}")
        print(f"Audiobook: {audiobook_path.name}\n")

        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")

        print("\n" + "=" * 70)
        print(
            "The audiobook was generated but has potential quality issues.\n"
            "Review the issues above before publishing.\n"
        )

        # Option to view full verification report
        view_report = input("View full verification report? (y/N): ")
        if view_report.lower() in ["y", "yes"]:
            verifier = AudiobookVerifier(source_path, audiobook_path)
            verifier.print_report()

        accept = input("\nAccept audiobook anyway? (y/N): ")
        return accept.lower() in ["y", "yes"]


def check_chapter_during_synthesis(
    chapter_number: int,
    chapter_title: str,
    audio_path: Path,
    script_word_count: int,
    confirm: bool = False,
    threshold: float = 0.7,
) -> bool:
    """Check chapter during synthesis with optional user prompt.

    Args:
        chapter_number: Episode number
        chapter_title: Chapter title
        audio_path: Path to audio file
        script_word_count: Word count of script
        confirm: If True, skip user prompts
        threshold: Minimum acceptable ratio

    Returns:
        True to continue, False to abort
    """
    is_ok, message, ratio = ChapterDurationChecker.check_chapter(
        chapter_number, chapter_title, audio_path, script_word_count, threshold
    )

    logger.info(message)

    if is_ok:
        return True

    # If not OK, prompt user unless in confirm mode
    if confirm:
        logger.warning(
            f"Chapter {chapter_number} is short but continuing "
            "(--confirm flag set)"
        )
        return True

    return VerificationPrompts.prompt_short_chapter(
        chapter_number, chapter_title, ratio, threshold
    )


def verify_complete_audiobook(
    source_path: Optional[Path],
    audiobook_path: Path,
    confirm: bool = False,
    duration_ratio_threshold: float = 0.6,
) -> bool:
    """Verify complete audiobook with optional user prompt.

    Args:
        source_path: Path to source EPUB/PDF (or None to skip)
        audiobook_path: Path to generated M4B
        confirm: If True, skip user prompts
        duration_ratio_threshold: Minimum acceptable duration ratio

    Returns:
        True to accept, False if user rejects
    """
    if not source_path or not source_path.exists():
        logger.info("No source file specified, skipping audiobook verification")
        return True

    logger.info("Verifying complete audiobook...")

    passed, report = AudiobookVerificationCheck.verify_audiobook(
        source_path, audiobook_path, duration_ratio_threshold
    )

    if passed:
        logger.info("✅ Audiobook verification passed!")
        return True

    if confirm:
        logger.warning(
            "Audiobook has verification issues but continuing "
            "(--confirm flag set)"
        )
        return True

    return VerificationPrompts.prompt_audiobook_verification(
        source_path, audiobook_path, report
    )
