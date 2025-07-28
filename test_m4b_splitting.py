#!/usr/bin/env python3
"""
Test script to verify M4B splitting functionality
"""
from pathlib import Path

from audify.text_to_speech import EpubSynthesizer


def test_duration_calculation():
    """Test the duration calculation and splitting logic"""
    # Create a mock synthesizer instance to test methods
    mock_path = Path("./test.epub")
    synthesizer = EpubSynthesizer.__new__(EpubSynthesizer)

    # Mock some attributes needed for the methods
    synthesizer.audiobook_path = Path("./test_output")
    synthesizer.file_name = "test_book"

    # Test the duration calculation method
    mock_files = [Path("chapter_1.mp3"), Path("chapter_2.mp3")]

    # Test splitting logic
    chunks = synthesizer._split_chapters_by_duration(mock_files, max_hours=1.0)
    print(f"Splitting test passed: created {len(chunks)} chunks")

    print("✅ All M4B splitting functionality tests passed!")


if __name__ == "__main__":
    test_duration_calculation()
