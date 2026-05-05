#!/usr/bin/env python3
"""
Batch analyze all audiobooks using the verification tool.

Generates a comprehensive audit report with metrics for all processed audiobooks.
"""

import json
import logging
from pathlib import Path
from typing import Optional
import sys

try:
    from audify.verify import verify_audiobook_against_source, VerificationReport
    from audify.readers.ebook import EpubReader
except ImportError:
    print("Error: audify module not found. Run from project root.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_source_file(audiobook_dir: Path) -> Optional[Path]:
    """Find the source file for an audiobook."""
    parent = audiobook_dir.parent
    book_name = audiobook_dir.name
    
    # Try to find source file with matching name
    for ext in ['.epub', '.pdf', '.txt']:
        source_file = parent / f"{book_name}{ext}"
        if source_file.exists():
            return source_file
    
    # Try to find any ebook file in parent
    for source_file in parent.glob("*.epub"):
        return source_file
    for source_file in parent.glob("*.pdf"):
        return source_file
    
    return None


def find_audiobook_file(audiobook_dir: Path) -> Optional[Path]:
    """Find the audiobook file (M4B or MP3)."""
    for ext in ['.m4b', '.mp3']:
        audiobook_file = audiobook_dir / f"audiobook{ext}"
        if audiobook_file.exists():
            return audiobook_file
    
    # Try any m4b or mp3
    for audiobook_file in audiobook_dir.glob("*.m4b"):
        return audiobook_file
    for audiobook_file in audiobook_dir.glob("*.mp3"):
        return audiobook_file
    
    return None


def analyze_audiobook(source_path: Path, audiobook_path: Path) -> dict:
    """Analyze a single audiobook against its source."""
    try:
        logger.info(f"Analyzing: {source_path.name} → {audiobook_path.name}")
        
        result = verify_audiobook_against_source(source_path, audiobook_path)
        
        # Extract word count from source
        try:
            if source_path.suffix.lower() == '.epub':
                reader = EpubReader(str(source_path))
                content = reader.get_text()
                word_count = len(content.split())
            else:
                word_count = 0
        except Exception as e:
            logger.warning(f"Could not extract word count: {e}")
            word_count = 0
        
        return {
            'source': source_path.name,
            'audiobook': audiobook_path.name,
            'duration_ratio': round(result.duration_ratio, 3) if result.duration_ratio else None,
            'duration_seconds': result.audiobook_duration_seconds,
            'expected_duration_seconds': result.expected_duration_seconds,
            'chapters_matched': result.chapters_matched_count,
            'chapters_expected': result.expected_chapter_count,
            'chapters_extra': result.extra_chapters_count if hasattr(result, 'extra_chapters_count') else 0,
            'chapters_missing': result.missing_chapters_count if hasattr(result, 'missing_chapters_count') else 0,
            'word_count': word_count,
            'issues': [issue.message if hasattr(issue, 'message') else str(issue) 
                      for issue in (result.issues or [])],
            'issues_count': len(result.issues or []),
            'status': 'OK' if (not result.issues or len(result.issues) == 0) else 'ISSUES',
        }
    
    except Exception as e:
        logger.error(f"Error analyzing {audiobook_path.name}: {e}")
        return {
            'source': source_path.name,
            'audiobook': audiobook_path.name,
            'error': str(e),
            'status': 'ERROR',
        }


def analyze_all_audiobooks(output_dir: Path) -> list:
    """Analyze all audiobooks in output directory."""
    logger.info(f"Scanning output directory: {output_dir}")
    
    results = []
    audiobook_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(audiobook_dirs)} audiobook directories")
    
    for i, audiobook_dir in enumerate(audiobook_dirs, 1):
        logger.info(f"\n[{i}/{len(audiobook_dirs)}] Processing {audiobook_dir.name}")
        
        # Find source and audiobook files
        source_file = find_source_file(audiobook_dir)
        audiobook_file = find_audiobook_file(audiobook_dir)
        
        if not source_file:
            logger.warning(f"  ❌ No source file found")
            continue
        
        if not audiobook_file:
            logger.warning(f"  ❌ No audiobook file found")
            continue
        
        # Analyze
        analysis = analyze_audiobook(source_file, audiobook_file)
        results.append(analysis)
        
        # Print summary
        if analysis.get('status') == 'ERROR':
            logger.error(f"  ❌ Error: {analysis.get('error')}")
        elif analysis.get('status') == 'ISSUES':
            logger.warning(f"  ⚠️  Issues: {analysis.get('issues_count')}")
            for issue in analysis.get('issues', [])[:3]:
                logger.warning(f"     - {issue}")
        else:
            ratio = analysis.get('duration_ratio', 0)
            chapters = analysis.get('chapters_matched', 0)
            expected = analysis.get('chapters_expected', 0)
            logger.info(f"  ✅ Duration Ratio: {ratio:.1%}, Chapters: {chapters}/{expected}")
    
    return results


def print_summary(results: list):
    """Print analysis summary."""
    print("\n" + "="*80)
    print("AUDIOBOOK ANALYSIS SUMMARY")
    print("="*80 + "\n")
    
    if not results:
        print("No audiobooks analyzed.\n")
        return
    
    # Overall statistics
    total = len(results)
    errors = sum(1 for r in results if r.get('status') == 'ERROR')
    issues = sum(1 for r in results if r.get('status') == 'ISSUES')
    ok = sum(1 for r in results if r.get('status') == 'OK')
    
    print(f"Total Audiobooks:  {total}")
    print(f"  ✅ OK:           {ok}")
    print(f"  ⚠️  Issues:       {issues}")
    print(f"  ❌ Errors:       {errors}\n")
    
    # Duration ratio statistics
    valid_ratios = [r.get('duration_ratio', 0) for r in results 
                    if r.get('duration_ratio') is not None]
    if valid_ratios:
        avg_ratio = sum(valid_ratios) / len(valid_ratios)
        min_ratio = min(valid_ratios)
        max_ratio = max(valid_ratios)
        
        print(f"Duration Ratios:")
        print(f"  Average:         {avg_ratio:.1%}")
        print(f"  Range:           {min_ratio:.1%} - {max_ratio:.1%}\n")
    
    # Problem books
    problem_books = [r for r in results if r.get('duration_ratio', 1) < 0.60]
    if problem_books:
        print(f"Books with Duration < 60% ({len(problem_books)}):")
        for book in sorted(problem_books, key=lambda x: x.get('duration_ratio', 0)):
            ratio = book.get('duration_ratio', 0)
            print(f"  {ratio:.1%} - {book.get('source', 'unknown')}")
        print()
    
    # Missing chapters
    missing_chapter_books = [r for r in results 
                            if r.get('chapters_matched', 0) < r.get('chapters_expected', 1)]
    if missing_chapter_books:
        print(f"Books with Missing Chapters ({len(missing_chapter_books)}):")
        for book in missing_chapter_books:
            matched = book.get('chapters_matched', 0)
            expected = book.get('chapters_expected', 0)
            print(f"  {matched}/{expected} - {book.get('source', 'unknown')}")
        print()
    
    print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze all audiobooks using verification tool'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/output'),
        help='Directory containing processed audiobooks'
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=Path('audiobook_analysis_report.json'),
        help='Output file for JSON report'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate output directory
    if not args.output_dir.exists():
        logger.error(f"Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    # Analyze all audiobooks
    logger.info("Starting audiobook analysis...")
    results = analyze_all_audiobooks(args.output_dir)
    
    # Save report
    logger.info(f"\nSaving report to {args.report}")
    with open(args.report, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report saved: {args.report}")
    
    # Print summary
    print_summary(results)
    
    # Return appropriate exit code
    errors = sum(1 for r in results if r.get('status') == 'ERROR')
    sys.exit(1 if errors > 0 else 0)


if __name__ == '__main__':
    main()
