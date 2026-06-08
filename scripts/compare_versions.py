#!/usr/bin/env python3
"""
Compare verification results between two audiobook versions.

Useful for A/B testing improvements to the processing algorithm.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_report(report_path: Path) -> List[Dict[str, Any]]:
    """Load a verification report."""
    with open(report_path) as f:
        return json.load(f)


def find_matching_book(book_name: str, other_results: List[Dict]) -> Dict:
    """Find a book in results by name (with fuzzy matching)."""
    for result in other_results:
        source = result.get('source', '').lower()
        if book_name.lower() in source or source in book_name.lower():
            return result
    return {}


def compare_results(baseline: List[Dict], improved: List[Dict]) -> Dict:
    """Compare two sets of results."""
    comparisons = []
    
    for baseline_book in baseline:
        if baseline_book.get('status') == 'ERROR':
            continue
        
        book_name = baseline_book.get('source', '')
        improved_book = find_matching_book(book_name, improved)
        
        if not improved_book:
            print(f"⚠️  Warning: {book_name} not found in improved results")
            continue
        
        baseline_ratio = baseline_book.get('duration_ratio') or 0
        improved_ratio = improved_book.get('duration_ratio') or 0
        ratio_improvement = ((improved_ratio - baseline_ratio) / baseline_ratio * 100 
                            if baseline_ratio > 0 else 0)
        
        baseline_chapters = baseline_book.get('chapters_matched', 0)
        improved_chapters = improved_book.get('chapters_matched', 0)
        chapter_improvement = improved_chapters - baseline_chapters
        
        baseline_issues = baseline_book.get('issues_count', 0)
        improved_issues = improved_book.get('issues_count', 0)
        issue_improvement = baseline_issues - improved_issues
        
        comparison = {
            'source': book_name,
            'baseline_ratio': round(baseline_ratio, 3),
            'improved_ratio': round(improved_ratio, 3),
            'ratio_improvement_percent': round(ratio_improvement, 1),
            'baseline_chapters': baseline_chapters,
            'improved_chapters': improved_chapters,
            'chapter_improvement': chapter_improvement,
            'baseline_issues': baseline_issues,
            'improved_issues': improved_issues,
            'issue_improvement': issue_improvement,
        }
        
        comparisons.append(comparison)
    
    return {
        'total_books': len(comparisons),
        'comparisons': comparisons,
        'summary': summarize_comparisons(comparisons),
    }


def summarize_comparisons(comparisons: List[Dict]) -> Dict:
    """Generate summary statistics."""
    if not comparisons:
        return {}
    
    # Calculate averages
    avg_ratio_improvement = (
        sum(c['ratio_improvement_percent'] for c in comparisons) / len(comparisons)
    )
    
    avg_chapter_improvement = (
        sum(c['chapter_improvement'] for c in comparisons) / len(comparisons)
    )
    
    avg_issue_improvement = (
        sum(c['issue_improvement'] for c in comparisons) / len(comparisons)
    )
    
    # Count improved/regressed
    improved_count = sum(1 for c in comparisons if c['ratio_improvement_percent'] > 0)
    regressed_count = sum(1 for c in comparisons if c['ratio_improvement_percent'] < 0)
    same_count = sum(1 for c in comparisons if c['ratio_improvement_percent'] == 0)
    
    # Find best/worst
    best_improvement = max(comparisons, key=lambda x: x['ratio_improvement_percent'])
    worst_regression = min(comparisons, key=lambda x: x['ratio_improvement_percent'])
    
    return {
        'avg_ratio_improvement_percent': round(avg_ratio_improvement, 1),
        'avg_chapter_improvement': round(avg_chapter_improvement, 2),
        'avg_issue_improvement': round(avg_issue_improvement, 2),
        'books_improved': improved_count,
        'books_regressed': regressed_count,
        'books_same': same_count,
        'improvement_rate': round((improved_count / len(comparisons)) * 100, 1),
        'best_improvement_percent': round(best_improvement['ratio_improvement_percent'], 1),
        'best_improvement_book': best_improvement['source'],
        'worst_regression_percent': round(worst_regression['ratio_improvement_percent'], 1),
        'worst_regression_book': worst_regression['source'],
    }


def print_comparison_report(comparison: Dict):
    """Print formatted comparison report."""
    print("\n" + "="*100)
    print("PROCESSING IMPROVEMENT COMPARISON REPORT")
    print("="*100 + "\n")
    
    summary = comparison.get('summary', {})
    
    # Overall metrics
    print("OVERALL IMPROVEMENTS")
    print("-" * 100)
    print(f"Average Duration Ratio Improvement:  {summary.get('avg_ratio_improvement_percent', 0):>6.1f}%")
    print(f"Average Chapter Improvement:        {summary.get('avg_chapter_improvement', 0):>6.2f}")
    print(f"Average Issue Reduction:            {summary.get('avg_issue_improvement', 0):>6.2f}")
    print(f"Improvement Rate:                   {summary.get('improvement_rate', 0):>6.1f}%")
    print(f"  ✅ Improved:                      {summary.get('books_improved', 0):>6} books")
    print(f"  ➡️  Same:                         {summary.get('books_same', 0):>6} books")
    print(f"  ❌ Regressed:                     {summary.get('books_regressed', 0):>6} books")
    print()
    
    # Best and worst
    print(f"Best Improvement:   {summary.get('best_improvement_percent', 0):>+6.1f}% - {summary.get('best_improvement_book', 'N/A')}")
    print(f"Worst Regression:   {summary.get('worst_regression_percent', 0):>+6.1f}% - {summary.get('worst_regression_book', 'N/A')}")
    print()
    
    # Detailed results
    print("DETAILED RESULTS")
    print("-" * 100)
    print(f"{'Book Name':<50} {'Duration Ratio':<20} {'Chapters':<15} {'Issues'}")
    print(f"{'':50} {'Baseline → Improved':<20} {'∆':<15} {'∆'}")
    print("-" * 100)
    
    for comp in sorted(comparison.get('comparisons', []), 
                      key=lambda x: x['ratio_improvement_percent'], 
                      reverse=True):
        source = comp['source'][:49]
        baseline = comp['baseline_ratio']
        improved = comp['improved_ratio']
        change = comp['ratio_improvement_percent']
        
        ch_baseline = comp['baseline_chapters']
        ch_improved = comp['improved_chapters']
        ch_change = comp['chapter_improvement']
        
        issue_baseline = comp['baseline_issues']
        issue_improved = comp['improved_issues']
        issue_change = comp['issue_improvement']
        
        # Format change indicator
        if change > 0:
            change_str = f"✅ +{change:.1f}%"
        elif change < 0:
            change_str = f"❌ {change:.1f}%"
        else:
            change_str = "➡️  0%"
        
        ratio_str = f"{baseline:.1%} → {improved:.1%}"
        chapter_str = f"{ch_baseline} → {ch_improved} ({ch_change:+d})"
        issue_str = f"{issue_baseline} → {issue_improved} ({issue_change:+d})"
        
        print(f"{source:<50} {ratio_str:<20} {chapter_str:<15} {issue_str}")
    
    print("\n" + "="*100 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare audiobook verification results between two versions'
    )
    parser.add_argument(
        'baseline',
        type=Path,
        help='Baseline verification report (JSON)'
    )
    parser.add_argument(
        'improved',
        type=Path,
        help='Improved verification report (JSON)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('comparison_report.json'),
        help='Output file for detailed comparison (JSON)'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Only output JSON, no console report'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    for report_file in [args.baseline, args.improved]:
        if not report_file.exists():
            print(f"Error: Report file not found: {report_file}")
            sys.exit(1)
    
    # Load reports
    print(f"Loading baseline: {args.baseline}")
    baseline_results = load_report(args.baseline)
    
    print(f"Loading improved: {args.improved}")
    improved_results = load_report(args.improved)
    
    # Compare
    print("Comparing results...")
    comparison = compare_results(baseline_results, improved_results)
    
    # Save JSON report
    print(f"Saving detailed report: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print report
    if not args.json_only:
        print_comparison_report(comparison)
    
    print(f"✅ Comparison complete!")


if __name__ == '__main__':
    main()
