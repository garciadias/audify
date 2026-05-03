#!/usr/bin/env python3
r"""
Test redundancy analyzer using leave-one-out coverage analysis.

Strategy (per-test individual measurement):
  1. Run each test individually with coverage, collect which (file, line)
     items it covers.
  2. Build a per-line frequency map: for each (file, line), count how
     many tests cover it.
  3. A test is redundant if every line it covers has count >= 2
     (i.e., some other test also covers it).

This is mathematically equivalent to the "run all minus one" leave-one-out
approach, but ~10x faster because each iteration runs a single test instead
of 801 tests.

Proof of equivalence:
  - T is redundant under LOO  <=>  removing T from full suite loses 0 lines
  - <=> every line covered by T is also covered by some test in S\{T}
  - <=> for every line l in coverage(T), count(l) >= 2

Run with:  uv run python tests/leave_one_out_analysis.py [--subset N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):  # noqa: ARG001
        return iterable

# ── paths ───────────────────────────────────────────────────────────────
SINGLE_DATA_FILE = Path(".coverage.single")
BASELINE_DATA_FILE = Path(".coverage.baseline")
RESULTS_JSON = Path("test_redundancy_results.json")
REDACTED_TESTS = Path("redundant_tests.txt")

PYTEST_BASE = [
    "uv", "run", "pytest",
    "--cov=audify",
    "-q", "-p", "no:warnings",
]


def get_all_test_ids() -> list[str]:
    """Collect every test node-id using pytest --collect-only."""
    result = subprocess.run(
        ["uv", "run", "pytest", "--collect-only", "-q"],
        capture_output=True, text=True,
    )
    ids: list[str] = []
    for line in result.stdout.strip().splitlines():
        cleaned = re.sub(r"\s*\[\d+\]\s*$", "", line).strip()
        if cleaned and "::test_" in cleaned:
            ids.append(cleaned)
    return ids


def _run_pytest(extra_args: list[str], data_file: Path) -> subprocess.CompletedProcess[str]:
    """Run pytest + coverage, writing data to *data_file*."""
    cmd = PYTEST_BASE + extra_args
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(data_file)
    if data_file.exists():
        data_file.unlink()
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def get_coverage_set(data_file: Path) -> set[tuple[str, int]]:
    """Return {(relative_path, lineno)} covered by *data_file*."""
    import coverage
    cov = coverage.Coverage(data_file=str(data_file), source=["audify"])
    cov.load()
    data = cov.get_data()
    covered: set[tuple[str, int]] = set()
    for fname in data.measured_files():
        rel = os.path.relpath(fname)
        for lineno in data.lines(fname):
            covered.add((rel, lineno))
    try:
        data_file.unlink()
    except FileNotFoundError:
        pass
    return covered


# ── analysis ───────────────────────────────────────────────────────────

def analyze(test_ids: list[str], *, subset: int | None = None) -> None:
    """
    Analyze which tests are redundant (their coverage is fully subsumed
    by other tests).

    Parameters
    ----------
    test_ids : list[str]
        Full list of pytest node-ids.
    subset  : int | None
        If given, analyze at most *subset* tests (for a quick sanity-
        check).  Set to *None* for a complete run.
    """
    ids = test_ids[:subset] if subset is not None else test_ids
    total = len(ids)

    # ── 1. baseline (all tests together) ───────────────────────────────
    print(f"[baseline] Running ALL {len(test_ids)} tests together ...")
    _run_pytest([], BASELINE_DATA_FILE)
    baseline_set = get_coverage_set(BASELINE_DATA_FILE)
    print(f"[baseline] Covered {len(baseline_set)} (file, line) items.\n")

    print(f"[measure] Running each of {total} tests individually ...\n")

    # Map: (file, line) -> count of tests covering it
    line_counts: dict[tuple[str, int], int] = defaultdict(int)

    # Per-test: set of (file, line) it covers
    test_coverage: dict[str, set[tuple[str, int]]] = {}

    for tid in tqdm(ids, desc="Measure"):
        proc = _run_pytest([tid], SINGLE_DATA_FILE)
        if proc.returncode != 0:
            # Test failed to run -> it doesn't contribute coverage
            test_coverage[tid] = set()
            continue

        covered = get_coverage_set(SINGLE_DATA_FILE)
        test_coverage[tid] = covered
        for line in covered:
            line_counts[line] += 1

    # ── 3. determine redundancy ────────────────────────────────────────
    info: list[dict] = []

    for tid in ids:
        covered = test_coverage.get(tid, set())

        if not covered:
            # Test covers nothing -> redundant (or could not run)
            info.append({
                "test": tid,
                "redundant": True,
                "lines_lost": 0,
                "total_covered": 0,
            })
            continue

        # lines_lost = lines covered by THIS test that ONLY this test covers
        # i.e., lines where count == 1
        lines_uniquely_covered = sum(
            1 for line in covered if line_counts[line] == 1
        )
        info.append({
            "test": tid,
            "redundant": lines_uniquely_covered == 0,
            "lines_lost": lines_uniquely_covered,
            "total_covered": len(covered),
        })

    # ── 4. report ──────────────────────────────────────────────
    redundant_tests = [i for i in info if i["redundant"]]
    contributing = [i for i in info if not i["redundant"]]

    print(f"\n{'=' * 70}")
    print("  REDUNDANCY ANALYSIS COMPLETE")
    print(f"  Total tests analysed   : {total}")
    print(f"  Redundant (0 unique)   : {len(redundant_tests)}")
    print(f"  Contributing           : {len(contributing)}")
    print(f"{'=' * 70}\n")

    if redundant_tests:
        print("Redundant tests (every line they cover is covered by another test):")
        for r in redundant_tests:
            print(f"  * {r['test']}")

    if contributing:
        print("\nTop-20 most impactful tests (by uniquely covered lines):")
        top = sorted(contributing, key=lambda x: x["lines_lost"], reverse=True)[:20]
        for c in top:
            print(f"  + {c['lines_lost']:>5} unique lines  {c['test']}")

    # persist results
    RESULTS_JSON.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nDetailed results -> {RESULTS_JSON}")

    REDACTED_TESTS.write_text(
        "\n".join(r["test"] for r in redundant_tests),
        encoding="utf-8",
    )
    print(f"Redundant list   -> {REDACTED_TESTS}")


# ── entry-point ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-out test redundancy analyzer",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Only analyse the first N tests (quick sanity-check)",
    )
    args = parser.parse_args()

    ids = get_all_test_ids()
    if not ids:
        print("ERROR: no tests found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Collected {len(ids)} test nodes.\n")
    analyze(ids, subset=args.subset)


if __name__ == "__main__":
    main()
