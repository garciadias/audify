#!/usr/bin/env python3
"""
Clean, reprocess, and verify all ebooks from books_deepseek.

For each ebook:
1. Delete existing output dir (start fresh)
2. Run full audiobook pipeline (extract → LLM script → TTS → M4B)
3. Run verification/comparison
4. Collect results

Usage:
    python scripts/clean_reprocess_verify.py          # process all
    python scripts/clean_reprocess_verify.py --test    # process first English book only
    python scripts/clean_reprocess_verify.py --lang en # process one language
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Must run with `uv run` for import venv context

BOOKS_DIR = Path("/home/rd24/Downloads/books_deepseek")
OUTPUT_DIR = Path("data/output")
RESULTS_FILE = Path("batch_compare_results.json")
LOCK_FILE = Path("batch_reprocess.lock")

LLM_MODEL = "ministral-3:14b"  # gemma4:31b & qwen3.6:35b return empty responses. ministral is fast & reliable.
TTS_PROVIDER = "kokoro"
VOICE = "af_bella"

# Language -> (subdir, supporting models)
LANG_CONFIGS = {
    "en": ("en", "af_bella"),
    "es": ("es", "ef_dora"),
    "pt": ("pt", "pf_dora"),
}

# Map from expected output dir name to source
def build_pairs(languages=None):
    pairs = []
    for lang, (subdir, voice) in LANG_CONFIGS.items():
        if languages and lang not in languages:
            continue
        lang_dir = BOOKS_DIR / subdir
        if not lang_dir.exists():
            continue
        for f in sorted(lang_dir.iterdir()):
            if f.suffix.lower() in (".epub", ".pdf"):
                pairs.append((lang, voice, f))
    return pairs


def get_output_dir(source_path: Path, base: Path) -> Path:
    """Match the output dir naming convention used by AudiobookCreator."""
    from audify.utils.text import get_file_name_title
    name = get_file_name_title(source_path.stem)
    # The create_audiobook_series uses folder_safe_name derived from file_name.stem
    folder_safe = name  # PathManager.clean_file_name already does this
    return base / folder_safe


def delete_output(out_dir: Path):
    """Delete output directory if it exists."""
    if out_dir.exists():
        print(f"  🗑️  Deleting {out_dir}")
        import shutil
        shutil.rmtree(out_dir)


def run_pipeline(source: Path, voice: str) -> bool:
    """Run the full audify pipeline for one ebook.
    Returns True on success, False on failure.
    """
    cmd = [
        "uv", "run", "audify",
        str(source),
        "--llm-model", LLM_MODEL,
        "--task", "audiobook",
        "--tts-provider", TTS_PROVIDER,
        "--voice", voice,
        "-y",  # skip confirmation
    ]
    
    print(f"  🚀 Running: {' '.join(cmd)}")
    t0 = time.time()
    
    # Use subprocess with unbuffered Python and very long timeout
    cmd = [
        "uv", "run", "audify",
        str(source),
        "--llm-model", LLM_MODEL,
        "--task", "audiobook",
        "--tts-provider", TTS_PROVIDER,
        "--voice", voice,
        "-y",
    ]
    # Write audify output to a per-book log file (avoid pipe buffer deadlock).
    log_path = Path(f"audify_run_{source.stem[:30]}.log")
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            timeout=28800,  # 8h max per book
        )
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"  ❌ Pipeline FAILED (exit={result.returncode}, {elapsed:.0f}s)")
        # Show last 20 lines from log
        with open(log_path) as log_f:
            lines = log_f.readlines()
        for line in lines[-40:]:
            print(f"     {line.rstrip()}")
        return False
    
    print(f"  ✅ Pipeline OK ({elapsed:.0f}s)")
    return True


def run_verification(source: Path, out_dir: Path) -> dict | None:
    """Run verification and return result dict."""
    m4b_files = sorted(out_dir.glob("*.m4b"))
    if not m4b_files:
        print(f"  ⚠️  No M4B found for verification")
        return None
    
    m4b_path = m4b_files[0]
    print(f"  🔍 Verifying against {m4b_path.name}...")
    
    try:
        from audify.verify import AudiobookVerifier
        verifier = AudiobookVerifier(str(source), str(m4b_path))
        report = verifier.verify()
        report_dict = verifier.generate_report()
        
        dur = report.duration_hint or report.analyze_duration()
        
        result = {
            "source_name": source.name,
            "m4b_name": m4b_path.name,
            "out_dir": out_dir.name,
            "platform": getattr(source.parent, 'name', 'en'),
            "matches": report.matched,
            "total_source": report.total_source,
            "total_audio": report.total_audiobook,
            "missing": len(report.missing),
            "extra": len(report.extra),
            "order_violations": len(report.order_violations),
            "match_pct": report.overall_match_percentage,
            "duration_actual_s": round(dur.actual_duration_s, 1) if dur else None,
            "duration_expected_s": round(dur.expected_duration_s, 1) if dur else None,
            "duration_ratio": round(dur.ratio, 3) if dur and dur.ratio else None,
            "word_count": dur.source_word_count if dur else None,
            "issues": [],
        }
        
        if report.missing:
            result["issues"].append(f"Missing chapters: {', '.join(m.title for m in report.missing)}")
        if report.extra:
            result["issues"].append(f"Extra chapters: {', '.join(e.title for e in report.extra)}")
        if report.order_violations:
            result["issues"].append(f"Order violations: {len(report.order_violations)}")
        
        return result
    
    except Exception as e:
        print(f"  ❌ Verification error: {e}")
        return None


def format_result(r: dict) -> str:
    """Format a result dict for display."""
    if r is None:
        return "❌ NO RESULT"
    
    pct = r.get("match_pct", 0)
    if pct >= 100:
        grade = "✅ PERFECT"
    elif pct >= 90:
        grade = "🟢 GOOD"
    elif pct >= 70:
        grade = "🟡 FAIR"
    elif pct >= 50:
        grade = "🟠 POOR"
    else:
        grade = "🔴 CRITICAL"
    
    ratio = r.get("duration_ratio", 0)
    ratio_str = f"{ratio:.1%}" if ratio else "?"
    
    parts = [
        grade,
        f"match={pct}%",
        f"ch={r.get('matches',0)}/{r.get('total_source',0)}",
        f"missing={r.get('missing',0)}",
        f"extra={r.get('extra',0)}",
        f"dur={ratio_str}",
        f"words={r.get('word_count',0):,}" if r.get('word_count') else "",
    ]
    return " | ".join(p for p in parts if p)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean, reprocess, and verify ebooks")
    parser.add_argument("--test", action="store_true", help="Process only first English book")
    parser.add_argument("--lang", choices=["en", "es", "pt", "all"], default="all",
                       help="Language to process")
    parser.add_argument("--skip-pipeline", action="store_true",
                       help="Skip full pipeline, just verify existing output")
    parser.add_argument("--single", type=str, default=None,
                       help="Process a single source filename (partial match)")
    parser.add_argument("--resume", action="store_true",
                       help="Skip books that already have output AND results in batch_compare_results.json")
    parser.add_argument("--only-verify", action="store_true",
                       help="Re-run verification on existing output without re-running pipeline")
    args = parser.parse_args()
    
    # Lock file to prevent duplicate runs
    if LOCK_FILE.exists():
        print(f"Lock file {LOCK_FILE} exists — another batch may be running.")
        print("Delete it manually if you're sure no other process is active.")
        return
    LOCK_FILE.write_text(str(os.getpid()))
    # Mark that this process owns the lock so cleanup can stay scoped.
    main.lock_owned = True  # type: ignore[attr-defined]
    
    langs = ["en", "es", "pt"] if args.lang == "all" else [args.lang]
    pairs = build_pairs(langs)
    
    if args.single:
        pairs = [(l, v, f) for l, v, f in pairs if args.single.lower() in f.name.lower()]
    
    if args.test:
        pairs = pairs[:1]
        print(f"🧪 TEST MODE: processing 1 book")
    
    print(f"📚 Found {len(pairs)} ebooks to process")
    if not pairs:
        print("No ebooks found!")
        return
    
    # Load existing results for resume/only-verify
    existing_results = {}
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE) as f:
                for r in json.load(f):
                    existing_results[r.get("source_name", "")] = r
        except (json.JSONDecodeError, IOError):
            pass
    
    if args.resume:
        # Filter out already-processed books
        new_pairs = []
        for l, v, f in pairs:
            existing = existing_results.get(f.name)
            if existing and existing.get("status") in ("success", "pipeline_failed", "verification_failed"):
                print(f"  ⏭️  Skipping {f.name} (already processed)")
            else:
                new_pairs.append((l, v, f))
        pairs = new_pairs
        print(f"  Remaining: {len(pairs)} books")
    
    results = list(existing_results.values()) if args.resume else []
    existing_names = {r.get("source_name", "") for r in results}
    total_ok = sum(1 for r in results if r.get("status") == "success")
    total_fail = sum(1 for r in results if r.get("status") != "success")
    
    for idx, (lang, voice, source) in enumerate(pairs, 1):
        if source.name in existing_names:
            continue
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(pairs)}] {lang.upper()}: {source.name}")
        print(f"{'='*70}")
        
        out_dir = get_output_dir(source, OUTPUT_DIR)
        
        # Step 1: Delete existing output
        pipeline_needed = not args.skip_pipeline and not args.only_verify
        if pipeline_needed:
            delete_output(out_dir)
        
        # Step 2: Run pipeline (skip if only-verify or skip-pipeline)
        if pipeline_needed:
            success = run_pipeline(source, voice)
            if not success:
                total_fail += 1
                results.append({
                    "source_name": source.name,
                    "status": "pipeline_failed",
                    "out_dir": out_dir.name if out_dir else None,
                })
                continue
        elif args.only_verify and not out_dir.exists():
            print(f"  ⚠️  No output dir for {source.name}, skipping")
            continue
        
        # Step 3: Verify
        result = run_verification(source, out_dir)
        if result:
            result["status"] = "success"
            results.append(result)
            print(f"  {format_result(result)}")
            total_ok += 1
        else:
            total_fail += 1
            results.append({
                "source_name": source.name,
                "status": "verification_failed",
                "out_dir": out_dir.name if out_dir else None,
            })
        
        # Save incrementally in case of crash
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY: {total_ok} passed, {total_fail} failed of {len(pairs)} total")
    print(f"{'='*70}")
    
    for r in results:
        if r.get("status") == "success":
            print(f"  {format_result(r)}")
        else:
            print(f"  ❌ {r.get('source_name','?')}: {r.get('status','?')}")
    
    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to {RESULTS_FILE}")
    
    # Clean lock file (only if we own it)
    _release_lock()

    return 0 if total_fail == 0 else 1


def _release_lock() -> None:
    """Remove the lock file only if this process owns it.

    Guards against a second invocation (which returned early without acquiring
    the lock) deleting the lock held by the original running process.
    """
    if not getattr(main, "lock_owned", False):
        return
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text().strip() == str(os.getpid()):
            LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


if __name__ == "__main__":
    try:
        ret = main()
    except Exception:
        _release_lock()
        raise
    _release_lock()
    sys.exit(ret)
