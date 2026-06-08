#!/usr/bin/env python3
"""
Batch process all books from books_deepseek using direct httpx API calls.
Much faster than audify subprocess - ~25min per book.

Usage:
    uv run python3 scripts/batch_direct.py              # all books
    uv run python3 scripts/batch_direct.py --test        # first English book
    uv run python3 scripts/batch_direct.py --lang pt     # Portuguese only
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BOOKS_DIR = Path("/home/rd24/Downloads/books_deepseek")
OUTPUT_DIR = Path("data/output")
RESULTS_FILE = Path("batch_compare_results.json")

VOICES = {"en": "af_bella", "es": "ef_dora", "pt": "pf_dora"}


def build_pairs():
    pairs = []
    for lang, voice in VOICES.items():
        lang_dir = BOOKS_DIR / lang
        if not lang_dir.exists():
            continue
        for f in sorted(lang_dir.iterdir()):
            if f.suffix.lower() in (".epub", ".pdf"):
                pairs.append((lang, voice, f))
    return pairs


def run_verification(pairs):
    """Run verification on all books in pairs. Returns 0 on success."""
    from audify.verify import AudiobookVerifier
    from audify.utils.text import get_file_name_title
    
    print(f"\n🔄 Running verification on all books...")
    results = []
    for lang, voice, source in pairs:
        expected_name = get_file_name_title(source.stem)
        m4b_dir = OUTPUT_DIR / expected_name
        m4b_files = sorted(m4b_dir.glob("*.m4b")) if m4b_dir.exists() else []
        m4b = m4b_files[0] if m4b_files else None

        if not m4b:
            print(f"  ⚠️  No M4B for {source.name}")
            continue

        try:
            verifier = AudiobookVerifier(str(source), str(m4b))
            report = verifier.verify()
            dur = report.duration_hint or report.analyze_duration()

            result = {
                "source_name": source.name,
                "m4b_name": m4b.name,
                "dir_name": m4b.parent.name,
                "lang": lang,
                "matches": report.matched,
                "total_source": report.total_source,
                "total_audio": report.total_audiobook,
                "missing": len(report.missing),
                "extra": len(report.extra),
                "order_violations": len(report.order_violations),
                "match_pct": report.overall_match_percentage,
                "duration_ratio": round(dur.ratio, 3) if dur and dur.ratio else None,
                "word_count": dur.source_word_count if dur else None,
            }
            results.append(result)

            pct = result["match_pct"]
            grade = "✅" if pct >= 100 else "🟢" if pct >= 90 else "🟡" if pct >= 70 else "🔴"
            print(f"  {grade} {source.name}: {result['matches']}/{result['total_source']} ({pct}%)")
        except Exception as e:
            print(f"  ❌ Verify failed for {source.name}: {e}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {RESULTS_FILE}")

    if results:
        pcts = [r["match_pct"] for r in results if r.get("match_pct") is not None]
        if pcts:
            avg = sum(pcts) / len(pcts)
            perfect = sum(1 for p in pcts if p >= 100)
            good = sum(1 for p in pcts if 90 <= p < 100)
            poor = sum(1 for p in pcts if p < 90)
            print(f"\n📊 Summary: avg={avg:.1f}%, perfect={perfect}, good={good}, poor={poor} of {len(pcts)} books")

    return 0


def main():
    test_mode = "--test" in sys.argv
    resume = "--resume" in sys.argv
    lang_filter = None
    for a in sys.argv[1:]:
        if a.startswith("--lang="):
            lang_filter = a.split("=", 1)[1]
        elif a == "--lang" and sys.argv.index(a) + 1 < len(sys.argv):
            idx = sys.argv.index(a)
            lang_filter = sys.argv[idx + 1]

    pairs = build_pairs()
    if lang_filter:
        pairs = [(l, v, f) for l, v, f in pairs if l == lang_filter]
    if test_mode:
        pairs = pairs[:1]
    
    all_pairs = list(pairs)  # Keep for verification at the end
    
    if resume:
        from audify.utils.text import get_file_name_title
        completed = set()
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and list(d.glob("*.m4b")):
                completed.add(d.name)
        new_pairs = []
        for l, v, f in pairs:
            expected_name = get_file_name_title(f.stem)
            if expected_name in completed:
                print(f"  ⏭️  Skipping {f.name} (already processed)")
            else:
                new_pairs.append((l, v, f))
        pairs = new_pairs
    
    print(f"📚 Found {len(pairs)} books to process")
    if not pairs:
        print("All books already processed!")
        return run_verification(all_pairs)
    
    total_start = time.time()

    for idx, (lang, voice, source) in enumerate(pairs, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(pairs)}] {lang.upper()}: {source.name}")
        print(f"{'='*70}")

        t0 = time.time()
        safe_name = source.stem[:40].replace("'", "").replace('"', "")
        log_path = OUTPUT_DIR.parent / f"book_{safe_name}.log"
        
        # Phase 1: Process-only (LLM script generation) with 30min timeout
        print(f"  Phase 1: LLM scripts (30min timeout)...")
        cmd = [
            "uv", "run", "audify", str(source),
            "--llm-model", "ministral-3:14b",
            "--task", "audiobook",
            "--process-only",
            "-y",
        ]
        with open(log_path, "w") as log_f:
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode != 0:
            print(f"  ❌ Phase 1 FAILED (exit={result.returncode})")
            lines = open(log_path).readlines()
            for line in lines[-10:]:
                if line.strip():
                    print(f"     {line.strip()}")
            continue
        print(f"  ✅ Phase 1 done ({time.time()-t0:.0f}s)")
        
        # Phase 2: Synthesize-only (TTS + M4B) with 180min timeout
        print(f"  Phase 2: TTS + M4B (180min timeout)...")
        t1 = time.time()
        cmd = [
            "uv", "run", "audify", str(source),
            "--synthesize-only",
            "--voice", voice,
            "--tts-provider", "kokoro",
            "-y",
        ]
        with open(log_path, "a") as log_f:
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=10800)
        elapsed = time.time() - t0
        tts_elapsed = time.time() - t1
        
        if result.returncode != 0:
            print(f"  ❌ Phase 2 FAILED (exit={result.returncode}, {tts_elapsed:.0f}s)")
            lines = open(log_path).readlines()
            for line in lines[-10:]:
                if line.strip():
                    print(f"     {line.strip()}")
            continue
        
        print(f"  ✅ Done ({elapsed:.0f}s)")

    elapsed_total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"✅ ALL DONE in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"{'='*70}")

    return run_verification(all_pairs)


if __name__ == "__main__":
    sys.exit(main())
