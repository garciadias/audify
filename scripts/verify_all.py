#!/usr/bin/env python3
"""Verify all processed audiobooks from books_deepseek."""
import json
from pathlib import Path

from audify.verify import AudiobookVerifier
from audify.utils.text import get_file_name_title

BOOKS_DIR = Path("/home/rd24/Downloads/books_deepseek")
OUTPUT_DIR = Path("data/output")
RESULTS = []

for lang in ["en", "es", "pt"]:
    for f in sorted((BOOKS_DIR / lang).iterdir()):
        if f.suffix.lower() not in (".epub", ".pdf"):
            continue
        book_name = get_file_name_title(f.stem)
        out_dir = OUTPUT_DIR / book_name
        m4b_files = sorted(out_dir.glob("*.m4b")) if out_dir.exists() else []
        if not m4b_files:
            print(f"\u23ed\ufe0f  NO M4B: {f.name}")
            continue

        m4b = m4b_files[0]
        try:
            verifier = AudiobookVerifier(str(f), str(m4b))
            report = verifier.verify()
            match_pct = report.overall_match_percentage
            result = {
                "source": f.name, "lang": lang,
                "matched": report.matched, "total": report.total_source,
                "match_pct": match_pct,
                "missing": len(report.missing), "extra": len(report.extra),
            }

            if match_pct >= 90:
                grade = "\U0001f7e2"
            elif match_pct >= 70:
                grade = "\U0001f7e1"
            else:
                grade = "\U0001f534"

            missing_info = ""
            if report.missing:
                titles = [m.title for m in report.missing[:3]]
                missing_info = " (" + ", ".join(titles) + ")"

            out = f"{grade} {f.name}: {report.matched}/{report.total_source} ({match_pct}%){missing_info}"
            print(out)
            RESULTS.append(result)
        except Exception as e:
            print(f"\u274c {f.name}: {e}")

if RESULTS:
    valid = [r for r in RESULTS if r.get("match_pct") is not None]
    if valid:
        avg = sum(r["match_pct"] for r in valid) / len(valid)
        perfect = sum(1 for r in valid if r["match_pct"] >= 100)
        good = sum(1 for r in valid if 90 <= r["match_pct"] < 100)
        poor = sum(1 for r in valid if r["match_pct"] < 90)
        print()
        print(f"Summary: {len(valid)} books, avg={avg:.1f}%, perfect={perfect}, good={good}, poor={poor}")

with open("verify_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)
print("Saved to verify_results.json")
