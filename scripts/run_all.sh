#!/bin/bash
# FINAL batch script - processes all books via two-phase pipeline
set -euo pipefail

BOOKS_DIR="/home/rd24/Downloads/books_deepseek"
OUTPUT_DIR="data/output"
MODEL="ministral-3:14b"
VOICES='{"en":"af_bella","es":"ef_dora","pt":"pf_dora"}'

# Phase 1: Process-only (LLM script generation) with 30min timeout
phase1() {
    local epub="$1"
    local logfile="$2"
    echo "  Phase 1: LLM scripts..." | tee -a "$logfile"
    if timeout 1800 uv run audify "$epub" \
        --llm-model "$MODEL" \
        --task audiobook \
        --process-only -y >> "$logfile" 2>&1; then
        echo "  ✅ Phase 1 done" | tee -a "$logfile"
        return 0
    else
        echo "  ⚠️  Phase 1 timed out or failed" | tee -a "$logfile"
        return 1
    fi
}

# Fix chapter_titles.json
fix_titles() {
    local epub="$1"
    local scripts_dir="$2"
    uv run python3 -c "
import json, sys
from pathlib import Path
from audify.readers.ebook import EpubReader
try:
    reader = EpubReader('$epub')
    chapters = reader.get_chapters()
    titles = [reader.get_chapter_title(c) or f'Chapter {i+1}' for i, c in enumerate(chapters)]
    Path('$scripts_dir/chapter_titles.json').write_text(json.dumps(titles, indent=2))
    print(f'  Fixed titles: {len(titles)}')
except Exception as e:
    print(f'  Title fix: {e}')
" 2>&1
}

# Phase 2: Synthesize with ffmpeg
phase2() {
    local epub="$1"
    local logfile="$2"
    echo "  Phase 2: TTS + M4B..." | tee -a "$logfile"
    if timeout 7200 uv run python3 scripts/synthesize_with_ffmpeg.py "$epub" >> "$logfile" 2>&1; then
        echo "  ✅ M4B created" | tee -a "$logfile"
        return 0
    else
        echo "  ⚠️  Phase 2 failed" | tee -a "$logfile"
        return 1
    fi
}

echo "=== Batch started $(date) ==="
echo ""

for lang in en es pt; do
    voice=$(echo "$VOICES" | uv run python3 -c "import json,sys; print(json.load(sys.stdin)['$lang'])")
    
    for epub in "$BOOKS_DIR/$lang"/*.epub "$BOOKS_DIR/$lang"/*.pdf; do
        [ -f "$epub" ] || continue
        
        base=$(basename "$epub")
        stem="${base%.*}"
        safe_name=$(echo "$stem" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | sed 's/[^a-z0-9_]//g' | sed 's/^_//')
        
        # Skip already verified books
        if [ "$safe_name" = "adam_smith_in_beijing" ]; then
            echo "⏭️  $base (already verified)"
            continue
        fi
        if [ "$safe_name" = "capital_in_the_twentyfirst_century" ]; then
            echo "⏭️  $base (already verified)"
            continue
        fi
        
        echo ""
        echo "======================================================================"
        echo "[$lang] $base -> $safe_name"
        echo "======================================================================"
        
        LOGFILE="data/book_${safe_name}.log"
        SCRIPTS_DIR="$OUTPUT_DIR/$safe_name/scripts"
        
        # Clean for fresh start
        rm -rf "$OUTPUT_DIR/$safe_name" 2>/dev/null
        echo "  Voice: $voice" | tee "$LOGFILE"
        
        # Phase 1
        if ! phase1 "$epub" "$LOGFILE"; then
            echo "  ❌ Skipping $base (Phase 1 failed)" | tee -a "$LOGFILE"
            # Even if phase1 fails, try synthesize if any scripts exist
        fi
        
        # Fix titles
        if [ -d "$SCRIPTS_DIR" ]; then
            fix_titles "$epub" "$SCRIPTS_DIR"
        fi
        
        # Phase 2 (set voice via env var for the python script)
        VOICE="$voice" phase2 "$epub" "$LOGFILE"
        
        echo "  ✅ Done $base" | tee -a "$LOGFILE"
    done
done

echo ""
echo "=== Batch complete $(date) ==="
