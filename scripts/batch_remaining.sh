#!/bin/bash
# Simple batch script for remaining books
# Runs process-only then synthesize-only for each book
set -euo pipefail

BOOKS_DIR="/home/rd24/Downloads/books_deepseek"
OUTPUT_DIR="data/output"
LOG_DIR="data"
MODEL="ministral-3:14b"

# Books that already have M4B (skip)
already_done() {
    local book_name="$1"
    local out_dir="$OUTPUT_DIR/$book_name"
    [ -d "$out_dir" ] && ls "$out_dir"/*.m4b &>/dev/null
}

echo "=== Batch started at $(date) ==="

for lang in en es pt; do
    for epub in "$BOOKS_DIR/$lang"/*.epub "$BOOKS_DIR/$lang"/*.pdf; do
        [ -f "$epub" ] || continue
        
        base=$(basename "$epub")
        stem="${base%.*}"
        
        # Determine output name
        safe_name=$(echo "$stem" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | sed 's/[^a-z0-9_]//g')
        
        echo ""
        echo "======================================================================"
        echo "[$lang] $base -> $safe_name"
        echo "======================================================================"
        
        # Only skip Adam Smith which we already verified
        if [ "$safe_name" = "adam_smith_in_beijing" ]; then
            echo "  ⏭️  Already verified, skipping"
            continue
        fi
        
        # Clean existing output for this book (start fresh)
        if [ -d "$OUTPUT_DIR/$safe_name" ]; then
            echo "  🗑️  Cleaning existing output"
            rm -rf "$OUTPUT_DIR/$safe_name"
        fi
        
        BOOK_LOG="$LOG_DIR/book_${safe_name}.log"
        
        # Phase 1: Process-only (LLM scripts)
        echo "  Phase 1: LLM scripts..."
        if timeout 1800 uv run audify "$epub" \
            --llm-model "$MODEL" \
            --task audiobook \
            --process-only \
            -y > "$BOOK_LOG" 2>&1; then
            echo "  ✅ Phase 1 done"
        else
            echo "  ⚠️  Phase 1 exit=$? (may have partial results)"
        fi
        
        # Fix chapter_titles.json (ensure it has proper titles)
        if [ -d "$OUTPUT_DIR/$safe_name/scripts" ]; then
            uv run python3 -c "
import json
from pathlib import Path
from audify.readers.ebook import EpubReader

src = '$epub'
scripts_dir = Path('$OUTPUT_DIR/$safe_name/scripts')
try:
    reader = EpubReader(src)
    chapters = reader.get_chapters()
    titles = [reader.get_chapter_title(c) or f'Chapter {i+1}' for i, c in enumerate(chapters)]
    with open(scripts_dir / 'chapter_titles.json', 'w') as f:
        json.dump(titles, f, indent=2)
    print(f'  Fixed chapter_titles.json: {len(titles)} titles')
except Exception as e:
    print(f'  Could not fix titles: {e}')
" 2>&1
        fi
        
        # Phase 2: Synthesize-only (TTS + M4B)
        voice="af_bella"
        [ "$lang" = "es" ] && voice="ef_dora"
        [ "$lang" = "pt" ] && voice="pf_dora"
        
        echo "  Phase 2: TTS + M4B..."
        if timeout 10800 uv run audify "$epub" \
            --synthesize-only \
            --voice "$voice" \
            --tts-provider kokoro \
            -y >> "$BOOK_LOG" 2>&1; then
            echo "  ✅ Phase 2 done"
        else
            echo "  ⚠️  Phase 2 exit=$?"
        fi
        
        echo "  ✅ Done processing $base"
    done
done

echo ""
echo "=== Batch complete at $(date) ==="
