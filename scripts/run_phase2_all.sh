#!/bin/bash
# Run Phase 2 (synthesize) for all books with complete scripts
# Skips books that already have M4B

BOOKS_DIR="/home/rd24/Downloads/books_deepseek"
OUTPUT_DIR="data/output"

for lang in en es pt; do
    for epub in "$BOOKS_DIR/$lang"/*.epub "$BOOKS_DIR/$lang"/*.pdf; do
        [ -f "$epub" ] || continue
        base=$(basename "$epub")
        stem="${base%.*}"
        safe=$(echo "$stem" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | sed 's/[^a-z0-9_]//g' | sed 's/^_//')
        
        scripts_dir="$OUTPUT_DIR/$safe/scripts"
        [ -d "$scripts_dir" ] || continue
        
        # Count episode scripts
        count=$(ls "$scripts_dir"/episode_*_script.txt 2>/dev/null | wc -l)
        [ "$count" -lt 1 ] && continue
        
        # Skip if M4B already exists
        m4b=$(ls "$OUTPUT_DIR/$safe"/*.m4b 2>/dev/null | head -1)
        [ -n "$m4b" ] && echo "⏭️  $safe (has M4B)" && continue
        
        voice="af_bella"
        [ "$lang" = "es" ] && voice="ef_dora"
        [ "$lang" = "pt" ] && voice="pf_dora"
        
        echo "🔄 Synthesizing $safe ($count scripts)..."
        VOICE="$voice" timeout 7200 uv run python3 scripts/synthesize_with_ffmpeg.py "$epub" > /dev/null 2>&1
        
        # Verify
        m4b=$(ls "$OUTPUT_DIR/$safe"/*.m4b 2>/dev/null | head -1)
        if [ -n "$m4b" ]; then
            uv run python3 -c "
from audify.verify import AudiobookVerifier
v = AudiobookVerifier('$epub', '$m4b')
r = v.verify()
print(f'  ✅ {r.matched}/{r.total_source} ({r.overall_match_percentage}%)')
" 2>/dev/null
        fi
    done
done
echo "=== Phase 2 complete ==="
