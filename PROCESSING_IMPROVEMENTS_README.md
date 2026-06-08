# Processing Improvements with Verification Tool

This branch uses the audiobook verification feature to analyze existing audiobooks, identify issues, and iteratively improve the content extraction and processing algorithm.

## 🎯 Objective

Transform the verification tool into a **feedback loop** for continuous improvement:

```
Analyze Audiobooks → Identify Issues → Root Cause Analysis → 
Implement Improvements → Test & Validate → Deploy → Repeat
```

## 📊 Quick Start

### Phase 1: Baseline Analysis (Today)

```bash
# Run analysis on all processed audiobooks
python scripts/analyze_audiobooks.py \
  --output-dir data/output \
  --report baseline_analysis.json

# View summary
cat baseline_analysis.json | jq '.' | less
```

This generates:
- Overall statistics
- List of problem audiobooks
- Duration ratio analysis
- Chapter accuracy metrics

### Phase 2: Deep Analysis (Next)

```bash
# Examine a specific problematic audiobook
audify compare source.epub data/output/audiobook.m4b --json

# Compare extracted vs generated content
diff data/output/book_name/original_text_c01.txt \
     data/output/book_name/generated_script_c01.txt | head -50
```

### Phase 3: Identify Improvements

Based on analysis, target specific areas:

- **Content Extraction:** Better chapter boundaries, OCR quality
- **Script Generation:** Adjust summarization ratio
- **Duration Management:** Preserve more content
- **Chapter Handling:** Better spine structure detection

### Phase 4: Test Improvements

```bash
# Create test configuration
cp configs/default.toml configs/test_improvement_v1.toml
# Edit configs/test_improvement_v1.toml

# Test on a sample book
audify convert test_book.epub \
  --config configs/test_improvement_v1.toml \
  --output-dir data/test_v1

# Compare against baseline
audify compare test_book.epub data/test_v1/audiobook.m4b --json > test_v1_results.json
audify compare test_book.epub data/baseline/audiobook.m4b --json > baseline_results.json

# Analyze improvements
python scripts/compare_versions.py baseline_results.json test_v1_results.json
```

### Phase 5: Batch Reprocess

Once improvements are validated:

```bash
# Reprocess all books with improved algorithm
for book in data/original_sources/*.epub; do
  audify convert "$book" \
    --config configs/improved_final.toml \
    --output-dir data/output_v2 \
    --confirm
done

# Verify improvements
python scripts/analyze_audiobooks.py \
  --output-dir data/output_v2 \
  --report improved_analysis.json

# Compare baseline vs improvements
python scripts/compare_versions.py \
  baseline_analysis.json \
  improved_analysis.json \
  --output improvement_summary.json
```

## 📁 Files Structure

```
audify/
├── PROCESSING_IMPROVEMENT_PLAN.md       ← Detailed plan
├── scripts/
│   ├── analyze_audiobooks.py            ← Batch analysis tool
│   ├── compare_versions.py              ← Version comparison tool
│   ├── check_tts_health.py              (existing)
│   └── qwen_tts_api.py                  (existing)
└── configs/
    └── (improvement configurations)
```

## 🔍 Analysis Tools

### 1. Batch Analysis: `analyze_audiobooks.py`

**Purpose:** Audit all processed audiobooks and generate metrics

**Usage:**
```bash
python scripts/analyze_audiobooks.py \
  --output-dir data/output \
  --report baseline_analysis.json \
  --verbose
```

**Output:**
- Console summary with statistics
- JSON report with detailed metrics per book
- Problem identification (duration ratio < 60%, missing chapters, etc.)

**Metrics Collected:**
- Duration ratio (actual vs expected)
- Chapter count (matched, expected, missing, extra)
- Issue count and types
- Word count from source
- Processing status

### 2. Version Comparison: `compare_versions.py`

**Purpose:** Compare results between baseline and improved versions

**Usage:**
```bash
python scripts/compare_versions.py \
  baseline_analysis.json \
  improved_analysis.json \
  --output comparison_report.json
```

**Output:**
- Console report with side-by-side comparison
- Improvement percentages
- Best improvements and regressions
- JSON detailed results

**Metrics:**
- Duration ratio improvement %
- Chapter accuracy changes
- Issue reduction
- Improvement rate (% of books improved)

### 3. Standalone Verification: `audify compare`

**Purpose:** Compare any source/audiobook pair

**Usage:**
```bash
# Human-readable output
audify compare source.epub audiobook.m4b

# JSON output for analysis
audify compare source.epub audiobook.m4b -- --json

# Show differences
audify compare source.epub audiobook.m4b --show-diff
```

## 📈 Key Metrics

### Duration Ratio

- **Definition:** Audiobook duration / Expected duration (at 75 wpm)
- **Baseline:** ~0.44 (44% of expected - due to LLM summarization)
- **Target:** 0.60-0.70 (60-70% of expected)
- **Improvement Opportunity:** +35-60% longer audiobooks

### Chapter Accuracy

- **Definition:** Chapters extracted from audiobook match source
- **Baseline:** 100% (6/6 chapters for Six Easy Pieces)
- **Target:** 100% (maintain accuracy)
- **Risk:** None identified

### Content Preservation

- **Definition:** Generated script length vs original text
- **Baseline:** 60% (typical LLM summarization)
- **Target:** 75%+ (more detailed scripts)
- **Improvement:** Better prompt engineering, less aggressive summarization

## 🚀 Improvement Hypothesis

### Problem: Short Audiobooks

**Symptoms:**
- Duration ratio 0.44 (44% of expected)
- Users report audiobooks are too brief
- Generated scripts are brief summaries

**Root Cause Analysis:**
1. LLM is summarizing aggressively
2. Default prompt too concise
3. Settings favor brevity over completeness

**Potential Solutions:**
- Adjust LLM temperature (more diverse output)
- Change summarization prompt (ask for more detail)
- Add content preservation constraint
- Increase generated script length target

### Problem: Content Loss

**Symptoms:**
- 40% of content lost (60% preserved)
- Important details missing
- Technical examples removed

**Root Cause Analysis:**
1. Aggressive summarization
2. Insufficient context window
3. Token limits forcing cuts

**Potential Solutions:**
- Increase prompt context length
- Break large chapters into sections
- Preserve examples and technical details
- Add importance weighting to content

### Problem: Chapter Extraction

**Symptoms:**
- Occasional missing or extra chapters
- Out-of-order chapters in some ebooks

**Root Cause Analysis:**
1. Complex EPUB spine structures
2. Nested chapter hierarchies
3. Front/back matter confusion

**Potential Solutions:**
- Better TOC parsing
- Dynamic spine traversal
- Hierarchical chapter flattening
- Manual chapter mapping support

## 🎯 Success Criteria

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Duration Ratio | 0.44 | 0.60+ | 🎯 |
| Content Preservation | 60% | 75%+ | 🎯 |
| Chapter Accuracy | 100% | 100% | ✅ |
| Processing Time | ~2min | <3min | 🎯 |
| User Satisfaction | Low | High | 🎯 |

## 📊 Example Workflow

### Day 1: Establish Baseline

```bash
# Generate baseline metrics
python scripts/analyze_audiobooks.py \
  --output-dir data/output \
  --report baseline.json

# Review results
cat baseline.json | jq '.summary'
```

**Output:**
```json
{
  "avg_duration_ratio": 0.44,
  "total_books": 12,
  "books_with_issues": 8,
  "avg_chapter_accuracy": 0.95,
  "problem_categories": {
    "short_audiobooks": 10,
    "missing_chapters": 2,
    "extraction_errors": 1
  }
}
```

### Day 2: Root Cause Analysis

```bash
# Examine top problem books
cat baseline.json | jq '.[] | select(.duration_ratio < 0.50)' | head -20

# Deep dive on specific book
audify compare source.epub data/output/book1/audiobook.m4b --json
```

### Day 3-4: Implement Improvement

```bash
# Create improved configuration
vim configs/preservation_v1.toml

# Test on sample
audify convert test_book.epub \
  --config configs/preservation_v1.toml \
  --output-dir data/test_v1

# Compare
python scripts/compare_versions.py \
  baseline.json test_v1.json
```

**Output:**
```
OVERALL IMPROVEMENTS
─────────────────────────────────────────
Average Duration Ratio Improvement:  +25.0%
Books Improved:                       10/12
Best Improvement:                    +45.5% - book_1.epub
Worst Regression:                     -5.0% - book_2.epub
```

### Day 5: Deploy Improvements

```bash
# Reprocess all books
for book in data/original_sources/*.epub; do
  audify convert "$book" \
    --config configs/preservation_v1.toml \
    --output-dir data/output_v2 \
    --confirm
done

# Verify improvements
python scripts/analyze_audiobooks.py \
  --output-dir data/output_v2 \
  --report improved.json

# Final comparison
python scripts/compare_versions.py baseline.json improved.json
```

## 🔧 Tools Available

### Verification Feature (From Previous Work)

- ✅ `audify/verify.py` - Core verification logic
- ✅ `audify/verification_integration.py` - Pipeline integration
- ✅ `audify compare` CLI command
- ✅ Per-chapter duration checks
- ✅ Full-audiobook verification
- ✅ Interactive and automation modes

### Analysis Tools (New)

- ✅ `scripts/analyze_audiobooks.py` - Batch analysis
- ✅ `scripts/compare_versions.py` - Version comparison

### To Create

- [ ] `audify/analysis/content_comparison.py` - Content preservation analysis
- [ ] `scripts/find_improvement_opportunities.py` - Automated opportunity detection
- [ ] `scripts/generate_improvement_report.py` - Detailed recommendations
- [ ] Configuration profiles for different strategies

## 📚 Documentation

All details in:
- **`PROCESSING_IMPROVEMENT_PLAN.md`** - Full plan (14K words)
  - Phase breakdown
  - Root cause analysis templates
  - Implementation details
  - Metrics definitions
  - Success criteria

## 🚀 Next Steps

1. ✅ Create `improve-processing-with-verification` branch
2. ✅ Add analysis tools
3. ⏭️ Generate baseline analysis
4. ⏭️ Perform root cause analysis
5. ⏭️ Implement improvements
6. ⏭️ Test and validate
7. ⏭️ Deploy and reprocess

## 💡 Key Insights

- **Verification tool is now a feedback mechanism** for continuous improvement
- **Data-driven approach** to understanding audiobook quality
- **Iterative improvement** with A/B testing capability
- **Batch operations** enable rapid testing across multiple books
- **Graceful degradation** ensures we never break existing audiobooks

---

**Status:** 🚀 Ready to start analysis  
**Branch:** `improve-processing-with-verification`  
**Verification Feature:** ✅ Available (from `qwen-tts-integration-clean`)

Let's improve audiobooks! 📚🔊
