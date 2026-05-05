# Branch Guide: Verification Feature & Processing Improvements

## 🌳 Branch Structure

```
main (origin/main)
├── Latest stable version
│
├─→ qwen-tts-integration-clean ⭐ READY FOR PR
│   ├── Audiobook verification feature (complete)
│   ├── Per-chapter duration checks
│   ├── Full-audiobook verification
│   ├── 37/37 tests passing
│   ├── 28,000+ words of documentation
│   ├── Ready for immediate production merge
│   │
│   └─→ improve-processing-with-verification 🚀 (Current)
│       ├── Uses verification tool to analyze audiobooks
│       ├── Tools for batch analysis
│       ├── A/B testing framework
│       ├── Processing improvement workflow
│       └── Production improvement cycle
```

## 📋 What's on Each Branch

### Branch: `main`
**Status:** Production  
**Latest:** Latest stable code from GitHub

```bash
git checkout main
```

### Branch: `qwen-tts-integration-clean` ⭐ **RECOMMENDED FOR PR**
**Status:** Ready for GitHub PR  
**Purpose:** Implement audiobook verification feature

**What's Included:**
- ✅ `audify/verify.py` - Core verification logic (671 lines)
- ✅ `audify/verification_integration.py` - Pipeline integration (344 lines)
- ✅ `tests/test_verify.py` - 18 unit tests
- ✅ `tests/test_verification_integration.py` - 19 integration tests
- ✅ `audify/cli.py` - `audify compare` command
- ✅ Comprehensive documentation (28,000+ words)
- ✅ Real-world validation (Six Easy Pieces)
- ✅ 37/37 tests passing
- ✅ Zero breaking changes

**PR Title:**
```
Add audiobook verification feature with pipeline integration
```

**How to Use:**
```bash
git checkout qwen-tts-integration-clean

# Create PR on GitHub
# Base: main
# Compare: qwen-tts-integration-clean

# Use verification in processing
audify convert book.epub --llm-model qwen
audify compare source.epub audiobook.m4b
```

### Branch: `improve-processing-with-verification` 🚀 **CURRENT**
**Status:** Development  
**Purpose:** Use verification tool to analyze and improve audiobooks

**What's Included:**
- ✅ Everything from `qwen-tts-integration-clean`
- ✅ `PROCESSING_IMPROVEMENT_PLAN.md` - Detailed 5-phase plan (14K words)
- ✅ `PROCESSING_IMPROVEMENTS_README.md` - Quick start guide (10K words)
- ✅ `scripts/analyze_audiobooks.py` - Batch analysis tool
- ✅ `scripts/compare_versions.py` - A/B testing comparison tool

**How to Use:**
```bash
git checkout improve-processing-with-verification

# 1. Generate baseline
python scripts/analyze_audiobooks.py \
  --output-dir data/output \
  --report baseline_analysis.json

# 2. Analyze results
cat baseline_analysis.json | jq '.summary'

# 3. Deep dive on specific books
audify compare source.epub data/output/book_name/audiobook.m4b --json

# 4. Test improvements
audify convert test_book.epub --llm-model qwen --output-dir data/test_v1

# 5. Compare versions
python scripts/compare_versions.py baseline.json test.json

# 6. Deploy when satisfied
python scripts/analyze_audiobooks.py --output-dir data/output_v2 --report improved.json
```

---

## 🔄 Workflow: From Verification to Improvement

### Step 1: Get Verification Feature
```bash
# Option A: Use from qwen-tts-integration-clean
git checkout qwen-tts-integration-clean

# Option B: Merge into your branch
git merge qwen-tts-integration-clean
```

**What You Get:**
- Verification module: `audify/verify.py`
- Integration layer: `audify/verification_integration.py`
- CLI command: `audify compare`
- Tests: 37/37 passing
- Documentation: Complete

### Step 2: Analyze Existing Audiobooks
```bash
# Switch to improvement branch
git checkout improve-processing-with-verification

# Run analysis on all audiobooks
python scripts/analyze_audiobooks.py --output-dir data/output --report baseline.json

# Results tell you:
# - Which audiobooks are too short
# - Which have missing chapters
# - Overall quality metrics
```

### Step 3: Root Cause Analysis
```bash
# Use verification tool to deep-dive
audify compare source.epub data/output/book_name/audiobook.m4b --json

# Compare extracted vs generated
diff data/output/book_name/original_text_c01.txt \
     data/output/book_name/generated_script_c01.txt
```

### Step 4: Implement Improvements
```bash
# Test improvements
audify convert test_book.epub --llm-model qwen --output-dir data/test_v1

# Verify
audify compare test_book.epub data/test_v1/audiobook.m4b --json
```

### Step 5: Measure Impact
```bash
# Compare baseline vs improved
python scripts/compare_versions.py baseline.json improved.json

# Output shows:
# - % improvement per book
# - Average improvement across all books
# - Best and worst cases
```

### Step 6: Deploy at Scale
```bash
# Reprocess all books with improved config
for book in data/original_sources/*.epub; do
  audify convert "$book" --output-dir data/output_v2 --confirm
done

# Verify all
python scripts/analyze_audiobooks.py --output-dir data/output_v2 --report final.json

# Final comparison
python scripts/compare_versions.py baseline.json final.json
```

---

## 📊 Comparison: Two Branches

| Feature | `qwen-tts-integration-clean` | `improve-processing-with-verification` |
|---------|------------------------------|----------------------------------------|
| Verification Tool | ✅ | ✅ |
| Per-chapter checks | ✅ | ✅ |
| Full-audiobook checks | ✅ | ✅ |
| CLI: `audify compare` | ✅ | ✅ |
| Tests (37 passing) | ✅ | ✅ |
| Documentation | ✅ | ✅ |
| **Batch analysis** | ❌ | ✅ |
| **A/B testing** | ❌ | ✅ |
| **Improvement workflow** | ❌ | ✅ |
| **Analysis tools** | ❌ | ✅ |

---

## 🎯 Use Cases

### Use Case 1: Want Verification Feature Only
```bash
git checkout qwen-tts-integration-clean
# Create PR to main
# Use: audify compare, verification in pipeline
```

### Use Case 2: Want to Analyze Audiobooks
```bash
git checkout improve-processing-with-verification
python scripts/analyze_audiobooks.py
python scripts/compare_versions.py
```

### Use Case 3: Want to Improve Processing Algorithm
```bash
git checkout improve-processing-with-verification
# Follow 5-phase improvement plan in PROCESSING_IMPROVEMENT_PLAN.md
```

### Use Case 4: Want Both Features
```bash
git checkout improve-processing-with-verification
# Gets verification tool + analysis tools
```

---

## 📚 Documentation

### Quick Reference
- **PROCESSING_IMPROVEMENTS_README.md** - Start here for quick examples
- **PROCESSING_IMPROVEMENT_PLAN.md** - Detailed methodology

### Verification Feature
- **VERIFICATION_FEATURE.md** (on qwen-tts-integration-clean)
- **TEST_RESULTS.md** (on qwen-tts-integration-clean)
- **PIPELINE_INTEGRATION.md** (on qwen-tts-integration-clean)

### PR Documentation
- **PR_DESCRIPTION.md** (on qwen-tts-integration-clean)
- **COPY_PR_DESCRIPTION.txt** (on qwen-tts-integration-clean)

---

## 🚀 Recommended Workflow

### For Users Who Just Want Verification:
1. Merge `qwen-tts-integration-clean` to `main`
2. Use `audify compare` for manual verification
3. Verification integrated in generation pipeline

### For Users Who Want to Improve Processing:
1. Keep both branches
2. Start with `improve-processing-with-verification`
3. Run baseline analysis
4. Identify improvements
5. Test on samples
6. Deploy at scale
7. Track improvements with metrics

### For Maintainers:
1. Review `qwen-tts-integration-clean` PR
2. Merge to `main`
3. Use `improve-processing-with-verification` for future improvements
4. Keep iteration cycle running

---

## 🔄 Git Commands

### Switch between branches
```bash
git checkout qwen-tts-integration-clean
git checkout improve-processing-with-verification
git checkout main
```

### View branch history
```bash
git log --oneline --graph --all | head -20
```

### Create PR from qwen-tts-integration-clean
```bash
# Go to https://github.com/garciadias/audify/pulls
# Base: main
# Compare: qwen-tts-integration-clean
```

### Sync with main (if needed)
```bash
git fetch origin
git merge origin/main
```

---

## 📈 Current Status

| Branch | Status | Ready For |
|--------|--------|-----------|
| `main` | ✅ Production | Current production use |
| `qwen-tts-integration-clean` | ✅ Ready | GitHub PR (immediate) |
| `improve-processing-with-verification` | 🚀 Development | Processing improvements |

---

## ⚡ Quick Commands

### Generate Baseline
```bash
git checkout improve-processing-with-verification
python scripts/analyze_audiobooks.py --output-dir data/output --report baseline.json
```

### Verify Single Audiobook
```bash
git checkout qwen-tts-integration-clean
audify compare source.epub audiobook.m4b
```

### Compare Two Versions
```bash
git checkout improve-processing-with-verification
python scripts/compare_versions.py baseline.json improved.json
```

---

## 📞 Support

**Questions?** Check:
1. PROCESSING_IMPROVEMENTS_README.md - For quick examples
2. PROCESSING_IMPROVEMENT_PLAN.md - For detailed methodology
3. VERIFICATION_FEATURE.md - For verification details (on qwen-tts-integration-clean)

---

**Summary:**
- **qwen-tts-integration-clean**: Verification feature, ready for PR
- **improve-processing-with-verification**: Analysis tools + improvement workflow

**Both branches** complement each other to create a complete quality assurance and improvement pipeline!
