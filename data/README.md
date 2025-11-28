# Data Directory Structure

## Active Data

### `processed/` - Training-ready datasets
**Current 900K dataset (used for RunPod training):**
- `moraz_900k_cleaned.json` (220 MB) - Cleaned source dataset
- `moraz_900k_train.json` (176 MB) - Training split (720K examples)
- `moraz_900k_val.json` (22 MB) - Validation split (90K examples)
- `moraz_900k_test.json` (22 MB) - Test split (90K examples)

**Universal Dependencies test sets:**
- `ud_test.json` (99 KB) - UD Azerbaijani test set (726 examples)
- `train.json` (60 KB) - Original UD train split
- `val.json` (13 KB) - Original UD val split
- `test.json` (13 KB) - Original UD test split

**Vocabulary:**
- `char_vocab.json` (4.3 KB) - Character vocabulary for model

### `raw/` - Source text data
- `dollma_1M_fresh.txt` (121 MB) - 1M sentences from DOLLMA corpus

### `specialized_tests/` - Linguistic test sets
- Custom test sets for specific phenomena

### `processed/analysis/` - Analysis results
- `cleanup_issues_analysis.json` - Dataset quality analysis
- `ud_test_stats.json` - UD test set statistics
- `moraz_900k_changes_deduplicated.json` - Cleanup review summary
- `metadata.json` - Dataset metadata

## Archived Data (~3GB)

### `processed/archive/` - Historical datasets
Organized by iteration:

**500k_iteration/** - First large-scale attempt (8 files)
- moraz_50k.json, moraz_500k*.json

**850k_iteration/** - Intermediate expansion (3 files)
- moraz_850k_train/val/test.json

**1M_iterations/** - Various 1M processing attempts (5 files)
- moraz_1M_annotated.json - Raw annotations
- moraz_1M_cleaned.json - Early cleanup
- moraz_900k.json - Filtered source

**cleanup_attempts/** - Rejected cleanup experiments (4 files)
- moraz_900k_advanced_cleaned.json - Aggressive cleanup (rejected)
- moraz_900k_pattern_cleaned.json - Pattern cleanup
- Change tracking files

**verification/** - Manual and AI verification (2 directories)
- lemma_verification/ - Batch review files
- ai_verified/ - AI verification results

### `raw/archive/` - Old raw data downloads (4 files, 73 MB)
- Early DOLLMA download batches (100K, 200K, 300K)
- az_text_100k.txt

## Quick Stats

| Category | Size | Files |
|----------|------|-------|
| Active processed data | 440 MB | 9 files |
| Active raw data | 121 MB | 1 file |
| Archived data | 3.1 GB | 30+ files |
| **Total** | **~3.7 GB** | **40+ files** |

## Notes

- All active files are used by the current training pipeline
- Archive is kept for reference and reproducibility
- Specialized tests remain separate for evaluation purposes

---

*Last Updated: November 22, 2025*
*After cleanup and organization*
