#!/usr/bin/env python3
"""
Analyze vocabulary coverage for new dataset
Checks if moraz_500k.json contains characters not in char_vocab.json
"""

import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def analyze_coverage():
    """Check vocabulary coverage and identify issues"""

    # Load existing vocabulary
    vocab_path = Path("data/processed/char_vocab.json")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    existing_chars = set(vocab_data['char2idx'].keys())
    special_tokens = {'<PAD>', '<START>', '<END>', '<UNK>'}
    existing_chars_no_special = existing_chars - special_tokens

    print(f"Existing vocabulary:")
    print(f"  Total characters: {len(existing_chars)} (including 4 special tokens)")
    print(f"  Regular characters: {len(existing_chars_no_special)}")

    # Check for duplicate indices
    idx2char = vocab_data['idx2char']
    char_to_indices = {}
    for idx_str, char in idx2char.items():
        if char not in char_to_indices:
            char_to_indices[char] = []
        char_to_indices[char].append(idx_str)

    duplicates = {char: indices for char, indices in char_to_indices.items()
                  if len(indices) > 1}

    if duplicates:
        print(f"\n⚠ WARNING: Found duplicate indices:")
        for char, indices in duplicates.items():
            print(f"  '{char}' → indices {indices}")

    # Scan new dataset
    dataset_path = Path("data/processed/moraz_500k.json")
    if not dataset_path.exists():
        print(f"\n✗ Dataset not found: {dataset_path}")
        return

    print(f"\nAnalyzing dataset: {dataset_path.name}")
    print(f"  Size: {dataset_path.stat().st_size / 1024 / 1024:.1f} MB")

    all_chars = Counter()
    missing_chars = Counter()
    total_examples = 0

    print(f"  Loading JSON array...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"  Loaded {len(dataset):,} examples")

    for example in tqdm(dataset, desc="Scanning"):
        total_examples += 1

        # Check word and lemma
        for text in [example['word'], example['lemma']]:
            for char in text:
                all_chars[char] += 1
                if char not in existing_chars:
                    missing_chars[char] += 1

    print(f"\nDataset statistics:")
    print(f"  Total examples: {total_examples:,}")
    print(f"  Unique characters found: {len(all_chars)}")
    print(f"  Characters in vocabulary: {len(existing_chars_no_special)}")

    if missing_chars:
        print(f"\n❌ MISSING CHARACTERS ({len(missing_chars)}):")
        print(f"  These characters appear in the dataset but not in vocabulary:")
        for char, count in missing_chars.most_common(20):
            char_display = char if char.isprintable() else f"\\u{ord(char):04x}"
            print(f"    '{char_display}' → {count:,} occurrences")

        if len(missing_chars) > 20:
            print(f"    ... and {len(missing_chars) - 20} more")
    else:
        print(f"\n✅ COMPLETE COVERAGE: All characters are in vocabulary")

    # Character frequency analysis
    print(f"\n📊 Top 20 characters in new dataset:")
    for char, count in all_chars.most_common(20):
        char_display = char if char.isprintable() else f"\\u{ord(char):04x}"
        in_vocab = "✓" if char in existing_chars else "✗"
        print(f"  {in_vocab} '{char_display}' → {count:,}")

    # Recommendations
    print(f"\n" + "="*60)
    print(f"RECOMMENDATIONS:")
    print(f"="*60)

    if duplicates:
        print(f"1. ⚠ Fix duplicate indices in char_vocab.json")
        print(f"   Current issue: {list(duplicates.keys())}")

    if missing_chars:
        print(f"2. ❌ Rebuild vocabulary to include {len(missing_chars)} missing characters")
        print(f"   OR: Accept <UNK> token for {sum(missing_chars.values()):,} character occurrences")
    else:
        print(f"✅ Current vocabulary is sufficient for the new dataset")

    # Coverage percentage
    total_char_occurrences = sum(all_chars.values())
    missing_char_occurrences = sum(missing_chars.values())
    coverage_pct = 100 * (1 - missing_char_occurrences / total_char_occurrences) if total_char_occurrences > 0 else 100.0

    print(f"\n📈 Coverage Statistics:")
    print(f"  Total character occurrences: {total_char_occurrences:,}")
    print(f"  Covered character occurrences: {total_char_occurrences - missing_char_occurrences:,}")
    print(f"  Missing character occurrences: {missing_char_occurrences:,}")
    print(f"  Coverage percentage: {coverage_pct:.2f}%")

    return {
        'duplicates': duplicates,
        'missing_chars': dict(missing_chars),
        'coverage_pct': coverage_pct,
        'total_examples': total_examples
    }


if __name__ == "__main__":
    analyze_coverage()
