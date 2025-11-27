#!/usr/bin/env python3
"""
Filter annotated data and split into train/val/test
Combines Steps 3 and 4 for thesis dataset preparation
"""

import json
import random
from pathlib import Path
from collections import Counter
import argparse


def filter_examples(examples):
    """
    Filter out low-quality examples

    Criteria:
    - Remove empty lemmas
    - Remove very short words (< 3 chars)
    - Remove if lemma longer than word (invalid)
    - Remove large edit distances (> 10)
    - Fix capitalization (lowercase lemmas)
    """
    filtered = []
    stats = {
        'total': len(examples),
        'empty_lemma': 0,
        'too_short': 0,
        'invalid_form': 0,
        'large_edit_distance': 0,
        'capitalization_fixed': 0,
        'kept': 0
    }

    for ex in examples:
        word = ex.get('word', '').strip()
        lemma = ex.get('lemma', '').strip()

        # Skip empty lemmas
        if not lemma or not word:
            stats['empty_lemma'] += 1
            continue

        # Skip very short words
        if len(word) < 3:
            stats['too_short'] += 1
            continue

        # Skip if lemma longer than word (invalid)
        if len(lemma) > len(word):
            stats['invalid_form'] += 1
            continue

        # Skip large edit distances
        edit_dist = levenshtein_distance(word, lemma)
        if edit_dist > 10:
            stats['large_edit_distance'] += 1
            continue

        # Fix capitalization (lemmas should be lowercase)
        if lemma[0].isupper() and word[0].islower():
            lemma = lemma.lower()
            ex['lemma'] = lemma
            stats['capitalization_fixed'] += 1

        filtered.append(ex)
        stats['kept'] += 1

    return filtered, stats


def levenshtein_distance(s1, s2):
    """Calculate edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def deduplicate_examples(examples):
    """Remove exact duplicate (word, context, lemma) triples"""
    seen = set()
    unique = []

    for ex in examples:
        # Create tuple key
        key = (ex['word'], ex['context'], ex['lemma'])
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    print(f"  Removed {len(examples) - len(unique):,} duplicates")
    return unique


def split_dataset(examples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test

    Args:
        examples: List of examples
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
        test_ratio: Proportion for test (default 0.1)
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Shuffle with fixed seed
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # Calculate split points
    total = len(shuffled)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # Split
    train_data = shuffled[:train_size]
    val_data = shuffled[train_size:train_size + val_size]
    test_data = shuffled[train_size + val_size:]

    return train_data, val_data, test_data


def analyze_dataset(examples, name="Dataset"):
    """Print statistics about dataset"""
    print(f"\n{'='*60}")
    print(f"{name} Statistics")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples):,}")

    # Word length distribution
    word_lengths = [len(ex['word']) for ex in examples]
    print(f"\nWord length:")
    print(f"  Min: {min(word_lengths)}")
    print(f"  Max: {max(word_lengths)}")
    print(f"  Avg: {sum(word_lengths) / len(word_lengths):.1f}")

    # Lemma length distribution
    lemma_lengths = [len(ex['lemma']) for ex in examples]
    print(f"\nLemma length:")
    print(f"  Min: {min(lemma_lengths)}")
    print(f"  Max: {max(lemma_lengths)}")
    print(f"  Avg: {sum(lemma_lengths) / len(lemma_lengths):.1f}")

    # Edit distance distribution
    edit_dists = [levenshtein_distance(ex['word'], ex['lemma']) for ex in examples[:10000]]
    print(f"\nEdit distance (sample of 10K):")
    print(f"  Min: {min(edit_dists)}")
    print(f"  Max: {max(edit_dists)}")
    print(f"  Avg: {sum(edit_dists) / len(edit_dists):.1f}")

    # Identity ratio
    identity_count = sum(1 for ex in examples if ex['word'] == ex['lemma'])
    print(f"\nIdentity predictions: {identity_count:,} ({identity_count/len(examples)*100:.1f}%)")

    # Most common lemmas
    lemma_counts = Counter(ex['lemma'] for ex in examples)
    print(f"\nMost common lemmas:")
    for lemma, count in lemma_counts.most_common(10):
        print(f"  {lemma}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Filter and split annotated dataset")
    parser.add_argument('--input', type=str, required=True,
                        help="Input annotated JSON file")
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help="Output directory")
    parser.add_argument('--prefix', type=str, default='moraz_850k',
                        help="Prefix for output files")
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help="Training set ratio (default: 0.8)")
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help="Test set ratio (default: 0.1)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Dataset Filtering and Splitting")
    print("="*60)
    print(f"\nInput: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratio: {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")

    # Load data
    print(f"\n📂 Loading data from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} examples")

    # Step 1: Filter
    print(f"\n🔍 Filtering examples...")
    filtered_data, filter_stats = filter_examples(data)

    print(f"\nFiltering Statistics:")
    print(f"  Total input: {filter_stats['total']:,}")
    print(f"  Empty lemmas: {filter_stats['empty_lemma']:,}")
    print(f"  Too short: {filter_stats['too_short']:,}")
    print(f"  Invalid form: {filter_stats['invalid_form']:,}")
    print(f"  Large edit distance: {filter_stats['large_edit_distance']:,}")
    print(f"  Capitalization fixed: {filter_stats['capitalization_fixed']:,}")
    print(f"  ✓ Kept: {filter_stats['kept']:,} ({filter_stats['kept']/filter_stats['total']*100:.1f}%)")

    # Step 2: Deduplicate
    print(f"\n🔄 Removing duplicates...")
    unique_data = deduplicate_examples(filtered_data)
    print(f"  ✓ Unique examples: {len(unique_data):,}")

    # Step 3: Split
    print(f"\n✂️  Splitting into train/val/test...")
    train_data, val_data, test_data = split_dataset(
        unique_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print(f"  Train: {len(train_data):,} examples ({len(train_data)/len(unique_data)*100:.1f}%)")
    print(f"  Val:   {len(val_data):,} examples ({len(val_data)/len(unique_data)*100:.1f}%)")
    print(f"  Test:  {len(test_data):,} examples ({len(test_data)/len(unique_data)*100:.1f}%)")

    # Step 4: Save
    print(f"\n💾 Saving datasets...")

    train_path = output_dir / f'{args.prefix}_train.json'
    val_path = output_dir / f'{args.prefix}_val.json'
    test_path = output_dir / f'{args.prefix}_test.json'

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved train: {train_path}")

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved val: {val_path}")

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved test: {test_path}")

    # Step 5: Analyze
    analyze_dataset(train_data, "Training Set")
    analyze_dataset(val_data, "Validation Set")
    analyze_dataset(test_data, "Test Set")

    # Summary
    print(f"\n{'='*60}")
    print("✅ COMPLETE!")
    print(f"{'='*60}")
    print(f"\nFinal Dataset:")
    print(f"  Total: {len(unique_data):,} examples")
    print(f"  Train: {len(train_data):,} examples")
    print(f"  Val:   {len(val_data):,} examples")
    print(f"  Test:  {len(test_data):,} examples")
    print(f"\nSaved to: {output_dir}")
    print(f"  {train_path.name}")
    print(f"  {val_path.name}")
    print(f"  {test_path.name}")
    print(f"\n🎯 Target achieved: {len(unique_data):,} examples")
    print(f"   (Thesis target: 850,000 examples)")
    print("="*60)


if __name__ == "__main__":
    main()
