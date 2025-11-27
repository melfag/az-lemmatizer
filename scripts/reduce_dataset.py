#!/usr/bin/env python3
"""
Intelligently reduce dataset size while maintaining quality and diversity.

Strategies:
1. Remove exact duplicates (word, lemma, context)
2. Remove near-duplicates (same word-lemma, similar context)
3. Prioritize shorter edit distances (more reliable lemmas)
4. Maintain lemma diversity (avoid over-representation)
5. Keep balanced word length distribution

Usage:
    python scripts/reduce_dataset.py \
        --input data/processed/moraz_1M_cleaned.json \
        --output data/processed/moraz_900k.json \
        --target-size 900000
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import hashlib

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate edit distance between two strings."""
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

def compute_example_hash(ex: Dict, include_context: bool = True) -> str:
    """Compute hash for duplicate detection."""
    if include_context:
        key = f"{ex['word']}|{ex['lemma']}|{ex.get('context', '')}"
    else:
        key = f"{ex['word']}|{ex['lemma']}"
    return hashlib.md5(key.encode()).hexdigest()

def compute_quality_score(ex: Dict) -> float:
    """
    Compute quality score for an example.

    Higher score = higher quality
    """
    score = 0.0
    word = ex['word']
    lemma = ex['lemma']
    context = ex.get('context', '')

    # 1. Prefer shorter edit distance (more conservative lemmatization)
    edit_dist = levenshtein_distance(word, lemma)
    if edit_dist == 0:
        score += 5.0  # Identity (very reliable)
    elif edit_dist <= 2:
        score += 3.0  # Small change (likely correct)
    elif edit_dist <= 4:
        score += 1.0  # Medium change (okay)
    else:
        score -= 1.0  # Large change (potentially wrong)

    # 2. Prefer examples with context
    if context and len(context) > 10:
        score += 2.0

    # 3. Penalize very short lemmas (often over-stemmed)
    if len(lemma) < 2:
        score -= 3.0
    elif len(lemma) == 2:
        score -= 1.0

    # 4. Penalize very long lemmas (might be under-stemmed)
    if len(lemma) > 15:
        score -= 1.0

    # 5. Prefer reasonable word-lemma ratio
    if len(word) > 0:
        ratio = len(lemma) / len(word)
        if 0.5 <= ratio <= 1.0:
            score += 1.0  # Good ratio
        elif ratio > 1.0:
            score -= 2.0  # Lemma longer than word (wrong!)

    # 6. Check if lemma is substring of word (usually good sign)
    if lemma in word:
        score += 1.0

    return score

def remove_exact_duplicates(data: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Remove exact duplicates (word, lemma, context)."""
    seen = set()
    unique_data = []
    stats = {'removed': 0, 'kept': 0}

    for ex in data:
        ex_hash = compute_example_hash(ex, include_context=True)
        if ex_hash not in seen:
            seen.add(ex_hash)
            unique_data.append(ex)
            stats['kept'] += 1
        else:
            stats['removed'] += 1

    return unique_data, stats

def remove_near_duplicates(data: List[Dict], max_per_word_lemma: int = 5) -> Tuple[List[Dict], Dict]:
    """
    Remove near-duplicates: keep at most N examples per (word, lemma) pair.
    Keeps highest quality examples.
    """
    # Group by (word, lemma)
    groups = defaultdict(list)
    for ex in data:
        key = (ex['word'], ex['lemma'])
        groups[key].append(ex)

    filtered_data = []
    stats = {'removed': 0, 'kept': 0, 'groups_reduced': 0}

    for key, examples in groups.items():
        if len(examples) <= max_per_word_lemma:
            # Keep all
            filtered_data.extend(examples)
            stats['kept'] += len(examples)
        else:
            # Keep top N by quality score
            examples_with_scores = [(ex, compute_quality_score(ex)) for ex in examples]
            examples_with_scores.sort(key=lambda x: x[1], reverse=True)

            top_examples = [ex for ex, score in examples_with_scores[:max_per_word_lemma]]
            filtered_data.extend(top_examples)

            stats['kept'] += len(top_examples)
            stats['removed'] += len(examples) - len(top_examples)
            stats['groups_reduced'] += 1

    return filtered_data, stats

def filter_by_quality(data: List[Dict], target_size: int) -> Tuple[List[Dict], Dict]:
    """
    Filter to target size by removing lowest quality examples.
    """
    if len(data) <= target_size:
        return data, {'removed': 0, 'kept': len(data)}

    print(f"  Computing quality scores for {len(data):,} examples...")

    # Compute quality scores
    examples_with_scores = [(ex, compute_quality_score(ex)) for ex in data]

    # Sort by score (highest first)
    examples_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Keep top target_size
    filtered_data = [ex for ex, score in examples_with_scores[:target_size]]

    stats = {
        'removed': len(data) - target_size,
        'kept': target_size,
        'min_score': examples_with_scores[target_size - 1][1],
        'max_score': examples_with_scores[0][1],
        'avg_score': sum(score for ex, score in examples_with_scores[:target_size]) / target_size
    }

    return filtered_data, stats

def main():
    parser = argparse.ArgumentParser(description='Intelligently reduce dataset size')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--target-size', type=int, required=True, help='Target dataset size')
    parser.add_argument('--max-per-pair', type=int, default=5,
                        help='Max examples per (word, lemma) pair')

    args = parser.parse_args()

    print("=" * 70)
    print("INTELLIGENT DATASET REDUCTION")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} examples")
    print(f"Target size: {args.target_size:,} examples")
    print(f"Need to remove: {len(data) - args.target_size:,} examples ({(len(data) - args.target_size) / len(data) * 100:.1f}%)")

    # Step 1: Remove exact duplicates
    print("\n" + "=" * 70)
    print("STEP 1: Remove exact duplicates")
    print("=" * 70)

    data, dup_stats = remove_exact_duplicates(data)
    print(f"Removed: {dup_stats['removed']:,} exact duplicates")
    print(f"Remaining: {len(data):,} examples")

    # Step 2: Remove near-duplicates
    print("\n" + "=" * 70)
    print("STEP 2: Remove near-duplicates")
    print("=" * 70)
    print(f"Max examples per (word, lemma) pair: {args.max_per_pair}")

    data, near_dup_stats = remove_near_duplicates(data, args.max_per_pair)
    print(f"Reduced {near_dup_stats['groups_reduced']:,} groups")
    print(f"Removed: {near_dup_stats['removed']:,} near-duplicates")
    print(f"Remaining: {len(data):,} examples")

    # Step 3: Quality-based filtering (if still above target)
    if len(data) > args.target_size:
        print("\n" + "=" * 70)
        print("STEP 3: Quality-based filtering")
        print("=" * 70)

        data, quality_stats = filter_by_quality(data, args.target_size)
        print(f"Removed: {quality_stats['removed']:,} low-quality examples")
        print(f"Remaining: {len(data):,} examples")
        print(f"Score range: {quality_stats['min_score']:.2f} to {quality_stats['max_score']:.2f}")
        print(f"Average score: {quality_stats['avg_score']:.2f}")

    # Save output
    print("\n" + "=" * 70)
    print("SAVING OUTPUT")
    print("=" * 70)
    print(f"Saving to {args.output}...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Final statistics
    print("\n" + "=" * 70)
    print("REDUCTION COMPLETE")
    print("=" * 70)

    original_size = len(data) + dup_stats['removed'] + near_dup_stats['removed']
    if len(data) > args.target_size:
        original_size += (len(data) - args.target_size)

    print(f"Original size:  {original_size:,} examples")
    print(f"Final size:     {len(data):,} examples")
    print(f"Reduction:      {original_size - len(data):,} examples ({(original_size - len(data)) / original_size * 100:.1f}%)")
    print(f"\n✅ Reduced dataset saved to: {args.output}")

    # Print some statistics about the final dataset
    lemma_counts = Counter(ex['lemma'] for ex in data)
    word_lengths = [len(ex['word']) for ex in data]
    lemma_lengths = [len(ex['lemma']) for ex in data]

    print("\n" + "=" * 70)
    print("FINAL DATASET STATISTICS")
    print("=" * 70)
    print(f"Unique lemmas:      {len(lemma_counts):,}")
    print(f"Avg word length:    {sum(word_lengths) / len(word_lengths):.1f}")
    print(f"Avg lemma length:   {sum(lemma_lengths) / len(lemma_lengths):.1f}")
    print(f"Top 10 lemmas:")
    for lemma, count in lemma_counts.most_common(10):
        print(f"  {lemma}: {count:,}")

if __name__ == "__main__":
    main()
