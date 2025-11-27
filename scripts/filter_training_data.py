"""
Filter training data for better quality
"""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.metrics import levenshtein_distance


def filter_examples(examples):
    """
    Remove low-quality examples

    Filters:
    1. Empty lemmas
    2. Very short words (< 3)
    3. Lemma longer than word
    4. Excessive edit distance (> 10)
    5. Fix capitalization issues
    """
    filtered = []
    stats = {
        'total': len(examples),
        'empty_lemma': 0,
        'too_short': 0,
        'lemma_longer': 0,
        'large_edit_dist': 0,
        'invalid_chars': 0,
        'case_fixed': 0,
        'kept': 0
    }

    for ex in examples:
        word = ex['word']
        lemma = ex['lemma']

        # Skip empty lemmas
        if not lemma or len(lemma) == 0:
            stats['empty_lemma'] += 1
            continue

        # Skip very short words
        if len(word) < 3:
            stats['too_short'] += 1
            continue

        # Skip if lemma is longer than word (unlikely)
        if len(lemma) > len(word):
            stats['lemma_longer'] += 1
            continue

        # Skip if edit distance too large
        if levenshtein_distance(word, lemma) > 10:
            stats['large_edit_dist'] += 1
            continue

        # Fix capitalization (lemmas should be lowercase unless proper nouns)
        if lemma[0].isupper() and word[0].isupper():
            # Both capitalized - probably sentence start, lowercase lemma
            ex_fixed = ex.copy()
            ex_fixed['lemma'] = lemma.lower()
            filtered.append(ex_fixed)
            stats['case_fixed'] += 1
            stats['kept'] += 1
            continue

        # Keep example
        filtered.append(ex)
        stats['kept'] += 1

    # Print statistics
    print("\nFiltering Statistics:")
    print(f"  Total examples: {stats['total']}")
    print(f"  Removed:")
    print(f"    Empty lemma: {stats['empty_lemma']}")
    print(f"    Too short: {stats['too_short']}")
    print(f"    Lemma longer: {stats['lemma_longer']}")
    print(f"    Large edit distance: {stats['large_edit_dist']}")
    print(f"    Invalid chars: {stats['invalid_chars']}")
    print(f"  Fixed:")
    print(f"    Capitalization: {stats['case_fixed']}")
    print(f"  Kept: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")

    return filtered


if __name__ == '__main__':
    # Load training data
    print("Loading training data...")
    with open('data/processed/moraz_500k_train.json') as f:
        train_data = json.load(f)

    # Filter
    print(f"\nFiltering {len(train_data)} examples...")
    filtered_data = filter_examples(train_data)

    # Save
    output_file = 'data/processed/moraz_500k_train_filtered.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved filtered data to: {output_file}")
    print(f"Removed: {len(train_data) - len(filtered_data)} examples")
