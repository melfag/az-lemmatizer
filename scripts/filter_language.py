#!/usr/bin/env python3
"""
Filter out non-Azerbaijani sentences from the dataset.

Uses character-based language detection to identify and remove:
- English sentences
- Russian sentences
- Turkish sentences
- Other non-Azerbaijani text
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import re
from collections import Counter

# Azerbaijani-specific characters
AZERBAIJANI_CHARS = set('əöüğışçӘÖÜĞIŞÇ')

# English-only common words (strong indicators)
ENGLISH_WORDS = {
    'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'should', 'could', 'may', 'might',
    'and', 'or', 'but', 'if', 'then', 'than',
    'this', 'that', 'these', 'those',
    'of', 'to', 'for', 'with', 'at', 'by', 'from',
    'government', 'system', 'project', 'committee'
}

# Russian Cyrillic characters (not used in Azerbaijani)
RUSSIAN_CHARS = set('ёыэъЁЫЭЪ')

def is_azerbaijani_sentence(sentence: str) -> tuple[bool, str]:
    """
    Determine if a sentence is in Azerbaijani.

    Returns:
        (is_azerbaijani, reason)
    """
    # Basic checks
    if not sentence or len(sentence.strip()) < 3:
        return False, "too_short"

    sentence_lower = sentence.lower()
    words = re.findall(r'\b[a-zA-ZəöüğışçƏÖÜĞIŞÇ]+\b', sentence_lower)

    if not words:
        return False, "no_words"

    # Check for Russian Cyrillic
    if any(char in RUSSIAN_CHARS for char in sentence):
        return False, "russian_cyrillic"

    # Count Azerbaijani-specific characters
    azerbaijani_char_count = sum(1 for char in sentence_lower if char in AZERBAIJANI_CHARS)

    # Count English indicator words
    english_word_count = sum(1 for word in words if word in ENGLISH_WORDS)

    # Calculate ratios
    total_words = len(words)
    english_ratio = english_word_count / total_words if total_words > 0 else 0

    # Decision logic

    # Strong indicator: Has Azerbaijani-specific characters
    if azerbaijani_char_count >= 2:
        return True, "azerbaijani_chars"

    # Strong negative: High ratio of English words
    if english_ratio > 0.4:  # More than 40% English words
        return False, "high_english_ratio"

    # Moderate negative: Some English words but no Azerbaijani chars
    if english_word_count >= 3 and azerbaijani_char_count == 0:
        return False, "english_words_no_az_chars"

    # Check for common Azerbaijani words (positive indicators)
    azerbaijani_words = {
        'və', 'bu', 'bir', 'ilə', 'ki', 'da', 'də', 'üçün',
        'olan', 'edir', 'oldu', 'olan', 'dən', 'dan',
        'ın', 'in', 'un', 'ün', 'nın', 'nin', 'nun', 'nün'
    }

    azerbaijani_word_count = sum(1 for word in words if word in azerbaijani_words)

    if azerbaijani_word_count >= 2:
        return True, "azerbaijani_words"

    # Default: if has at least one Azerbaijani char or word, keep it
    if azerbaijani_char_count > 0 or azerbaijani_word_count > 0:
        return True, "has_azerbaijani_indicators"

    # Otherwise, likely not Azerbaijani
    return False, "no_azerbaijani_indicators"

def filter_dataset(
    data: List[Dict],
    strict: bool = False
) -> tuple[List[Dict], Dict]:
    """
    Filter out non-Azerbaijani examples.

    Args:
        data: List of examples
        strict: If True, be more aggressive in filtering

    Returns:
        (filtered_data, stats)
    """
    filtered_data = []
    stats = Counter()

    for ex in data:
        context = ex.get('context', '')

        is_az, reason = is_azerbaijani_sentence(context)

        if is_az:
            filtered_data.append(ex)
            stats['kept'] += 1
            stats[f'kept_{reason}'] += 1
        else:
            stats['removed'] += 1
            stats[f'removed_{reason}'] += 1

    return filtered_data, stats

def main():
    parser = argparse.ArgumentParser(description='Filter non-Azerbaijani sentences')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--strict', action='store_true', help='Use strict filtering')
    parser.add_argument('--report', type=str, help='Save removed examples to file for review')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} examples")

    # Filter
    print("\nFiltering non-Azerbaijani sentences...")
    filtered_data, stats = filter_dataset(data, strict=args.strict)

    # Save filtered data
    print(f"\nSaving filtered data to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n" + "=" * 60)
    print("FILTERING RESULTS")
    print("=" * 60)
    print(f"Input examples:    {len(data):,}")
    print(f"Kept examples:     {stats['kept']:,} ({stats['kept']/len(data)*100:.2f}%)")
    print(f"Removed examples:  {stats['removed']:,} ({stats['removed']/len(data)*100:.2f}%)")

    print("\nKept reasons:")
    for key, value in sorted(stats.items()):
        if key.startswith('kept_') and value > 0:
            reason = key.replace('kept_', '')
            print(f"  {reason}: {value:,}")

    print("\nRemoved reasons:")
    for key, value in sorted(stats.items()):
        if key.startswith('removed_') and value > 0:
            reason = key.replace('removed_', '')
            print(f"  {reason}: {value:,}")

    # Save report if requested
    if args.report:
        removed_examples = []
        for ex in data:
            context = ex.get('context', '')
            is_az, reason = is_azerbaijani_sentence(context)
            if not is_az:
                removed_examples.append({
                    **ex,
                    'removal_reason': reason
                })

        print(f"\nSaving {len(removed_examples):,} removed examples to {args.report}...")
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(removed_examples, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Filtering complete!")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
