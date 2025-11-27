#!/usr/bin/env python3
"""
Apply systematic pattern-based corrections to the dataset.

Fixes:
1. Specific lemma replacements (onun → o, edildi → et, etc.)
2. Over-stemmed verb forms (etmişdi* → et, *etdi* → et)
3. Remove English words
"""

import json
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter
import argparse

# English words to remove
ENGLISH_WORDS = {
    # Common articles, prepositions, conjunctions
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'about', 'as', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'over',
    'and', 'or', 'but', 'if', 'then', 'than', 'so', 'because',

    # Common verbs
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'done',
    'will', 'would', 'should', 'could', 'may', 'might', 'must',
    'can', 'shall',

    # Common nouns
    'time', 'year', 'people', 'way', 'day', 'man', 'thing', 'woman',
    'life', 'child', 'world', 'school', 'state', 'family', 'student',
    'group', 'country', 'problem', 'hand', 'part', 'place', 'case',
    'week', 'company', 'system', 'program', 'question', 'work',
    'government', 'number', 'night', 'point', 'home', 'water', 'room',
    'mother', 'area', 'money', 'story', 'fact', 'month', 'lot',
    'right', 'study', 'book', 'eye', 'job', 'word', 'business',
    'issue', 'side', 'kind', 'head', 'house', 'service', 'friend',
    'father', 'power', 'hour', 'game', 'line', 'end', 'member',
    'law', 'car', 'city', 'community', 'name', 'president', 'team',
    'minute', 'idea', 'kid', 'body', 'information', 'back', 'parent',
    'face', 'others', 'level', 'office', 'door', 'health', 'person',
    'art', 'war', 'history', 'party', 'result', 'change', 'morning',
    'reason', 'research', 'girl', 'guy', 'moment', 'air', 'teacher',
    'force', 'education', 'foundation', 'project', 'committee',

    # Common adjectives
    'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own',
    'other', 'old', 'right', 'big', 'high', 'different', 'small',
    'large', 'next', 'early', 'young', 'important', 'few', 'public',
    'bad', 'same', 'able',

    # Other common words
    'all', 'one', 'two', 'three', 'other', 'some', 'many', 'most',
    'several', 'such', 'no', 'yes', 'not', 'only', 'just', 'very',
    'even', 'also', 'back', 'there', 'where', 'when', 'what', 'which',
    'who', 'how', 'why', 'this', 'that', 'these', 'those',
}

# Specific lemma replacements
LEMMA_REPLACEMENTS = {
    'onun': 'o',       # Pronoun normalization
    'edildi': 'et',    # Past passive → base verb
}

def is_english_word(lemma: str) -> bool:
    """Check if a lemma is an English word."""
    lemma_lower = lemma.lower().strip()

    # Direct match
    if lemma_lower in ENGLISH_WORDS:
        return True

    # Check for common English patterns
    # Words ending in -tion, -sion, -ment, -ness, etc.
    english_suffixes = [
        'tion', 'sion', 'ment', 'ness', 'ship', 'hood',
        'ful', 'less', 'ish', 'ous', 'ive', 'ing', 'ed'
    ]

    for suffix in english_suffixes:
        if len(lemma_lower) > 5 and lemma_lower.endswith(suffix):
            # Additional check: no Azerbaijani-specific characters
            if not any(c in lemma_lower for c in 'əöüğışçƏÖÜĞIŞÇ'):
                return True

    return False

def apply_pattern_corrections(data: List[Dict]) -> tuple[List[Dict], Dict]:
    """
    Apply systematic corrections to the dataset.

    Returns:
        (corrected_data, stats)
    """
    corrected_data = []
    stats = Counter()

    for ex in data:
        word = ex['word']
        lemma = ex['lemma']
        original_lemma = lemma

        # Skip if lemma is empty
        if not lemma:
            stats['empty_lemma'] += 1
            continue

        # Check for English words
        if is_english_word(lemma):
            stats['removed_english'] += 1
            stats[f'removed_english_{lemma.lower()}'] += 1
            continue

        # Apply specific replacements
        if lemma in LEMMA_REPLACEMENTS:
            lemma = LEMMA_REPLACEMENTS[lemma]
            stats[f'replaced_{original_lemma}_to_{lemma}'] += 1
            stats['total_replacements'] += 1

        # Fix over-stemmed verbs: etmişdi* → et
        elif lemma.startswith('etmişdi'):
            lemma = 'et'
            stats['fixed_etmişdi'] += 1
            stats['total_replacements'] += 1

        # Fix over-stemmed verbs: *etdi* → et
        elif 'etdi' in lemma:
            lemma = 'et'
            stats['fixed_etdi'] += 1
            stats['total_replacements'] += 1

        # Update example with corrected lemma
        corrected_ex = {**ex, 'lemma': lemma}
        corrected_data.append(corrected_ex)

        if lemma != original_lemma:
            stats['total_changed'] += 1
        else:
            stats['unchanged'] += 1

    return corrected_data, stats

def main():
    parser = argparse.ArgumentParser(description='Apply pattern-based corrections')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--backup', action='store_true', help='Create backup of input file')

    args = parser.parse_args()

    # Load data
    print("=" * 60)
    print("PATTERN-BASED CLEANUP")
    print("=" * 60)
    print(f"\nLoading data from {args.input}...")

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} examples")

    # Create backup if requested
    if args.backup:
        backup_path = Path(args.input).with_suffix('.backup.json')
        print(f"\nCreating backup at {backup_path}...")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Backup created")

    # Apply corrections
    print("\n" + "=" * 60)
    print("APPLYING PATTERN CORRECTIONS")
    print("=" * 60)

    corrected_data, stats = apply_pattern_corrections(data)

    # Save corrected data
    print(f"\nSaving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_data, f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n" + "=" * 60)
    print("CORRECTION STATISTICS")
    print("=" * 60)
    print(f"Input examples:     {len(data):,}")
    print(f"Output examples:    {len(corrected_data):,}")
    print(f"Removed:            {len(data) - len(corrected_data):,}")
    print(f"Changed:            {stats['total_changed']:,}")
    print(f"Unchanged:          {stats['unchanged']:,}")

    print("\n" + "-" * 60)
    print("SPECIFIC CORRECTIONS")
    print("-" * 60)

    # Lemma replacements
    print("\nLemma replacements:")
    for old, new in LEMMA_REPLACEMENTS.items():
        count = stats.get(f'replaced_{old}_to_{new}', 0)
        if count > 0:
            print(f"  {old} → {new}: {count:,} examples")

    # Verb corrections
    print("\nVerb form corrections:")
    if stats['fixed_etmişdi'] > 0:
        print(f"  etmişdi* → et: {stats['fixed_etmişdi']:,} examples")
    if stats['fixed_etdi'] > 0:
        print(f"  *etdi* → et: {stats['fixed_etdi']:,} examples")

    # English words removed
    print(f"\nEnglish words removed: {stats['removed_english']:,} examples")

    # Show top removed English words
    english_removals = [(k.replace('removed_english_', ''), v)
                       for k, v in stats.items()
                       if k.startswith('removed_english_') and k != 'removed_english']

    if english_removals:
        english_removals.sort(key=lambda x: x[1], reverse=True)
        print("\nTop removed English words:")
        for word, count in english_removals[:20]:
            print(f"  '{word}': {count:,}")

    print("\n" + "=" * 60)
    print("✅ CLEANUP COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {args.output}")
    print(f"\nSummary:")
    print(f"  Total replacements: {stats['total_replacements']:,}")
    print(f"  English removed:    {stats['removed_english']:,}")
    print(f"  Empty lemmas:       {stats['empty_lemma']:,}")
    print(f"  Final dataset:      {len(corrected_data):,} examples")

if __name__ == "__main__":
    main()
