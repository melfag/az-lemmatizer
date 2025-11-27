#!/usr/bin/env python3
"""
Advanced pattern-based cleanup based on AI verification results.

SIMPLIFIED VERSION - Conservative approach to avoid over-stripping.

Fixes:
1. Only very clear possessive+case suffix combinations
2. Only obvious verb past tense markers
3. Over-stemming corrections (noun/verb confusion)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import argparse

# Common nouns that should NOT be stemmed to verbs
NOUN_NOT_VERB = {
    'çıxış': 'çıxış',      # exit/speech (not çıx)
    'mübarizə': 'mübarizə', # struggle (not mübariz)
    'davam': 'davam',       # continuation (not dava)
    'başlıq': 'başlıq',     # title/heading (not başla)
    'axış': 'axış',         # flow (not ax)
    'baxış': 'baxış',       # view (not bax)
    'yayış': 'yayış',       # spread (not yay)
    'giriş': 'giriş',       # entrance (not gir)
    'yazış': 'yazış',       # correspondence (not yaz)
    'müğənni': 'müğənni',   # singer (not müğən)
    'balıq': 'balıq',       # fish (not bal)
}

def get_vowel_harmony(word: str) -> str:
    """Determine if word follows back or front vowel harmony."""
    for char in reversed(word.lower()):
        if char in BACK_VOWELS:
            return 'back'
        elif char in FRONT_VOWELS:
            return 'front'
    return 'back'  # default

def strip_case_suffix(lemma: str) -> Tuple[str, bool]:
    """
    Remove ONLY very clear possessive+case suffix combinations.

    CONSERVATIVE: Only strip long, unambiguous suffixes that are
    clearly not part of the word root.

    Returns:
        (cleaned_lemma, was_changed)
    """
    original = lemma

    # Minimum length check - don't touch short words
    if len(lemma) < 8:
        return lemma, False

    # ONLY strip very long, unambiguous possessive+case combinations
    # These are 5-6 character suffixes that are clearly grammatical

    # Possessive + genitive: -sının, -sinin, -sunun, -sünün (5 chars)
    if re.search(r'(sının|sinin|sunun|sünün)$', lemma) and len(lemma) > 8:
        lemma = re.sub(r'(sının|sinin|sunun|sünün)$', '', lemma)

    # Possessive + locative: -sında, -sində (5 chars)
    elif re.search(r'(sında|sində)$', lemma) and len(lemma) > 8:
        lemma = re.sub(r'(sında|sində)$', '', lemma)

    # Possessive + dative: -sına, -sinə (4 chars)
    elif re.search(r'(sına|sinə)$', lemma) and len(lemma) > 7:
        lemma = re.sub(r'(sına|sinə)$', '', lemma)

    # That's it - no more aggressive stripping of simple suffixes
    # like -nı, -da, -lar which can be part of word roots

    return lemma, lemma != original

def strip_verb_suffix(lemma: str) -> Tuple[str, bool]:
    """
    Remove verb suffixes to get root.
    VERY CONSERVATIVE: Only strip long, unambiguous verb markers.

    Returns:
        (cleaned_lemma, was_changed)
    """
    original = lemma

    # Minimum length - only process longer words
    if len(lemma) < 8:
        return lemma, False

    # Only strip very long, clear verb markers (6+ chars)
    # Complex past forms: -mışdır, -mişdir (6 chars)
    if re.search(r'(mışdır|mişdir|muşdur|müşdür)$', lemma) and len(lemma) > 9:
        lemma = re.sub(r'(mışdır|mişdir|muşdur|müşdür)$', '', lemma)

    # That's it - no more aggressive verb suffix stripping
    # Suffixes like -mışdı, -ıldı are too risky

    return lemma, lemma != original

def fix_over_stemming(lemma: str, word: str) -> Tuple[str, bool]:
    """
    Fix cases where lemma was over-stemmed (noun → verb root).

    Returns:
        (fixed_lemma, was_changed)
    """
    # Check if the current lemma should actually be a noun
    for noun, correct_form in NOUN_NOT_VERB.items():
        # If word contains the noun form but lemma is the verb root
        if noun in word.lower() and lemma.lower() != correct_form.lower():
            # Check if lemma looks like an over-stemmed version
            if len(lemma) < len(correct_form) and correct_form.startswith(lemma):
                return correct_form, True

    return lemma, False

def apply_advanced_cleanup(data: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Apply advanced pattern-based cleanup.

    Returns:
        (cleaned_data, stats)
    """
    cleaned_data = []
    stats = Counter()

    for ex in data:
        word = ex['word']
        lemma = ex['lemma']
        original_lemma = lemma
        changed = False

        # Skip if lemma is empty or very short
        if not lemma or len(lemma) < 2:
            cleaned_data.append(ex)
            stats['skipped_short'] += 1
            continue

        # 1. Fix over-stemming first
        lemma, over_stemmed_fixed = fix_over_stemming(lemma, word)
        if over_stemmed_fixed:
            stats['fixed_over_stemming'] += 1
            changed = True

        # 2. Strip case suffixes from lemmas
        lemma, case_stripped = strip_case_suffix(lemma)
        if case_stripped:
            stats['stripped_case_suffix'] += 1
            changed = True

        # 3. Strip verb suffixes (only if lemma still looks like inflected verb)
        if len(lemma) > 4:  # Only for longer words
            lemma, verb_stripped = strip_verb_suffix(lemma)
            if verb_stripped:
                stats['stripped_verb_suffix'] += 1
                changed = True

        # 4. Ensure lemma isn't longer than word (sanity check)
        if len(lemma) > len(word):
            lemma = original_lemma  # Revert if we made it worse
            changed = False
            stats['reverted_longer'] += 1

        # 5. Ensure lemma isn't too short (sanity check)
        if len(lemma) < 2 and len(original_lemma) >= 2:
            lemma = original_lemma  # Revert if we made it too short
            changed = False
            stats['reverted_too_short'] += 1

        # Update example
        if changed:
            cleaned_ex = {**ex, 'lemma': lemma}
            stats['total_changed'] += 1
        else:
            cleaned_ex = ex
            stats['unchanged'] += 1

        cleaned_data.append(cleaned_ex)

    return cleaned_data, dict(stats)

def main():
    parser = argparse.ArgumentParser(description='Advanced pattern-based cleanup')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')

    args = parser.parse_args()

    print("=" * 70)
    print("ADVANCED PATTERN-BASED CLEANUP")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} examples")

    # Apply cleanup
    print("\nApplying advanced pattern fixes...")
    cleaned_data, stats = apply_advanced_cleanup(data)

    # Save output
    print(f"\nSaving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n" + "=" * 70)
    print("CLEANUP RESULTS")
    print("=" * 70)
    print(f"Total examples:         {len(data):,}")
    print(f"Changed:                {stats.get('total_changed', 0):,}")
    print(f"Unchanged:              {stats.get('unchanged', 0):,}")

    print("\nBreakdown of fixes:")
    print(f"  Case suffixes stripped:   {stats.get('stripped_case_suffix', 0):,}")
    print(f"  Verb suffixes stripped:   {stats.get('stripped_verb_suffix', 0):,}")
    print(f"  Over-stemming fixed:      {stats.get('fixed_over_stemming', 0):,}")
    print(f"  Reverted (sanity check):  {stats.get('reverted_longer', 0) + stats.get('reverted_too_short', 0):,}")

    change_rate = (stats.get('total_changed', 0) / len(data)) * 100
    print(f"\nChange rate: {change_rate:.2f}%")

    print(f"\n✅ Cleaned dataset saved to: {args.output}")

if __name__ == "__main__":
    main()
