#!/usr/bin/env python3
"""
Analyze the moraz_900k_cleaned.json dataset to identify problematic patterns
based on manual review feedback.

Issues identified from manual review:
1. Passive verbs over-stemmed (edilmişdir → edil instead of et)
2. Verbal nouns over-stemmed (böyüməsinə → böyümə instead of böyü)
3. Possessive+case stripping from words where it's part of root
4. Derivational suffixes being stripped incorrectly
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def analyze_passive_verbs(data):
    """
    Find passive verb forms that were likely over-stemmed.
    Pattern: lemmas ending in -ıl, -il, -ul, -ül (passive markers)
    """
    issues = []
    for ex in data:
        lemma = ex['lemma']
        word = ex['word']

        # Check if lemma ends with passive marker
        if re.search(r'(ıl|il|ul|ül)$', lemma) and len(lemma) > 4:
            # Check if word has passive + past tense markers
            if re.search(r'(ılmış|ilmiş|ulmuş|ülmüş|ıldı|ildi|uldu|üldü)', word):
                issues.append({
                    'word': word,
                    'lemma': lemma,
                    'issue': 'passive_verb_overstemmed',
                    'pattern': f'{lemma} ends with passive marker'
                })

    return issues


def analyze_verbal_nouns(data):
    """
    Find verbal nouns that might be over-stemmed.
    Pattern: lemmas ending in -mə, -ma (verbal noun suffix)
    """
    issues = []
    for ex in data:
        lemma = ex['lemma']
        word = ex['word']

        # Check if lemma ends with verbal noun marker
        if re.search(r'(mə|ma)$', lemma) and len(lemma) > 4:
            # Check if word has possessive/case suffix after verbal noun
            if re.search(r'(məsinin|masının|məsinə|masına|məsində|masında)', word):
                issues.append({
                    'word': word,
                    'lemma': lemma,
                    'issue': 'verbal_noun_overstemmed',
                    'pattern': f'{lemma} ends with verbal noun suffix'
                })

    return issues


def analyze_consonant_cluster_loss(data):
    """
    Find words where final consonant was incorrectly removed.
    Pattern: word ends in -Csının/-sinin but lemma missing final consonant
    """
    issues = []
    for ex in data:
        lemma = ex['lemma']
        word = ex['word']

        # Common patterns where consonant cluster gets broken
        if re.search(r'(sının|sinin|sunun|sünün)$', word):
            # Extract what should be the base
            base = re.sub(r'(sının|sinin|sunun|sünün)$', '', word)

            # Check if lemma is shorter than expected
            if len(lemma) < len(base) and base.startswith(lemma):
                missing = base[len(lemma):]
                if re.match(r'^[bcçdfgğhjklmnpqrsştvxyz]+$', missing):
                    issues.append({
                        'word': word,
                        'lemma': lemma,
                        'expected_lemma': base,
                        'issue': 'consonant_cluster_broken',
                        'pattern': f'Missing: {missing}'
                    })

    return issues


def analyze_derivational_suffixes(data):
    """
    Find words with derivational suffixes that might be incorrectly stripped.
    Patterns: -laş/-ləş, -lan/-lən, -laşdır/-ləşdir
    """
    issues = []
    for ex in data:
        lemma = ex['lemma']
        word = ex['word']

        # Check for derivational patterns
        if re.search(r'(laşdır|ləşdir|landır|ləndir)mış', word):
            # Lemma should probably end in root, not in derivational suffix
            if re.search(r'(laşdır|ləşdir|landır|ləndir)$', lemma):
                issues.append({
                    'word': word,
                    'lemma': lemma,
                    'issue': 'derivational_suffix_in_lemma',
                    'pattern': 'Contains -laşdır/-ləşdir etc.'
                })

    return issues


def analyze_misspellings(data):
    """
    Find obvious misspellings in the dataset.
    Patterns: consonant clusters without vowels, repeated consonants
    """
    issues = []
    for ex in data:
        word = ex['word']
        lemma = ex['lemma']

        # Check for consonant clusters without vowels
        if re.search(r'[bcçdfgğhjklmnpqrsştvxyz]{4,}', word.lower()):
            issues.append({
                'word': word,
                'lemma': lemma,
                'issue': 'likely_misspelling',
                'pattern': 'Long consonant cluster in word'
            })

        # Check for very short lemmas (likely over-stripped)
        if len(lemma) < 2:
            issues.append({
                'word': word,
                'lemma': lemma,
                'issue': 'lemma_too_short',
                'pattern': f'Lemma is only {len(lemma)} char(s)'
            })

    return issues


def main():
    print("=" * 70)
    print("ANALYZING MORAZ_900K_CLEANED.JSON FOR ISSUES")
    print("=" * 70)

    # Load the cleaned dataset
    print("\nLoading data...")
    with open('data/processed/moraz_900k_cleaned.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} examples")

    # Run analyses
    print("\n" + "=" * 70)
    print("ANALYZING PATTERNS")
    print("=" * 70)

    analyses = {
        'passive_verbs': analyze_passive_verbs(data),
        'verbal_nouns': analyze_verbal_nouns(data),
        'consonant_clusters': analyze_consonant_cluster_loss(data),
        'derivational_suffixes': analyze_derivational_suffixes(data),
        'misspellings': analyze_misspellings(data),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ISSUES")
    print("=" * 70)

    total_issues = 0
    for issue_type, issues in analyses.items():
        count = len(issues)
        total_issues += count
        pct = (count / len(data)) * 100
        print(f"{issue_type:30s}: {count:6,} ({pct:5.2f}%)")

    print(f"\n{'TOTAL UNIQUE ISSUES':30s}: {total_issues:6,}")

    # Show samples for each issue type
    print("\n" + "=" * 70)
    print("SAMPLE ISSUES (10 per type)")
    print("=" * 70)

    for issue_type, issues in analyses.items():
        if not issues:
            continue

        print(f"\n{issue_type.upper()}:")
        print("-" * 70)
        for i, issue in enumerate(issues[:10]):
            print(f"{i+1}. {issue['word']} → {issue['lemma']}")
            print(f"   Issue: {issue['issue']}")
            print(f"   {issue['pattern']}")
            if 'expected_lemma' in issue:
                print(f"   Expected: {issue['expected_lemma']}")
            print()

    # Save detailed report
    output_file = 'data/processed/cleanup_issues_analysis.json'
    print(f"\nSaving detailed report to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_examples': len(data),
            'total_issues': total_issues,
            'issue_counts': {k: len(v) for k, v in analyses.items()},
            'issues_by_type': analyses
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Analysis complete!")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    issue_rate = (total_issues / len(data)) * 100

    if issue_rate < 2:
        print(f"✅ Issue rate is low ({issue_rate:.2f}%)")
        print("   → RECOMMEND: Proceed with training")
        print("   → The model may learn to correct these patterns")
    elif issue_rate < 5:
        print(f"⚠️  Issue rate is moderate ({issue_rate:.2f}%)")
        print("   → CONSIDER: Quick fixes for most common patterns")
        print("   → OR proceed with training and evaluate")
    else:
        print(f"❌ Issue rate is high ({issue_rate:.2f}%)")
        print("   → RECOMMEND: Fix issues before training")
        print("   → High error rate may confuse the model")

if __name__ == "__main__":
    main()
