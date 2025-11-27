#!/usr/bin/env python3
"""
Apply manual corrections from TSV review files back to the full dataset.

Usage:
    python scripts/apply_corrections.py \
        --input data/processed/moraz_1M_annotated.json \
        --corrections review_samples/*.tsv \
        --output data/processed/moraz_1M_corrected.json
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Set
import argparse
from collections import defaultdict

def load_corrections(csv_files: List[Path]) -> Dict[tuple, str]:
    """
    Load corrections from CSV files.

    Returns:
        Dict mapping (word, context, old_lemma) -> corrected_lemma or 'DELETE'
    """
    corrections = {}
    deletions = set()
    stats = defaultdict(int)

    for csv_file in csv_files:
        print(f"\nReading {csv_file.name}...")

        with open(csv_file, 'r', encoding='utf-8') as f:
            # Skip comment lines
            lines = [line for line in f if not line.startswith('#')]

            reader = csv.DictReader(lines)

            for row in reader:
                word = row['word'].strip().strip('"')
                old_lemma = row['lemma'].strip().strip('"')
                context = row.get('context', '').strip().strip('"')
                correct = row.get('correct', '').strip()
                corrected_lemma = row.get('corrected_lemma', '').strip()

                # Skip if not marked as incorrect
                if correct != '0':
                    continue

                # Skip if no correction provided
                if not corrected_lemma:
                    stats['skipped_no_correction'] += 1
                    continue

                # Handle DELETE marker
                key = (word, context, old_lemma)
                if corrected_lemma.upper() == 'DELETE':
                    deletions.add(key)
                    stats['deletions_loaded'] += 1
                else:
                    corrections[key] = corrected_lemma
                    stats['corrections_loaded'] += 1

        print(f"  Loaded {stats['corrections_loaded']} corrections and {stats['deletions_loaded']} deletions")

    return corrections, deletions, stats

def apply_corrections_to_dataset(
    data: List[Dict],
    corrections: Dict[tuple, str],
    deletions: Set[tuple]
) -> tuple[List[Dict], Dict]:
    """
    Apply corrections and deletions to dataset.

    Returns:
        (corrected_data, stats)
    """
    corrected_data = []
    stats = defaultdict(int)

    for ex in data:
        word = ex['word']
        lemma = ex['lemma']
        context = ex.get('context', '')

        # Check if this example should be deleted
        key = (word, context, lemma)

        if key in deletions:
            # Skip this example (delete it)
            stats['deleted'] += 1
            continue

        if key in corrections:
            # Apply correction
            new_lemma = corrections[key]
            corrected_ex = {**ex, 'lemma': new_lemma}
            corrected_data.append(corrected_ex)
            stats['corrected'] += 1
        else:
            # Keep original
            corrected_data.append(ex)
            stats['unchanged'] += 1

    return corrected_data, stats

def main():
    parser = argparse.ArgumentParser(description='Apply manual corrections to dataset')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--corrections', type=str, nargs='+', required=True,
                        help='TSV correction files (can use wildcards)')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')

    args = parser.parse_args()

    # Load corrections
    print("=" * 60)
    print("LOADING CORRECTIONS")
    print("=" * 60)

    csv_files = []
    for pattern in args.corrections:
        csv_files.extend(Path('.').glob(pattern))

    if not csv_files:
        print("❌ No correction files found!")
        return

    corrections, deletions, load_stats = load_corrections(csv_files)

    print("\n" + "=" * 60)
    print("CORRECTION SUMMARY")
    print("=" * 60)
    print(f"Total corrections: {len(corrections):,}")
    print(f"Total deletions: {len(deletions):,}")
    print(f"Skipped (no correction provided): {load_stats['skipped_no_correction']:,}")

    # Load dataset
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    print(f"Reading {args.input}...")

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} examples")

    # Apply corrections
    print("\n" + "=" * 60)
    print("APPLYING CORRECTIONS AND DELETIONS")
    print("=" * 60)

    corrected_data, apply_stats = apply_corrections_to_dataset(data, corrections, deletions)

    # Save corrected dataset
    print(f"\nSaving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_data, f, ensure_ascii=False, indent=2)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Input examples: {len(data):,}")
    print(f"Corrections applied: {apply_stats['corrected']:,}")
    print(f"Deleted: {apply_stats['deleted']:,}")
    print(f"Unchanged: {apply_stats['unchanged']:,}")
    print(f"Output examples: {len(corrected_data):,}")
    print(f"\n✅ Corrected dataset saved to: {output_path}")

    # Show correction and deletion rates
    total_changes = len(corrections) + len(deletions)
    if total_changes > 0:
        applied_changes = apply_stats['corrected'] + apply_stats['deleted']
        applied_rate = (applied_changes / total_changes) * 100
        print(f"\n📊 Changes application rate: {applied_rate:.1f}% ({applied_changes:,}/{total_changes:,})")
        print(f"   Corrections: {apply_stats['corrected']:,}/{len(corrections):,}")
        print(f"   Deletions: {apply_stats['deleted']:,}/{len(deletions):,}")

        if applied_rate < 100:
            missing = total_changes - applied_changes
            print(f"⚠️  {missing:,} changes not found in dataset")
            print("   This may be due to context mismatches or examples already filtered out")

if __name__ == "__main__":
    main()
