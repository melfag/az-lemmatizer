"""
Prepare UD treebank data for training.
This script:
1. Loads converted UD JSON data
2. Splits into train/val/test
3. Builds character vocabulary
4. Saves everything in the expected format
"""

import json
import random
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.vocabulary import CharacterVocabulary


def split_data(examples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data into train/val/test sets.

    Args:
        examples: List of examples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        (train, val, test) splits
    """
    random.seed(seed)

    # Shuffle
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # Calculate split points
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description='Prepare UD data for training')

    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/ud_test.json',
        help='Input JSON file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Training set ratio'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PREPARING UD DATA FOR TRAINING")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} examples")

    # Remove POS tags if present (not needed for training)
    for ex in examples:
        if 'pos' in ex:
            del ex['pos']

    # Split data
    print(f"\nSplitting data...")
    train, val, test = split_data(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print(f"  Train: {len(train)} examples ({args.train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val)} examples ({args.val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test)} examples ({args.test_ratio*100:.0f}%)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    print(f"\nSaving split data to {output_dir}")

    with open(output_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    print(f"  Saved train.json ({len(train)} examples)")

    with open(output_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    print(f"  Saved val.json ({len(val)} examples)")

    with open(output_dir / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=2)
    print(f"  Saved test.json ({len(test)} examples)")

    # Build character vocabulary
    print(f"\nBuilding character vocabulary...")
    char_vocab = CharacterVocabulary()

    # Add characters from all examples
    all_texts = []
    for ex in examples:
        all_texts.extend([ex['word'], ex['lemma'], ex['context']])

    char_vocab.add_from_data(all_texts)

    print(f"Vocabulary size: {char_vocab.vocab_size}")

    # Save vocabulary
    vocab_path = output_dir / 'char_vocab.json'
    char_vocab.save(str(vocab_path))
    print(f"Saved vocabulary to {vocab_path}")

    # Save metadata
    metadata = {
        'source': 'UD_Azerbaijani-TueCL',
        'num_examples': len(examples),
        'train_examples': len(train),
        'val_examples': len(val),
        'test_examples': len(test),
        'char_vocab_size': char_vocab.vocab_size,
        'splits': {
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio
        },
        'seed': args.seed
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {metadata_path}")

    # Print sample examples
    print("\n" + "=" * 80)
    print("SAMPLE EXAMPLES")
    print("=" * 80)

    for i, ex in enumerate(train[:5]):
        print(f"\n{i+1}. Word: {ex['word']}")
        print(f"   Context: {ex['context']}")
        print(f"   Lemma: {ex['lemma']}")

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nStatistics:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Training: {len(train)} ({args.train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val)} ({args.val_ratio*100:.0f}%)")
    print(f"  Test: {len(test)} ({args.test_ratio*100:.0f}%)")
    print(f"  Character vocabulary size: {char_vocab.vocab_size}")

    print(f"\nOutput files:")
    print(f"  {output_dir}/train.json")
    print(f"  {output_dir}/val.json")
    print(f"  {output_dir}/test.json")
    print(f"  {output_dir}/char_vocab.json")
    print(f"  {output_dir}/metadata.json")

    print("\n✓ Data preparation successful!")
    print(f"\nNext step: python3 scripts/train.py")


if __name__ == "__main__":
    main()
