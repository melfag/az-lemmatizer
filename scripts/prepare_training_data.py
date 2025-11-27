#!/usr/bin/env python3
"""
Prepare training data by splitting 500K dataset
Implements Hybrid approach: 90% train, 10% val, + UD test
"""

import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm


def split_dataset(input_path: Path, output_dir: Path, train_ratio: float = 0.9,
                  val_ratio: float = 0.1, seed: int = 42):
    """
    Split dataset into train and validation sets

    Args:
        input_path: Path to moraz_500k.json
        output_dir: Output directory for splits
        train_ratio: Fraction for training (default 0.9)
        val_ratio: Fraction for validation (default 0.1)
        seed: Random seed for reproducibility
    """

    print("="*60)
    print("Preparing Training Data - Hybrid Split")
    print("="*60)

    # Validate ratios
    if abs(train_ratio + val_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0: {train_ratio} + {val_ratio} = {train_ratio + val_ratio}")

    # Set random seed
    random.seed(seed)

    # Load dataset
    print(f"\n1. Loading dataset: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_examples = len(dataset)
    print(f"   Total examples: {total_examples:,}")

    # Shuffle dataset
    print(f"\n2. Shuffling dataset (seed={seed})...")
    random.shuffle(dataset)

    # Calculate split sizes
    train_size = int(total_examples * train_ratio)
    val_size = total_examples - train_size

    print(f"\n3. Split sizes:")
    print(f"   Train: {train_size:,} ({train_ratio*100:.1f}%)")
    print(f"   Val:   {val_size:,} ({val_ratio*100:.1f}%)")

    # Split data
    train_data = dataset[:train_size]
    val_data = dataset[train_size:]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training set
    train_path = output_dir / "moraz_500k_train.json"
    print(f"\n4. Saving training set...")
    print(f"   Path: {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    train_size_mb = train_path.stat().st_size / 1024 / 1024
    print(f"   Size: {train_size_mb:.1f} MB")

    # Save validation set
    val_path = output_dir / "moraz_500k_val.json"
    print(f"\n5. Saving validation set...")
    print(f"   Path: {val_path}")
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    val_size_mb = val_path.stat().st_size / 1024 / 1024
    print(f"   Size: {val_size_mb:.1f} MB")

    # Sample verification
    print(f"\n6. Sample verification:")
    print(f"   First train example: {train_data[0]['word']} → {train_data[0]['lemma']}")
    print(f"   First val example: {val_data[0]['word']} → {val_data[0]['lemma']}")

    # Statistics
    print("\n" + "="*60)
    print("✓ SPLIT COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  Training:   {train_path} ({train_size:,} examples, {train_size_mb:.1f} MB)")
    print(f"  Validation: {val_path} ({val_size:,} examples, {val_size_mb:.1f} MB)")
    print(f"\nNote: UD test set remains at data/processed/ud_test.json (110 examples)")
    print("\nReady for training!")
    print("="*60)

    return {
        'train_path': str(train_path),
        'val_path': str(val_path),
        'train_size': train_size,
        'val_size': val_size,
        'seed': seed
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare training data splits")
    parser.add_argument('--input', type=str,
                        default="data/processed/moraz_500k.json",
                        help="Input dataset path")
    parser.add_argument('--output-dir', type=str,
                        default="data/processed",
                        help="Output directory")
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help="Training set ratio (default: 0.9)")
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"✗ Error: Input file not found: {input_path}")
        return

    split_dataset(
        input_path=input_path,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
