"""
Create random samples from validation set for testing
"""
import json
import random
import argparse
from pathlib import Path


def create_samples(input_file, output_dir, sizes=[1000, 5000, 10000], seed=42):
    """
    Create random samples of different sizes.

    Args:
        input_file: Input JSON file
        output_dir: Output directory
        sizes: List of sample sizes
        seed: Random seed for reproducibility
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total examples: {len(data)}")

    # Set seed
    random.seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create samples
    for size in sizes:
        if size > len(data):
            print(f"Warning: Requested size {size} exceeds data size {len(data)}, skipping")
            continue

        print(f"\nCreating sample of size {size}...")
        sample = random.sample(data, size)

        # Determine output filename
        input_name = Path(input_file).stem
        output_file = output_path / f"{input_name}_{size}.json"

        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Saved to: {output_file}")

    print(f"\n✅ Created {len(sizes)} samples")


def main():
    parser = argparse.ArgumentParser(description='Create random samples from dataset')
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/moraz_500k_val.json',
        help='Input JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory'
    )
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[1000, 5000, 10000],
        help='Sample sizes (e.g., --sizes 1000 5000 10000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()
    create_samples(args.input, args.output_dir, args.sizes, args.seed)


if __name__ == '__main__':
    main()
