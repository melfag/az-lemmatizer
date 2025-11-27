#!/usr/bin/env python3
"""
Download DOLLMA corpus sample for annotation
"""

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import argparse


def download_dollma(output_path: Path, num_sentences: int = 100000):
    """
    Download DOLLMA corpus sample from multiple configs

    Args:
        output_path: Path to save the text file
        num_sentences: Number of sentences to download
    """
    print(f"Downloading DOLLMA corpus...")
    print(f"Target: {num_sentences:,} sentences")
    print(f"Output: {output_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # DOLLMA configs (mix for diversity)
    configs = [
        'azwiki',           # Wikipedia
        'anl-news',         # News
        'elite-blogs',      # Blogs
        'elite-books',      # Books
    ]

    # Download from each config
    total_count = 0
    per_config = num_sentences // len(configs)

    with open(output_path, 'w', encoding='utf-8') as f:
        for config_name in configs:
            print(f"\n{'='*60}")
            print(f"Downloading from: {config_name}")
            print(f"Target: {per_config:,} sentences")

            try:
                # Load dataset
                dataset = load_dataset(
                    "allmalab/DOLLMA",
                    name=config_name,
                    split="train",
                    streaming=True
                )

                count = 0
                pbar = tqdm(total=per_config, desc=f"{config_name}")

                for example in dataset:
                    # Extract text
                    text = None

                    if isinstance(example, dict):
                        text = example.get('text') or example.get('sentence') or example.get('content')
                    elif isinstance(example, str):
                        text = example

                    if not text:
                        continue

                    # Clean and write
                    text = text.strip()
                    if len(text) < 10:  # Skip very short texts
                        continue

                    f.write(text + '\n')
                    count += 1
                    total_count += 1
                    pbar.update(1)

                    if count >= per_config:
                        break

                pbar.close()
                print(f"  ✓ Downloaded {count:,} sentences from {config_name}")

            except Exception as e:
                print(f"  ✗ Error with {config_name}: {e}")
                continue

    print(f"\n{'='*60}")
    print(f"✓ Total downloaded: {total_count:,} sentences")
    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return total_count


def main():
    parser = argparse.ArgumentParser(description="Download DOLLMA corpus sample")
    parser.add_argument('--output', type=str, default="data/raw/dollma_sample_100k.txt",
                        help="Output text file")
    parser.add_argument('--num-sentences', type=int, default=100000,
                        help="Number of sentences to download")

    args = parser.parse_args()

    download_dollma(
        output_path=Path(args.output),
        num_sentences=args.num_sentences
    )


if __name__ == "__main__":
    main()
