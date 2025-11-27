#!/usr/bin/env python3
"""
Download DOLLMA corpus sample for annotation - Version 2
Uses alternative loading method to avoid pickle errors
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import requests


def download_dollma_direct(output_path: Path, num_sentences: int = 100000):
    """
    Download DOLLMA corpus using direct HuggingFace API

    Args:
        output_path: Path to save the text file
        num_sentences: Number of sentences to download
    """
    print(f"Downloading DOLLMA corpus (direct method)...")
    print(f"Target: {num_sentences:,} sentences")
    print(f"Output: {output_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        # Try loading different configs without streaming
        configs = ['azwiki', 'anl-news', 'elite-blogs', 'elite-books']

        total_count = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for config in configs:
                if total_count >= num_sentences:
                    break

                print(f"\nTrying config: {config}")

                try:
                    # Try non-streaming first
                    ds = load_dataset(
                        "allmalab/DOLLMA",
                        config,
                        split="train[:10000]",  # Load first 10k as sample
                        trust_remote_code=True
                    )

                    print(f"  Loaded {len(ds)} examples from {config}")

                    # Process examples
                    for example in tqdm(ds, desc=f"Processing {config}"):
                        if total_count >= num_sentences:
                            break

                        # Extract text field
                        text = None
                        if isinstance(example, dict):
                            # Try common field names
                            for field in ['text', 'sentence', 'content', 'doc', 'document']:
                                if field in example:
                                    text = example[field]
                                    break

                        if text and len(text.strip()) >= 10:
                            f.write(text.strip() + '\n')
                            total_count += 1

                    print(f"  ✓ Processed {config}")

                except Exception as e:
                    print(f"  ✗ Error with {config}: {e}")
                    continue

        print(f"\n{'='*60}")
        print(f"✓ Total downloaded: {total_count:,} sentences")
        print(f"  Saved to: {output_path}")
        if output_path.exists():
            print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        return total_count

    except Exception as e:
        print(f"Error loading DOLLMA: {e}")
        print("\nFalling back to manual text generation from UD dataset...")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download DOLLMA corpus sample")
    parser.add_argument('--output', type=str, default="data/raw/dollma_sample_100k.txt",
                        help="Output text file")
    parser.add_argument('--num-sentences', type=int, default=100000,
                        help="Number of sentences to download")

    args = parser.parse_args()

    download_dollma_direct(
        output_path=Path(args.output),
        num_sentences=args.num_sentences
    )


if __name__ == "__main__":
    main()
