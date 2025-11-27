#!/usr/bin/env python3
"""
Download DOLLMA corpus - Version 5 (FINAL)
Successfully downloads from multiple parquet sources
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import os
import re
from huggingface_hub import hf_hub_download


def check_auth():
    """Check if user is authenticated with HuggingFace"""
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        token_file = Path.home() / '.cache' / 'huggingface' / 'token'
        if token_file.exists():
            token = token_file.read_text().strip()

    if not token:
        print("⚠ Warning: No HuggingFace token found.")
        print("  Please login with: huggingface-cli login")
        return None

    return token


def download_dollma(output_path: Path, num_sentences: int, sources: list = None):
    """
    Download from DOLLMA using direct parquet method

    Args:
        output_path: Output file path
        num_sentences: Target number of sentences
        sources: List of parquet files to download from
    """
    token = check_auth()
    if not token:
        return 0

    # Default sources - prioritize high-quality sources
    if sources is None:
        sources = [
            "azwiki/train-00000-of-00001.parquet",           # Wikipedia - 129K rows
            "elite-books/train-00000-of-00001.parquet",       # Elite books
            "anl-news/train-00000-of-00004.parquet",          # News
            "anl-news/train-00001-of-00004.parquet",
            "mediocore-books/train-00000-of-00006.parquet",   # Books
            "mediocore-books/train-00001-of-00006.parquet",
        ]

    print(f"DOLLMA Download - Direct Parquet Method")
    print(f"Target: {num_sentences:,} sentences")
    print(f"Sources: {len(sources)} parquet files")
    print("="*60)

    total_count = 0
    target_per_source = max(1000, num_sentences // len(sources))

    with open(output_path, 'w', encoding='utf-8') as f:
        for source_file in sources:
            if total_count >= num_sentences:
                break

            print(f"\n📥 Downloading: {source_file}")

            try:
                # Download parquet file
                local_path = hf_hub_download(
                    repo_id="allmalab/DOLLMA",
                    filename=source_file,
                    repo_type="dataset",
                    token=token
                )

                # Read with pandas
                df = pd.read_parquet(local_path)
                print(f"   Loaded {len(df):,} rows")
                print(f"   Columns: {list(df.columns)}")

                # Find text column
                text_col = None
                for col in ['text', 'sentence', 'content', 'doc', 'document']:
                    if col in df.columns:
                        text_col = col
                        break

                if not text_col:
                    print(f"   ✗ No text column found, skipping")
                    continue

                # Extract sentences
                source_count = 0
                max_for_source = min(target_per_source, num_sentences - total_count)

                for idx, row in tqdm(df.iterrows(),
                                    total=min(max_for_source, len(df)),
                                    desc=f"   Processing",
                                    leave=False):

                    if source_count >= max_for_source or total_count >= num_sentences:
                        break

                    text = str(row[text_col]).strip()

                    # Split into sentences
                    sentences = re.split(r'[.!?]+\s+', text)

                    for sent in sentences:
                        sent = sent.strip()

                        # Filter valid sentences
                        if (len(sent) >= 10 and
                            len(sent) <= 500 and
                            source_count < max_for_source and
                            total_count < num_sentences):

                            f.write(sent + '\n')
                            total_count += 1
                            source_count += 1

                print(f"   ✓ Extracted {source_count:,} sentences (total: {total_count:,})")

            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue

    print("\n" + "="*60)
    print(f"✓ DOWNLOAD COMPLETE")
    print(f"  Total sentences: {total_count:,}")
    print(f"  Saved to: {output_path}")
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  File size: {size_mb:.1f} MB")
    print("="*60)

    return total_count


def main():
    parser = argparse.ArgumentParser(description="Download DOLLMA corpus (v5)")
    parser.add_argument('--output', type=str, default="data/raw/dollma_100k.txt",
                        help="Output text file")
    parser.add_argument('--num-sentences', type=int, default=100000,
                        help="Number of sentences to download")
    parser.add_argument('--source', type=str, default='mixed',
                        choices=['azwiki', 'books', 'news', 'mixed'],
                        help="Data source selection")

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select sources based on user preference
    if args.source == 'azwiki':
        sources = ["azwiki/train-00000-of-00001.parquet"]
    elif args.source == 'books':
        sources = [
            "elite-books/train-00000-of-00001.parquet",
            "mediocore-books/train-00000-of-00006.parquet",
            "mediocore-books/train-00001-of-00006.parquet",
            "mediocore-books/train-00002-of-00006.parquet",
        ]
    elif args.source == 'news':
        sources = [
            "anl-news/train-00000-of-00004.parquet",
            "anl-news/train-00001-of-00004.parquet",
            "anl-news/train-00002-of-00004.parquet",
            "anl-news/train-00003-of-00004.parquet",
        ]
    else:  # mixed
        sources = [
            "azwiki/train-00000-of-00001.parquet",
            "elite-books/train-00000-of-00001.parquet",
            "anl-news/train-00000-of-00004.parquet",
            "anl-news/train-00001-of-00004.parquet",
            "mediocore-books/train-00000-of-00006.parquet",
            "mediocore-books/train-00001-of-00006.parquet",
        ]

    download_dollma(output_path, args.num_sentences, sources)


if __name__ == "__main__":
    main()
