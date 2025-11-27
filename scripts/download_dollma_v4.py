#!/usr/bin/env python3
"""
Download DOLLMA corpus - Version 4
Trying multiple methods: Polars, Dask, Datasets, and direct Parquet
"""

from pathlib import Path
import argparse
import os
import re
from tqdm import tqdm


def check_auth():
    """Check if user is authenticated with HuggingFace"""
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        # Try to read from file
        token_file = Path.home() / '.cache' / 'huggingface' / 'token'
        if token_file.exists():
            token = token_file.read_text().strip()

    if not token:
        print("⚠ Warning: No HuggingFace token found.")
        print("  Please login with: huggingface-cli login")
        print("  Or set: export HF_TOKEN=your_token")
        return None

    return token


def method_1_polars(output_path: Path, num_sentences: int):
    """
    Method 1: Download using Polars
    Polars can read Parquet files directly from HuggingFace
    """
    print("\n" + "="*60)
    print("METHOD 1: Polars")
    print("="*60)

    try:
        import polars as pl
        from huggingface_hub import hf_hub_download

        token = check_auth()
        if not token:
            return 0

        print("Downloading parquet file...")

        # Download first parquet file - using azwiki
        local_path = hf_hub_download(
            repo_id="allmalab/DOLLMA",
            filename="azwiki/train-00000-of-00001.parquet",
            repo_type="dataset",
            token=token
        )

        print(f"Reading parquet with Polars: {local_path}")
        df = pl.read_parquet(local_path)

        print(f"Columns: {df.columns}")
        print(f"Shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")

        # Find text column
        text_col = None
        for col in ['text', 'sentence', 'content', 'doc', 'document']:
            if col in df.columns:
                text_col = col
                break

        if not text_col:
            print(f"✗ No text column found. Available: {df.columns}")
            return 0

        # Extract sentences
        print(f"\nExtracting sentences from column: {text_col}")
        total_count = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for row in tqdm(df.iter_rows(named=True), total=min(num_sentences, len(df)), desc="Processing"):
                if total_count >= num_sentences:
                    break

                text = str(row[text_col]).strip()

                # Split into sentences
                sentences = re.split(r'[.!?]+\s+', text)

                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) >= 10 and len(sent) <= 500:
                        f.write(sent + '\n')
                        total_count += 1

                        if total_count >= num_sentences:
                            break

        print(f"✓ Polars method: {total_count:,} sentences")
        return total_count

    except ImportError:
        print("✗ Polars not installed: pip install polars")
        return 0
    except Exception as e:
        print(f"✗ Polars method failed: {e}")
        return 0


def method_2_dask(output_path: Path, num_sentences: int):
    """
    Method 2: Download using Dask
    Dask can handle large datasets with parallel processing
    """
    print("\n" + "="*60)
    print("METHOD 2: Dask")
    print("="*60)

    try:
        import dask.dataframe as dd
        from huggingface_hub import hf_hub_download

        token = check_auth()
        if not token:
            return 0

        print("Downloading parquet file...")

        # Download first parquet file - using azwiki
        local_path = hf_hub_download(
            repo_id="allmalab/DOLLMA",
            filename="azwiki/train-00000-of-00001.parquet",
            repo_type="dataset",
            token=token
        )

        print(f"Reading parquet with Dask: {local_path}")
        df = dd.read_parquet(local_path)

        print(f"Columns: {df.columns.tolist()}")
        print(f"Computing sample...")
        sample = df.head(5).compute()
        print(sample)

        # Find text column
        text_col = None
        for col in ['text', 'sentence', 'content', 'doc', 'document']:
            if col in df.columns:
                text_col = col
                break

        if not text_col:
            print(f"✗ No text column found. Available: {df.columns.tolist()}")
            return 0

        # Extract sentences
        print(f"\nExtracting sentences from column: {text_col}")
        total_count = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            # Compute in chunks
            for partition in tqdm(df.to_delayed(), desc="Processing partitions"):
                if total_count >= num_sentences:
                    break

                part_df = partition.compute()

                for _, row in part_df.iterrows():
                    if total_count >= num_sentences:
                        break

                    text = str(row[text_col]).strip()
                    sentences = re.split(r'[.!?]+\s+', text)

                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) >= 10 and len(sent) <= 500:
                            f.write(sent + '\n')
                            total_count += 1

                            if total_count >= num_sentences:
                                break

        print(f"✓ Dask method: {total_count:,} sentences")
        return total_count

    except ImportError:
        print("✗ Dask not installed: pip install dask[dataframe]")
        return 0
    except Exception as e:
        print(f"✗ Dask method failed: {e}")
        return 0


def method_3_datasets_parquet(output_path: Path, num_sentences: int):
    """
    Method 3: HuggingFace Datasets with direct parquet reading
    Bypass the loading script issue
    """
    print("\n" + "="*60)
    print("METHOD 3: Datasets (Parquet Direct)")
    print("="*60)

    try:
        from datasets import load_dataset

        token = check_auth()
        if not token:
            return 0

        print("Loading dataset from parquet files...")

        # Load directly from parquet files, not using the loading script
        ds = load_dataset(
            "parquet",
            data_files={
                "train": "hf://datasets/allmalab/DOLLMA/azwiki/train-00000-of-00001.parquet"
            },
            split="train",
            token=token
        )

        print(f"Dataset loaded: {len(ds)} examples")
        print(f"Features: {ds.features}")
        print(f"Sample: {ds[0]}")

        # Find text column
        text_col = None
        for col in ['text', 'sentence', 'content', 'doc', 'document']:
            if col in ds.features:
                text_col = col
                break

        if not text_col:
            print(f"✗ No text column found. Available: {list(ds.features.keys())}")
            return 0

        # Extract sentences
        print(f"\nExtracting sentences from column: {text_col}")
        total_count = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in tqdm(ds, total=min(num_sentences, len(ds)), desc="Processing"):
                if total_count >= num_sentences:
                    break

                text = str(example[text_col]).strip()
                sentences = re.split(r'[.!?]+\s+', text)

                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) >= 10 and len(sent) <= 500:
                        f.write(sent + '\n')
                        total_count += 1

                        if total_count >= num_sentences:
                            break

        print(f"✓ Datasets method: {total_count:,} sentences")
        return total_count

    except Exception as e:
        print(f"✗ Datasets method failed: {e}")
        return 0


def method_4_direct_parquet(output_path: Path, num_sentences: int):
    """
    Method 4: Direct Parquet download with pandas/pyarrow
    Most basic approach
    """
    print("\n" + "="*60)
    print("METHOD 4: Direct Parquet (pandas/pyarrow)")
    print("="*60)

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        token = check_auth()
        if not token:
            return 0

        print("Downloading parquet file...")

        # Download first parquet file - using azwiki
        local_path = hf_hub_download(
            repo_id="allmalab/DOLLMA",
            filename="azwiki/train-00000-of-00001.parquet",
            repo_type="dataset",
            token=token
        )

        print(f"Reading parquet with pandas: {local_path}")
        df = pd.read_parquet(local_path)

        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")

        # Find text column
        text_col = None
        for col in ['text', 'sentence', 'content', 'doc', 'document']:
            if col in df.columns:
                text_col = col
                break

        if not text_col:
            print(f"✗ No text column found. Available: {list(df.columns)}")
            return 0

        # Extract sentences
        print(f"\nExtracting sentences from column: {text_col}")
        total_count = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, row in tqdm(df.iterrows(), total=min(num_sentences, len(df)), desc="Processing"):
                if total_count >= num_sentences:
                    break

                text = str(row[text_col]).strip()
                sentences = re.split(r'[.!?]+\s+', text)

                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) >= 10 and len(sent) <= 500:
                        f.write(sent + '\n')
                        total_count += 1

                        if total_count >= num_sentences:
                            break

        print(f"✓ Direct parquet method: {total_count:,} sentences")
        return total_count

    except Exception as e:
        print(f"✗ Direct parquet method failed: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download DOLLMA corpus - Multiple Methods")
    parser.add_argument('--output', type=str, default="data/raw/dollma_100k.txt",
                        help="Output text file")
    parser.add_argument('--num-sentences', type=int, default=100000,
                        help="Number of sentences to download")
    parser.add_argument('--method', type=str, default='auto',
                        choices=['polars', 'dask', 'datasets', 'parquet', 'auto'],
                        help="Download method (auto tries all)")

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"DOLLMA Download Tool v4")
    print(f"Target: {args.num_sentences:,} sentences")
    print(f"Output: {output_path}")

    methods = {
        'polars': method_1_polars,
        'dask': method_2_dask,
        'datasets': method_3_datasets_parquet,
        'parquet': method_4_direct_parquet,
    }

    if args.method == 'auto':
        # Try all methods until one succeeds
        for name, method_func in methods.items():
            print(f"\n{'='*60}")
            print(f"Trying method: {name}")
            print(f"{'='*60}")

            count = method_func(output_path, args.num_sentences)

            if count > 0:
                print(f"\n{'='*60}")
                print(f"✓ SUCCESS with {name} method!")
                print(f"  Downloaded: {count:,} sentences")
                print(f"  Saved to: {output_path}")
                if output_path.exists():
                    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
                print(f"{'='*60}")
                return

        print("\n✗ All methods failed. Please check:")
        print("  1. HuggingFace authentication: huggingface-cli login")
        print("  2. Accept DOLLMA terms at: https://huggingface.co/datasets/allmalab/DOLLMA")
        print("  3. Install required libraries: pip install polars dask[dataframe] datasets pyarrow")

    else:
        # Use specific method
        method_func = methods[args.method]
        count = method_func(output_path, args.num_sentences)

        if count > 0:
            print(f"\n{'='*60}")
            print(f"✓ Downloaded {count:,} sentences")
            print(f"  Saved to: {output_path}")
            if output_path.exists():
                print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
            print(f"{'='*60}")
        else:
            print(f"\n✗ {args.method} method failed")


if __name__ == "__main__":
    main()
