#!/usr/bin/env python3
"""
Download DOLLMA corpus - Version 3
Using manual parquet file download
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import os


def download_dollma_parquet(output_path: Path, num_sentences: int = 100000):
    """
    Download DOLLMA from parquet files manually

    Args:
        output_path: Path to save the text file
        num_sentences: Number of sentences to download
    """
    print(f"Downloading DOLLMA corpus from parquet...")
    print(f"Target: {num_sentences:,} sentences")
    print(f"Output: {output_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Check HuggingFace token
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        if not token:
            # Try to read from file
            token_file = Path.home() / '.cache' / 'huggingface' / 'token'
            if token_file.exists():
                token = token_file.read_text().strip()

        if not token:
            print("⚠ Warning: No HuggingFace token found. This may limit access.")
            print("  You can set it with: export HF_TOKEN=your_token")

        # Try to download using huggingface_hub
        from huggingface_hub import hf_hub_download, list_repo_files

        print("\nFetching repository file list...")
        files = list_repo_files(
            repo_id="allmalab/DOLLMA",
            repo_type="dataset",
            token=token
        )

        # Filter for parquet files
        parquet_files = [f for f in files if f.endswith('.parquet')]
        print(f"Found {len(parquet_files)} parquet files")

        if not parquet_files:
            raise Exception("No parquet files found in repository")

        # Download and process parquet files
        total_count = 0
        target_per_file = max(5000, num_sentences // len(parquet_files))

        with open(output_path, 'w', encoding='utf-8') as out_f:
            for pf in parquet_files[:10]:  # Limit to first 10 files
                if total_count >= num_sentences:
                    break

                print(f"\nDownloading: {pf}")
                try:
                    # Download file
                    local_path = hf_hub_download(
                        repo_id="allmalab/DOLLMA",
                        filename=pf,
                        repo_type="dataset",
                        token=token,
                        cache_dir=Path.home() / '.cache' / 'huggingface'
                    )

                    # Read parquet
                    df = pd.read_parquet(local_path)
                    print(f"  Loaded {len(df)} rows")
                    print(f"  Columns: {list(df.columns)}")

                    # Find text column
                    text_col = None
                    for col in ['text', 'sentence', 'content', 'doc', 'document']:
                        if col in df.columns:
                            text_col = col
                            break

                    if not text_col:
                        print(f"  ✗ No text column found, skipping")
                        continue

                    # Extract sentences
                    count = 0
                    for idx, row in df.iterrows():
                        if total_count >= num_sentences:
                            break

                        text = str(row[text_col]).strip()
                        if len(text) >= 10:
                            out_f.write(text + '\n')
                            total_count += 1
                            count += 1

                        if count >= target_per_file:
                            break

                    print(f"  ✓ Extracted {count:,} sentences (total: {total_count:,})")

                except Exception as e:
                    print(f"  ✗ Error processing {pf}: {e}")
                    continue

        print(f"\n{'='*60}")
        print(f"✓ Total downloaded: {total_count:,} sentences")
        print(f"  Saved to: {output_path}")
        if output_path.exists():
            print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        return total_count

    except Exception as e:
        print(f"Error: {e}")
        print("\nFalling back to alternative method...")
        return fallback_method(output_path, num_sentences)


def fallback_method(output_path: Path, num_sentences: int):
    """
    Fallback: Use Azerbaijani Wikipedia or sample text
    """
    print("\nUsing fallback: Generating from available data...")

    try:
        import wikipedia
        wikipedia.set_lang('az')

        total_count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            # Get random articles
            search_terms = [
                'Azərbaycan', 'Bakı', 'tarix', 'mədəniyyət', 'dil',
                'ədəbiyyat', 'elm', 'sənət', 'coğrafiya', 'təbiət'
            ]

            for term in tqdm(search_terms, desc="Fetching Wikipedia"):
                try:
                    page = wikipedia.page(term, auto_suggest=False)
                    content = page.content

                    # Split into sentences (simple split by . ! ?)
                    import re
                    sentences = re.split(r'[.!?]+', content)

                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) >= 10 and total_count < num_sentences:
                            f.write(sent + '\n')
                            total_count += 1

                except Exception as e:
                    continue

        print(f"✓ Generated {total_count:,} sentences from Wikipedia")
        return total_count

    except:
        print("✗ Fallback also failed. Please check internet connection and try again.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download DOLLMA corpus")
    parser.add_argument('--output', type=str, default="data/raw/dollma_sample_100k.txt",
                        help="Output text file")
    parser.add_argument('--num-sentences', type=int, default=100000,
                        help="Number of sentences to download")

    args = parser.parse_args()

    download_dollma_parquet(
        output_path=Path(args.output),
        num_sentences=args.num_sentences
    )


if __name__ == "__main__":
    main()
