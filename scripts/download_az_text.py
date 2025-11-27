#!/usr/bin/env python3
"""
Download Azerbaijani text from Wikipedia and news sources
Simple, reliable method for getting Azerbaijani language data
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import requests
import re


def download_wikipedia_text(output_path: Path, target_sentences: int = 100000):
    """
    Download Azerbaijani text from Wikipedia

    Args:
        output_path: Path to save text
        target_sentences: Target number of sentences
    """
    print(f"Downloading Azerbaijani text from Wikipedia...")
    print(f"Target: {target_sentences:,} sentences")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Common Azerbaijani Wikipedia topics
    topics = [
        # Geography
        'Azərbaycan', 'Bakı', 'Gəncə', 'Sumqayıt', 'Mingəçevir',
        'Quba', 'Şəki', 'Naxçıvan', 'Lənkəran', 'Qəbələ',

        # History
        'Azərbaycan_tarixi', 'Azərbaycan_Demokratik_Respublikası',
        'Azərbaycan_SSR', 'Qafqaz_Albaniyası', 'Səfəvilər',

        # Culture
        'Azərbaycan_dili', 'Azərbaycan_ədəbiyyatı', 'Azərbaycan_musiqi',
        'Nizami_Gəncəvi', 'Füzuli', 'Nəsimi', 'Səməd_Vurğun',

        # Science
        'Riyaziyyat', 'Fizika', 'Kimya', 'Biologiya', 'Astronomiya',
        'İnformatika', 'Texnologiya', 'Elm',

        # Nature
        'Təbiət', 'Coğrafiya', 'İqlim', 'Flora', 'Fauna',

        # Society
        'Cəmiyyət', 'İqtisadiyyat', 'Siyasət', 'Təhsil', 'Mədəniyyət',
        'İncəsənət', 'Memarlıq', 'Bədii_sənət',

        # More topics
        'Sport', 'Futbol', 'Şahmat', 'İdman', 'Olimpiya_Oyunları',
        'Kitab', 'Teatr', 'Kino', 'Musiqi', 'Rəqs',
        'Fəlsəfə', 'Din', 'İslam', 'Xristianlıq',
        'Tibb', 'Səhiyyə', 'Gigiyena',
        'Neft', 'Qaz', 'Energetika', 'Sənaye',
        'Kənd_təsərrüfatı', 'Heyvandarlıq', 'Əkinçilik',
    ]

    total_sentences = 0

    try:
        import wikipedia
        wikipedia.set_lang('az')

        print(f"\nFetching {len(topics)} Wikipedia articles...")

        with open(output_path, 'w', encoding='utf-8') as f:
            for topic in tqdm(topics, desc="Downloading"):
                if total_sentences >= target_sentences:
                    break

                try:
                    # Get article
                    page = wikipedia.page(topic, auto_suggest=False)
                    content = page.content

                    # Split into sentences
                    # Azerbaijani uses . ! ? for sentence endings
                    sentences = re.split(r'[.!?]+\s+', content)

                    for sent in sentences:
                        sent = sent.strip()

                        # Filter valid sentences
                        if (len(sent) >= 10 and
                            len(sent) <= 500 and
                            not sent.startswith('=') and
                            total_sentences < target_sentences):

                            f.write(sent + '\n')
                            total_sentences += 1

                except wikipedia.exceptions.PageError:
                    continue
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try first option
                    try:
                        if e.options:
                            page = wikipedia.page(e.options[0])
                            content = page.content
                            sentences = re.split(r'[.!?]+\s+', content)

                            for sent in sentences:
                                sent = sent.strip()
                                if (len(sent) >= 10 and
                                    len(sent) <= 500 and
                                    not sent.startswith('=') and
                                    total_sentences < target_sentences):
                                    f.write(sent + '\n')
                                    total_sentences += 1
                    except:
                        continue
                except Exception as e:
                    continue

        print(f"\n{'='*60}")
        print(f"✓ Downloaded {total_sentences:,} sentences")
        print(f"  Saved to: {output_path}")
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"  File size: {size_mb:.1f} MB")

        return total_sentences

    except ImportError:
        print("\n✗ Wikipedia package not installed")
        print("  Install with: pip install wikipedia-api")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 0


def download_from_cc100(output_path: Path, target_sentences: int = 100000):
    """
    Alternative: Download from CC100 Azerbaijani corpus
    """
    print("\nTrying CC100 corpus...")

    try:
        from datasets import load_dataset

        print("Loading CC100 Azerbaijani...")
        ds = load_dataset("cc100", lang="az", split="train", streaming=True)

        total_sentences = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in tqdm(ds, total=target_sentences, desc="CC100"):
                if total_sentences >= target_sentences:
                    break

                text = example.get('text', '')

                # Split into sentences
                sentences = re.split(r'[.!?]+\s+', text)

                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) >= 10 and len(sent) <= 500:
                        f.write(sent + '\n')
                        total_sentences += 1

                        if total_sentences >= target_sentences:
                            break

        print(f"✓ Downloaded {total_sentences:,} sentences from CC100")
        return total_sentences

    except Exception as e:
        print(f"✗ CC100 failed: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download Azerbaijani text")
    parser.add_argument('--output', type=str, default="data/raw/az_text_100k.txt",
                        help="Output text file")
    parser.add_argument('--num-sentences', type=int, default=100000,
                        help="Number of sentences to download")
    parser.add_argument('--method', type=str, default='wikipedia',
                        choices=['wikipedia', 'cc100', 'both'],
                        help="Download method")

    args = parser.parse_args()

    output_path = Path(args.output)

    if args.method == 'wikipedia':
        download_wikipedia_text(output_path, args.num_sentences)
    elif args.method == 'cc100':
        download_from_cc100(output_path, args.num_sentences)
    else:
        # Try Wikipedia first, then CC100 if needed
        count = download_wikipedia_text(output_path, args.num_sentences)
        if count < args.num_sentences:
            print(f"\nNeed {args.num_sentences - count:,} more sentences...")
            download_from_cc100(output_path, args.num_sentences - count)


if __name__ == "__main__":
    main()
