#!/usr/bin/env python3
"""
MorAz-based annotation script for DOLLMA corpus
Generates (word, context, lemma) triples for training
"""

import subprocess
import json
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import argparse


class MorAzAnnotator:
    """Wrapper for MorAz morphological analyzer"""

    def __init__(self, analyzer_path="moraz/az.inv.ol"):
        self.analyzer_path = Path(analyzer_path)
        if not self.analyzer_path.exists():
            raise FileNotFoundError(f"MorAz analyzer not found at {analyzer_path}")

        print(f"Loaded MorAz analyzer from {self.analyzer_path}")

    def analyze_word(self, word: str) -> str:
        """
        Analyze word and extract lemma using MorAz

        Args:
            word: Azerbaijani word to analyze

        Returns:
            Lemma of the word (or original word if unknown)
        """
        try:
            # Run hfst-lookup
            result = subprocess.run(
                ['hfst-lookup', str(self.analyzer_path)],
                input=word + '\n',
                capture_output=True,
                text=True,
                timeout=5
            )

            # Parse output: "kitabları    kitab<NOM><Num:Pl><Poss:No><Case:Acc>    0.000000"
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('>'):
                    continue

                # Split by tabs
                parts = line.split('\t')
                if len(parts) >= 2:
                    analysis = parts[1]

                    # Check if it's unknown
                    if '+?' in analysis or 'inf' in parts[-1]:
                        continue

                    # Extract lemma (before first <)
                    lemma = analysis.split('<')[0].strip()

                    # Return first valid analysis
                    if lemma and lemma != word:
                        return lemma

            # If no valid analysis, return original word
            return word

        except Exception as e:
            print(f"Error analyzing '{word}': {e}")
            return word

    def clean_word(self, word: str) -> str:
        """Remove punctuation from word"""
        # Remove common punctuation
        word = re.sub(r'[.,!?;:"""„«»\'"()[\]{}<>]', '', word)
        return word.strip()

    def annotate_sentence(self, sentence: str) -> List[Dict[str, any]]:
        """
        Annotate all words in a sentence

        Args:
            sentence: Azerbaijani sentence

        Returns:
            List of annotations with word, context, lemma, position
        """
        # Tokenize (simple space-based)
        words = sentence.split()
        annotations = []

        for idx, word in enumerate(words):
            # Clean word (remove punctuation)
            clean = self.clean_word(word)

            # Skip empty or very short words
            if len(clean) < 2:
                continue

            # Analyze
            lemma = self.analyze_word(clean)

            # Create annotation
            annotations.append({
                'word': clean,
                'context': sentence,
                'lemma': lemma,
                'position': idx
            })

        return annotations

    def process_file(self, input_path: Path, output_path: Path, max_examples: int = 50000):
        """
        Process text file and generate annotations

        Args:
            input_path: Path to input text file (one sentence per line)
            output_path: Path to output JSON file
            max_examples: Maximum number of examples to generate
        """
        print(f"\nProcessing {input_path}")
        print(f"Target: {max_examples} examples")

        examples = []
        line_count = 0

        # Count lines for progress bar
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        print(f"Total lines in file: {total_lines:,}")

        # Process file
        with open(input_path, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=max_examples, desc="Generating examples")

            for line in f:
                sentence = line.strip()
                line_count += 1

                # Skip empty lines
                if not sentence or len(sentence) < 10:
                    continue

                # Annotate sentence
                annotations = self.annotate_sentence(sentence)

                # Add to examples
                for ann in annotations:
                    examples.append(ann)
                    pbar.update(1)

                    # Check if we have enough
                    if len(examples) >= max_examples:
                        break

                if len(examples) >= max_examples:
                    break

            pbar.close()

        print(f"\nProcessed {line_count:,} sentences")
        print(f"Generated {len(examples):,} annotations")

        # Save annotations
        print(f"Saving to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved {len(examples):,} examples")

        return examples

    def verify_quality(self, examples: List[Dict], sample_size: int = 100):
        """
        Print sample of annotations for manual verification

        Args:
            examples: List of annotation dictionaries
            sample_size: Number of samples to display
        """
        import random

        print(f"\n{'='*80}")
        print(f"Quality Check: Random Sample of {sample_size} Annotations")
        print(f"{'='*80}\n")

        sample = random.sample(examples, min(sample_size, len(examples)))

        for i, ex in enumerate(sample, 1):
            word = ex['word']
            lemma = ex['lemma']
            context = ex['context'][:100] + "..." if len(ex['context']) > 100 else ex['context']

            # Check if lemma different from word
            status = "✓" if lemma != word else "="

            print(f"{i:3d}. {status} {word:15s} → {lemma:15s}")
            print(f"     Context: {context}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Generate lemmatization annotations using MorAz")
    parser.add_argument('--input', type=str, required=True, help="Input text file (one sentence per line)")
    parser.add_argument('--output', type=str, required=True, help="Output JSON file")
    parser.add_argument('--max-examples', type=int, default=50000, help="Maximum examples to generate")
    parser.add_argument('--analyzer', type=str, default="moraz/az.inv.ol", help="Path to MorAz analyzer")
    parser.add_argument('--verify', action='store_true', help="Show quality check sample after generation")

    args = parser.parse_args()

    # Create annotator
    annotator = MorAzAnnotator(analyzer_path=args.analyzer)

    # Process file
    examples = annotator.process_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        max_examples=args.max_examples
    )

    # Verify quality if requested
    if args.verify:
        annotator.verify_quality(examples, sample_size=50)

    print("\n✓ Annotation complete!")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Count:  {len(examples):,} examples")


if __name__ == "__main__":
    # If run without arguments, show test examples
    import sys

    if len(sys.argv) == 1:
        print("MorAz Annotator - Test Mode\n")

        annotator = MorAzAnnotator()

        # Test sentences
        test_sentences = [
            "Mən kitab oxuyuram.",
            "O məktəbə gedir.",
            "Biz evdə oturmuşuq.",
            "Onlar parkda gəzir.",
            "Sən nə edirsən?"
        ]

        print("Testing MorAz analyzer:\n")
        for sent in test_sentences:
            print(f"Sentence: {sent}")
            annotations = annotator.annotate_sentence(sent)
            for ann in annotations:
                print(f"  {ann['word']:15s} → {ann['lemma']}")
            print()

        print("\nTo process a file:")
        print("  python scripts/moraz_annotator.py --input data/raw/dollma_sample.txt --output data/processed/moraz_50k.json --max-examples 50000")
    else:
        main()
