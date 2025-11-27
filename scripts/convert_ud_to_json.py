"""
Convert Universal Dependencies CoNLL-U format to project JSON format.

This script converts UD Azerbaijani treebank data into the format expected
by the lemmatization model:
{
    "word": "inflected_form",
    "context": "full sentence",
    "lemma": "base_form"
}
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def parse_conllu_file(filepath: str) -> List[Dict]:
    """
    Parse a CoNLL-U file and extract lemmatization examples.

    Args:
        filepath: Path to .conllu file

    Returns:
        List of (word, context, lemma) examples
    """
    examples = []

    with open(filepath, 'r', encoding='utf-8') as f:
        current_sentence = []
        sentence_text = None

        for line in f:
            line = line.strip()

            # Skip empty lines (sentence boundary)
            if not line:
                if current_sentence and sentence_text:
                    # Process the sentence
                    examples.extend(
                        extract_examples_from_sentence(current_sentence, sentence_text)
                    )
                # Reset for next sentence
                current_sentence = []
                sentence_text = None
                continue

            # Extract sentence text
            if line.startswith('# text = '):
                sentence_text = line.replace('# text = ', '')
                continue

            # Skip other comments
            if line.startswith('#'):
                continue

            # Parse token line
            parts = line.split('\t')

            # Skip multi-word tokens (lines like "1-2")
            if '-' in parts[0] or '.' in parts[0]:
                continue

            # Extract: index, word, lemma, pos
            token_id = parts[0]
            word_form = parts[1]
            lemma = parts[2]
            pos_tag = parts[3]

            current_sentence.append({
                'id': token_id,
                'word': word_form,
                'lemma': lemma,
                'pos': pos_tag
            })

        # Process last sentence if exists
        if current_sentence and sentence_text:
            examples.extend(
                extract_examples_from_sentence(current_sentence, sentence_text)
            )

    return examples


def extract_examples_from_sentence(
    tokens: List[Dict],
    sentence_text: str
) -> List[Dict]:
    """
    Extract lemmatization examples from a parsed sentence.

    Args:
        tokens: List of token dictionaries with word, lemma, pos
        sentence_text: Full sentence text

    Returns:
        List of examples
    """
    examples = []

    for token in tokens:
        word = token['word']
        lemma = token['lemma']
        pos = token['pos']

        # Skip punctuation
        if pos == 'PUNCT':
            continue

        # Skip if lemma is underscore (missing annotation)
        if lemma == '_':
            continue

        # Skip if word and lemma are identical (no inflection)
        # Actually, keep these - they're still valid examples
        # if word == lemma:
        #     continue

        example = {
            'word': word,
            'context': sentence_text,
            'lemma': lemma,
            'pos': pos  # Keep POS for potential filtering later
        }

        examples.append(example)

    return examples


def convert_ud_treebank(
    input_file: str,
    output_file: str,
    include_pos: bool = False
) -> Dict:
    """
    Convert UD treebank to JSON format.

    Args:
        input_file: Path to .conllu file
        output_file: Path to output .json file
        include_pos: Whether to include POS tags in output

    Returns:
        Statistics dictionary
    """
    print(f"Converting {input_file}...")

    # Parse CoNLL-U file
    examples = parse_conllu_file(input_file)

    # Optionally remove POS tags
    if not include_pos:
        for example in examples:
            del example['pos']

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    # Collect statistics
    stats = {
        'total_examples': len(examples),
        'unique_words': len(set(ex['word'] for ex in examples)),
        'unique_lemmas': len(set(ex['lemma'] for ex in examples)),
        'avg_context_length': sum(len(ex['context'].split()) for ex in examples) / len(examples) if examples else 0
    }

    # POS distribution
    if include_pos or any('pos' in ex for ex in examples):
        pos_counts = {}
        for ex in examples:
            if 'pos' in ex:
                pos = ex['pos']
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        stats['pos_distribution'] = pos_counts

    print(f"\nConversion complete!")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Unique words: {stats['unique_words']}")
    print(f"  Unique lemmas: {stats['unique_lemmas']}")
    print(f"  Avg context length: {stats['avg_context_length']:.1f} words")

    if 'pos_distribution' in stats:
        print(f"\n  POS distribution:")
        for pos, count in sorted(stats['pos_distribution'].items(), key=lambda x: -x[1]):
            print(f"    {pos}: {count}")

    print(f"\nSaved to: {output_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert UD CoNLL-U format to project JSON format'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input .conllu file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output .json file'
    )

    parser.add_argument(
        '--include-pos',
        action='store_true',
        help='Include POS tags in output JSON'
    )

    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Only convert first N examples (for testing)'
    )

    args = parser.parse_args()

    # Convert
    stats = convert_ud_treebank(
        args.input,
        args.output,
        include_pos=args.include_pos
    )

    # If sampling requested, re-save with sample
    if args.sample and args.sample < stats['total_examples']:
        print(f"\nSampling {args.sample} examples...")
        with open(args.output, 'r', encoding='utf-8') as f:
            examples = json.load(f)

        sampled = examples[:args.sample]

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(sampled, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(sampled)} examples to {args.output}")

    # Save statistics
    stats_file = args.output.replace('.json', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
