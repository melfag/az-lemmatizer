"""
Create specialized test sets for detailed evaluation.

This script generates:
1. Ambiguity test set
2. Morphological complexity test set
3. Linguistic phenomena test sets (vowel harmony, consonant alternation, etc.)

Based on Section 5.3 from the thesis.
"""

import argparse
import json
from pathlib import Path

from utils.specialized_tests import SpecializedTestSetGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create specialized test sets for lemmatization evaluation'
    )
    
    parser.add_argument(
        '--test-file',
        type=str,
        required=True,
        help='Path to test data file (JSON)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/specialized_tests',
        help='Output directory for test sets'
    )
    
    parser.add_argument(
        '--ambiguity-size',
        type=int,
        default=10000,
        help='Target size for ambiguity test set (default: 10000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--test-sets',
        nargs='+',
        choices=['ambiguity', 'complexity', 'phenomena', 'all'],
        default=['all'],
        help='Which test sets to create'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("SPECIALIZED TEST SET CREATION")
    print("=" * 80)
    
    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_examples = json.load(f)
    
    print(f"Loaded {len(test_examples)} test examples")
    
    # Create generator
    print(f"\nInitializing test set generator (seed={args.seed})...")
    generator = SpecializedTestSetGenerator(seed=args.seed)
    
    # Determine which test sets to create
    create_all = 'all' in args.test_sets
    test_sets_to_create = {
        'ambiguity': create_all or 'ambiguity' in args.test_sets,
        'complexity': create_all or 'complexity' in args.test_sets,
        'phenomena': create_all or 'phenomena' in args.test_sets
    }
    
    # Store generated test sets
    generated_test_sets = {}
    
    # 1. Ambiguity Test Set
    if test_sets_to_create['ambiguity']:
        print("\n" + "-" * 80)
        print("Creating Ambiguity Test Set (Section 5.3.1)")
        print("-" * 80)
        
        ambiguity_set = generator.create_ambiguity_test_set(
            test_examples,
            target_size=args.ambiguity_size
        )
        
        generated_test_sets['ambiguity'] = ambiguity_set
        
        print(f"\n✓ Created ambiguity test set with {len(ambiguity_set)} examples")
        
        # Show sample ambiguous words
        if len(ambiguity_set) > 0:
            word_to_lemmas = {}
            for ex in ambiguity_set[:100]:  # Sample first 100
                word = ex['word']
                lemma = ex['lemma']
                if word not in word_to_lemmas:
                    word_to_lemmas[word] = set()
                word_to_lemmas[word].add(lemma)
            
            print("\nSample ambiguous words:")
            for word, lemmas in list(word_to_lemmas.items())[:5]:
                if len(lemmas) > 1:
                    print(f"  '{word}' → {', '.join(lemmas)}")
    
    # 2. Morphological Complexity Test Set
    if test_sets_to_create['complexity']:
        print("\n" + "-" * 80)
        print("Creating Morphological Complexity Test Set (Section 5.3.2)")
        print("-" * 80)
        
        complexity_set = generator.create_morphological_complexity_test_set(
            test_examples
        )
        
        generated_test_sets['morphological_complexity'] = complexity_set
        
        print(f"\n✓ Created complexity test set with {len(complexity_set)} examples")
    
    # 3. Linguistic Phenomena Test Sets
    if test_sets_to_create['phenomena']:
        print("\n" + "-" * 80)
        print("Creating Linguistic Phenomena Test Sets (Section 5.3.3)")
        print("-" * 80)
        
        phenomena_sets = generator.create_linguistic_phenomena_test_sets(
            test_examples
        )
        
        # Add to generated test sets
        generated_test_sets.update(phenomena_sets)
        
        print(f"\n✓ Created {len(phenomena_sets)} linguistic phenomena test sets")
    
    # Save all test sets
    if generated_test_sets:
        print("\n" + "=" * 80)
        print("SAVING TEST SETS")
        print("=" * 80)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generator.save_test_sets(generated_test_sets, args.output_dir)
        
        # Create summary
        summary = {
            'source_file': args.test_file,
            'num_source_examples': len(test_examples),
            'seed': args.seed,
            'test_sets': {}
        }
        
        for test_name, test_set in generated_test_sets.items():
            summary['test_sets'][test_name] = {
                'num_examples': len(test_set),
                'file': f'{test_name}_test.json'
            }
        
        # Save summary
        summary_path = output_dir / 'test_sets_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved summary to {summary_path}")
        
        # Print final summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        print(f"\nSource: {args.test_file} ({len(test_examples)} examples)")
        print(f"Output: {args.output_dir}")
        print(f"\nGenerated test sets:")
        
        for test_name, info in summary['test_sets'].items():
            print(f"  ✓ {test_name}: {info['num_examples']} examples")
        
        print("\n" + "=" * 80)
        print("✓ Specialized test set creation complete!")
        print("=" * 80)
        
        print("\nYou can now use these test sets for detailed evaluation:")
        print("  python scripts/evaluate.py --specialized-tests")
    else:
        print("\n⚠ No test sets were created. Check your --test-sets argument.")


if __name__ == "__main__":
    main()