"""
Data preparation script for Azerbaijani lemmatization.

This script:
1. Loads the DOLLMA dataset
2. Creates or loads lemmatization examples
3. Splits data into train/val/test sets
4. Builds character vocabulary
5. Saves everything to disk
"""

import argparse
import json
import yaml
from pathlib import Path
from transformers import AutoTokenizer

from utils.vocabulary import CharacterVocabulary
from utils.data_loader import DOLLMADataProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data for lemmatization')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='allmalab/DOLLMA',
        help='HuggingFace dataset name'
    )
    
    parser.add_argument(
        '--annotated-file',
        type=str,
        default=None,
        help='Path to pre-annotated examples (JSON file)'
    )
    
    parser.add_argument(
        '--num-examples',
        type=int,
        default=10000,
        help='Number of synthetic examples to create (if no annotated file)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--bert-model',
        type=str,
        default='allmalab/bert-base-aze',
        help='BERT model for tokenizer'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("DATA PREPARATION FOR AZERBAIJANI LEMMATIZATION")
    print("=" * 80)
    
    # Initialize processor
    processor = DOLLMADataProcessor(dataset_name=args.dataset)
    
    # Load or create examples
    if args.annotated_file:
        print(f"\nLoading annotated examples from {args.annotated_file}")
        examples = processor.load_annotated_examples(args.annotated_file)
    else:
        print(f"\nCreating synthetic examples from DOLLMA dataset")
        print("WARNING: Using synthetic examples for demonstration.")
        print("For production, use properly annotated data with real lemmas.")
        
        # Load raw dataset
        raw_dataset = processor.load_raw_dataset()
        
        # Get texts from dataset
        # Note: Adjust this based on actual DOLLMA structure
        if 'train' in raw_dataset:
            texts = raw_dataset['train']['text'][:args.num_examples // 10]
        elif 'text' in raw_dataset:
            texts = raw_dataset['text'][:args.num_examples // 10]
        else:
            # Explore dataset structure
            print("\nDataset structure:")
            print(raw_dataset)
            print("\nPlease adjust the script based on actual DOLLMA structure")
            return
        
        # Create synthetic examples
        examples = processor.create_synthetic_examples(
            texts, 
            num_examples=args.num_examples
        )
    
    print(f"\nTotal examples: {len(examples)}")
    
    # Show sample examples
    print("\nSample examples:")
    for i, ex in enumerate(examples[:3]):
        print(f"\n{i+1}. Word: {ex['word']}")
        print(f"   Context: {ex['context'][:80]}...")
        print(f"   Lemma: {ex['lemma']}")
    
    # Split data
    print(f"\nSplitting data...")
    train_examples, val_examples, test_examples = processor.split_data(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Save split data
    print(f"\nSaving split data to {output_dir}")
    processor.save_split_data(
        train_examples,
        val_examples,
        test_examples,
        str(output_dir)
    )
    
    # Build character vocabulary
    print(f"\nBuilding character vocabulary...")
    char_vocab = CharacterVocabulary()
    
    # Add characters from all examples
    all_texts = []
    for ex in examples:
        all_texts.extend([ex['word'], ex['lemma'], ex['context']])
    
    char_vocab.add_from_data(all_texts)
    
    print(f"Vocabulary size: {char_vocab.vocab_size}")
    
    # Save vocabulary
    vocab_path = output_dir / 'char_vocab.json'
    char_vocab.save(str(vocab_path))
    print(f"Saved vocabulary to {vocab_path}")
    
    # Load BERT tokenizer and save info
    print(f"\nLoading BERT tokenizer: {args.bert_model}")
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    
    # Save metadata
    metadata = {
        'dataset': args.dataset,
        'num_examples': len(examples),
        'train_examples': len(train_examples),
        'val_examples': len(val_examples),
        'test_examples': len(test_examples),
        'char_vocab_size': char_vocab.vocab_size,
        'bert_model': args.bert_model,
        'bert_vocab_size': bert_tokenizer.vocab_size,
        'splits': {
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio
        },
        'seed': args.seed
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {metadata_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nStatistics:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Training: {len(train_examples)} ({args.train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_examples)} ({args.val_ratio*100:.0f}%)")
    print(f"  Test: {len(test_examples)} ({args.test_ratio*100:.0f}%)")
    print(f"  Character vocabulary size: {char_vocab.vocab_size}")
    print(f"  BERT vocabulary size: {bert_tokenizer.vocab_size}")
    
    print(f"\nOutput files:")
    print(f"  {output_dir}/train.json")
    print(f"  {output_dir}/val.json")
    print(f"  {output_dir}/test.json")
    print(f"  {output_dir}/char_vocab.json")
    print(f"  {output_dir}/metadata.json")
    
    print("\n✓ Data preparation successful!")
    print("\nNext step: python scripts/train.py")


if __name__ == "__main__":
    main()