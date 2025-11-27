"""
Evaluation script for the trained lemmatizer
"""

import torch
import yaml
import argparse
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.lemmatizer import ContextAwareLemmatizer
from utils.data_loader import create_dataloader
from utils.vocabulary import CharacterVocabulary
from evaluation.evaluator import Evaluator
from transformers import AutoTokenizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main evaluation function."""

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load checkpoint first to get config
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config from checkpoint or load from file
    if 'config' in checkpoint and args.config is None:
        config = checkpoint['config']
        print("Using config from checkpoint")
    elif args.config:
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        raise ValueError("No config found in checkpoint and no --config specified")

    # Load vocabulary
    vocab_path = args.vocab if args.vocab else config.get('vocab_path', 'data/processed/char_vocab.json')
    print(f"\nLoading vocabulary from {vocab_path}...")
    vocab = CharacterVocabulary.load(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")

    # Load BERT tokenizer
    bert_model_name = config['model']['bert_model_name']
    print(f"\nLoading BERT tokenizer: {bert_model_name}...")
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    # Create model
    print("\nCreating model...")
    model = ContextAwareLemmatizer(
        char_vocab_size=len(vocab),
        char_vocab=vocab,
        char_embedding_dim=config['model']['char_embedding_dim'],
        char_hidden_dim=config['model']['char_hidden_dim'],
        char_num_layers=config['model']['char_num_layers'],
        bert_model_name=config['model']['bert_model_name'],
        bert_freeze_layers=config['model']['bert_freeze_layers'],
        fusion_output_dim=config['model']['fusion_output_dim'],
        decoder_hidden_dim=config['model']['decoder_hidden_dim'],
        decoder_num_layers=config['model']['decoder_num_layers'],
        dropout=config['model']['dropout'],
        use_copy=config['model']['use_copy']
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")

    # Load test data
    test_path = args.test_data if args.test_data else config['data'].get('test_path')
    if not test_path:
        raise ValueError("No test data path specified. Use --test-data or ensure it's in config")

    print(f"\nLoading test data from {test_path}...")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_examples = json.load(f)
    print(f"Loaded {len(test_examples)} test examples")

    # Create dataloader
    batch_size = args.batch_size if args.batch_size else config['training'].get('batch_size', 32)
    test_dataloader = create_dataloader(
        examples=test_examples,
        char_vocab=vocab,
        bert_tokenizer=bert_tokenizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Test batches: {len(test_dataloader)}")

    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = Evaluator(
        model=model,
        vocab=vocab,
        device=device,
        output_dir=args.output_dir
    )

    # Main evaluation
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)

    results = evaluator.evaluate(
        dataloader=test_dataloader,
        dataset_name="test",
        save_predictions=args.save_predictions
    )

    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the Azerbaijani lemmatizer')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data JSON file'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default=None,
        help='Path to vocabulary file (default: use from config or data/processed/char_vocab.json)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional if stored in checkpoint)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for evaluation (default: use from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to file'
    )

    args = parser.parse_args()
    main(args)