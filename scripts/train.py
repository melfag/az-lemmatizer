"""
Main training script
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.lemmatizer import ContextAwareLemmatizer
from utils.data_loader import get_dataloaders
from utils.vocabulary import CharacterVocabulary
from training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function."""
    
    # Load config
    config = load_config(args.config)
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"\nUsing device: {device}")
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = CharacterVocabulary.load(config['vocab_path'])
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    print("\nCreating model...")
    model = ContextAwareLemmatizer(
        char_vocab_size=len(vocab),
        char_vocab=vocab,  # Add char_vocab for vowel harmony component
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
    
    # Load checkpoint if specified
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")
    
    # Load data
    print("\nLoading data...")
    import json
    from transformers import AutoTokenizer

    # Load examples from JSON files
    with open(config['data']['train_path'], 'r', encoding='utf-8') as f:
        train_examples = json.load(f)
    with open(config['data']['val_path'], 'r', encoding='utf-8') as f:
        val_examples = json.load(f)

    print(f"Loaded {len(train_examples)} training examples")
    print(f"Loaded {len(val_examples)} validation examples")

    # Load BERT tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_model_name'])

    # Create dataloaders
    train_dataloader, val_dataloader, _ = get_dataloaders(
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=[],  # Not using test set for training
        char_vocab=vocab,
        bert_tokenizer=bert_tokenizer,
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 0)
    )
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config['training'],
        device=device,
        checkpoint_dir=config['training']['checkpoint_dir']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Azerbaijani lemmatizer')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    main(args)