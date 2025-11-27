"""
Integration tests for the complete pipeline
"""

import unittest
import tempfile
import json
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.lemmatizer import ContextAwareLemmatizer
from utils.vocabulary import CharacterVocabulary
from utils.data_loader import create_dataloader
from training.losses import CompositeLoss


class TestEndToEndPipeline(unittest.TestCase):
    """Test end-to-end pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Create minimal test data
        self.test_data = [
            {
                "input_word": "kitablar",
                "lemma": "kitab",
                "context": "Mən kitablar oxuyuram",
                "pos": "NOUN"
            },
            {
                "input_word": "oxuyuram",
                "lemma": "oxumaq",
                "context": "Mən kitablar oxuyuram",
                "pos": "VERB"
            }
        ]
        
        # Create vocabulary
        self.vocab = CharacterVocabulary()
        texts = []
        for item in self.test_data:
            texts.append(item['input_word'])
            texts.append(item['lemma'])
        self.vocab.build_from_texts(texts)
        
        # Create temporary data file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        )
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
        
        # Device
        self.device = torch.device('cpu')
    
    def tearDown(self):
        """Clean up."""
        Path(self.temp_file.name).unlink()
    
    def test_model_creation(self):
        """Test creating the model."""
        model = ContextAwareLemmatizer(
            char_vocab_size=len(self.vocab),
            char_embedding_dim=64,
            char_hidden_dim=128,
            char_num_layers=1,
            bert_model_name="bert-base-multilingual-cased",  # Use smaller model for testing
            bert_freeze_layers=0,
            fusion_output_dim=256,
            decoder_hidden_dim=256,
            decoder_num_layers=1,
            dropout=0.1,
            use_copy=False  # Disable for faster testing
        )
        
        # Check model is created
        self.assertIsNotNone(model)
        
        # Check model has expected components
        self.assertIsNotNone(model.character_encoder)
        self.assertIsNotNone(model.contextual_encoder)
        self.assertIsNotNone(model.fusion)
        self.assertIsNotNone(model.decoder)
    
    def test_dataloader_creation(self):
        """Test creating dataloader."""
        dataloader = create_dataloader(
            data_path=self.temp_file.name,
            vocab=self.vocab,
            batch_size=2,
            shuffle=False
        )
        
        # Check dataloader is created
        self.assertIsNotNone(dataloader)
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Check batch has expected keys
        self.assertIn('input_char_ids', batch)
        self.assertIn('target_char_ids', batch)
        self.assertIn('context_sentences', batch)
    
    @unittest.skip("Requires AllmaBERT model - slow test")
    def test_forward_pass(self):
        """Test model forward pass."""
        # Create small model
        model = ContextAwareLemmatizer(
            char_vocab_size=len(self.vocab),
            char_embedding_dim=32,
            char_hidden_dim=64,
            char_num_layers=1,
            bert_model_name="bert-base-multilingual-cased",
            bert_freeze_layers=11,
            fusion_output_dim=128,
            decoder_hidden_dim=128,
            decoder_num_layers=1,
            dropout=0.0,
            use_copy=False
        ).to(self.device)
        
        # Create dataloader
        dataloader = create_dataloader(
            data_path=self.temp_file.name,
            vocab=self.vocab,
            batch_size=2,
            shuffle=False
        )
        
        # Get batch
        batch = next(iter(dataloader))
        
        # Move to device
        input_char_ids = batch['input_char_ids'].to(self.device)
        input_lengths = batch['input_lengths'].to(self.device)
        target_char_ids = batch['target_char_ids'].to(self.device)
        
        # Forward pass
        try:
            outputs = model(
                input_char_ids=input_char_ids,
                input_lengths=input_lengths,
                context_sentences=batch['context_sentences'],
                target_word_positions=batch['target_positions'],
                target_char_ids=target_char_ids,
                teacher_forcing_ratio=0.5
            )
            
            # Check outputs
            self.assertIn('outputs', outputs)
            self.assertIn('attentions', outputs)
            
            # Check shapes
            batch_size = input_char_ids.size(0)
            self.assertEqual(outputs['outputs'].size(0), batch_size)
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_loss_computation(self):
        """Test loss computation."""
        batch_size = 2
        seq_len = 10
        vocab_size = len(self.vocab)
        
        # Create dummy predictions and targets
        predictions = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create loss function
        criterion = CompositeLoss(
            char_hidden_dim=128,
            smoothing=0.1,
            char_classification_weight=0.1,
            attention_coverage_weight=0.05
        )
        
        # Compute loss
        loss, loss_dict = criterion(
            predictions=predictions,
            targets=targets
        )
        
        # Check loss is computed
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)
        
        # Check loss dict
        self.assertIn('total_loss', loss_dict)
        self.assertIn('main_loss', loss_dict)


class TestModelSaveLoad(unittest.TestCase):
    """Test model saving and loading."""
    
    def test_checkpoint_save_load(self):
        """Test saving and loading model checkpoint."""
        # Create small vocabulary
        vocab = CharacterVocabulary()
        vocab.build_from_texts(["kitab", "oxumaq"])
        
        # Create small model
        model = ContextAwareLemmatizer(
            char_vocab_size=len(vocab),
            char_embedding_dim=32,
            char_hidden_dim=64,
            char_num_layers=1,
            bert_model_name="bert-base-multilingual-cased",
            bert_freeze_layers=11,
            fusion_output_dim=128,
            decoder_hidden_dim=128,
            decoder_num_layers=1,
            dropout=0.0,
            use_copy=False
        )
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'vocab_size': len(vocab),
            'config': {}
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            torch.save(checkpoint, temp_path)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(temp_path, map_location='cpu')
            
            # Create new model and load state
            new_model = ContextAwareLemmatizer(
                char_vocab_size=len(vocab),
                char_embedding_dim=32,
                char_hidden_dim=64,
                char_num_layers=1,
                bert_model_name="bert-base-multilingual-cased",
                bert_freeze_layers=11,
                fusion_output_dim=128,
                decoder_hidden_dim=128,
                decoder_num_layers=1,
                dropout=0.0,
                use_copy=False
            )
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            
            # Check model loaded successfully
            self.assertIsNotNone(new_model)
            
        finally:
            Path(temp_path).unlink()


if __name__ == '__main__':
    unittest.main()