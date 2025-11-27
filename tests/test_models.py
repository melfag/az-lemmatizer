"""
Unit tests for model components
"""

import unittest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.components.attention import BahdanauAttention
from models.components.copy_mechanism import CopyMechanism
from models.components.vowel_harmony import VowelHarmonyAwareness
from models.components.morphological_features import MorphologicalFeatureExtractor
from models.character_encoder import CharacterEncoder
from models.fusion_mechanism import FusionMechanism
from models.decoder import Decoder


class TestAttention(unittest.TestCase):
    """Test attention mechanism."""
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        batch_size = 4
        seq_len = 10
        encoder_hidden_dim = 256
        decoder_hidden_dim = 512
        
        attention = BahdanauAttention(
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim
        )
        
        # Create dummy inputs
        decoder_hidden = torch.randn(batch_size, decoder_hidden_dim)
        encoder_outputs = torch.randn(batch_size, seq_len, encoder_hidden_dim)
        
        # Forward pass
        context, weights = attention(decoder_hidden, encoder_outputs)
        
        # Check shapes
        self.assertEqual(context.shape, (batch_size, encoder_hidden_dim))
        self.assertEqual(weights.shape, (batch_size, seq_len))
        
        # Check attention weights sum to 1
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(batch_size)))


class TestCopyMechanism(unittest.TestCase):
    """Test copy mechanism."""
    
    def test_copy_mechanism_forward(self):
        """Test copy mechanism forward pass."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        encoder_hidden_dim = 256
        decoder_hidden_dim = 512
        
        copy_mech = CopyMechanism(
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            vocab_size=vocab_size
        )
        
        # Create dummy inputs
        decoder_hidden = torch.randn(batch_size, decoder_hidden_dim)
        encoder_outputs = torch.randn(batch_size, seq_len, encoder_hidden_dim)
        attention_weights = torch.softmax(torch.randn(batch_size, seq_len), dim=1)
        vocab_distribution = torch.randn(batch_size, vocab_size)
        input_char_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        output = copy_mech(
            decoder_hidden=decoder_hidden,
            encoder_hidden_states=encoder_outputs,
            attention_weights=attention_weights,
            vocab_distribution=vocab_distribution,
            input_char_ids=input_char_ids
        )
        
        # Check shape
        self.assertEqual(output.shape, (batch_size, vocab_size))


class TestVowelHarmony(unittest.TestCase):
    """Test vowel harmony awareness."""
    
    def test_vowel_harmony_forward(self):
        """Test vowel harmony forward pass."""
        batch_size = 4
        seq_len = 10
        char_embedding_dim = 128
        hidden_dim = 512
        
        vowel_harmony = VowelHarmonyAwareness(
            char_embedding_dim=char_embedding_dim,
            hidden_dim=hidden_dim
        )
        
        # Create dummy inputs
        char_embeddings = torch.randn(batch_size, seq_len, char_embedding_dim)
        fused_representation = torch.randn(batch_size, hidden_dim)
        
        # Forward pass
        output = vowel_harmony(char_embeddings, fused_representation)
        
        # Check shape
        self.assertEqual(output.shape, (batch_size, hidden_dim))


class TestMorphologicalFeatures(unittest.TestCase):
    """Test morphological feature extractor."""
    
    def test_morphological_features_forward(self):
        """Test morphological feature extractor forward pass."""
        batch_size = 4
        seq_len = 10
        char_embedding_dim = 128
        feature_dim = 128
        
        extractor = MorphologicalFeatureExtractor(
            char_embedding_dim=char_embedding_dim,
            feature_dim=feature_dim
        )
        
        # Create dummy inputs
        char_embeddings = torch.randn(batch_size, seq_len, char_embedding_dim)
        
        # Forward pass
        features = extractor(char_embeddings)
        
        # Check shape (4 filter sizes, each producing feature_dim features)
        self.assertEqual(features.shape, (batch_size, feature_dim * 4))


class TestCharacterEncoder(unittest.TestCase):
    """Test character encoder."""
    
    def test_character_encoder_forward(self):
        """Test character encoder forward pass."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        char_embedding_dim = 128
        hidden_dim = 256
        
        encoder = CharacterEncoder(
            vocab_size=vocab_size,
            char_embedding_dim=char_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.3
        )
        
        # Create dummy inputs
        char_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        lengths = torch.tensor([seq_len] * batch_size)
        
        # Forward pass
        hidden_states, final_representation = encoder(char_ids, lengths)
        
        # Check shapes
        self.assertEqual(hidden_states.shape, (batch_size, seq_len, hidden_dim * 2))
        self.assertEqual(final_representation.shape, (batch_size, hidden_dim * 2))


class TestFusionMechanism(unittest.TestCase):
    """Test fusion mechanism."""
    
    def test_fusion_forward(self):
        """Test fusion mechanism forward pass."""
        batch_size = 4
        char_hidden_dim = 512
        contextual_hidden_dim = 768
        output_dim = 512
        
        fusion = FusionMechanism(
            char_hidden_dim=char_hidden_dim,
            contextual_hidden_dim=contextual_hidden_dim,
            output_dim=output_dim
        )
        
        # Create dummy inputs
        char_representation = torch.randn(batch_size, char_hidden_dim)
        contextual_representation = torch.randn(batch_size, contextual_hidden_dim)
        
        # Forward pass
        fused = fusion(char_representation, contextual_representation)
        
        # Check shape
        self.assertEqual(fused.shape, (batch_size, output_dim))


class TestDecoder(unittest.TestCase):
    """Test decoder."""
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        batch_size = 4
        src_seq_len = 10
        tgt_seq_len = 8
        vocab_size = 100
        char_embedding_dim = 128
        encoder_hidden_dim = 512
        decoder_hidden_dim = 512
        
        decoder = Decoder(
            vocab_size=vocab_size,
            char_embedding_dim=char_embedding_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            use_copy=False  # Disable copy for simplicity
        )
        
        # Create dummy inputs
        encoder_hidden_states = torch.randn(batch_size, src_seq_len, encoder_hidden_dim)
        encoder_final_state = torch.randn(batch_size, encoder_hidden_dim)
        target_char_ids = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
        
        # Forward pass
        outputs, attentions = decoder(
            encoder_hidden_states=encoder_hidden_states,
            encoder_final_state=encoder_final_state,
            target_char_ids=target_char_ids,
            teacher_forcing_ratio=0.5,
            max_length=tgt_seq_len
        )
        
        # Check shapes
        self.assertEqual(outputs.shape, (batch_size, tgt_seq_len, vocab_size))
        self.assertEqual(attentions.shape, (batch_size, tgt_seq_len, src_seq_len))


if __name__ == '__main__':
    unittest.main()