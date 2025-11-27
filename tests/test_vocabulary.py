"""
Unit tests for vocabulary
"""

import unittest
import tempfile
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.vocabulary import CharacterVocabulary


class TestCharacterVocabulary(unittest.TestCase):
    """Test character vocabulary."""
    
    def setUp(self):
        """Set up test vocabulary."""
        self.texts = [
            "kitab",
            "oxumaq",
            "gəlmək",
            "yazmaq"
        ]
        self.vocab = CharacterVocabulary()
        self.vocab.build_from_texts(self.texts)
    
    def test_special_tokens(self):
        """Test special tokens are present."""
        self.assertIn('<PAD>', self.vocab.char_to_idx)
        self.assertIn('<UNK>', self.vocab.char_to_idx)
        self.assertIn('<START>', self.vocab.char_to_idx)
        self.assertIn('<END>', self.vocab.char_to_idx)
        
        # Check indices
        self.assertEqual(self.vocab.pad_idx, 0)
        self.assertEqual(self.vocab.unk_idx, 1)
        self.assertEqual(self.vocab.start_idx, 2)
        self.assertEqual(self.vocab.end_idx, 3)
    
    def test_character_mapping(self):
        """Test character to index mapping."""
        # Common characters should be in vocabulary
        for char in 'kitab':
            self.assertIn(char, self.vocab.char_to_idx)
        
        # Azerbaijani-specific characters
        self.assertIn('ə', self.vocab.char_to_idx)
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        word = "kitab"
        
        # Encode
        char_ids = self.vocab.encode(word)
        
        # Should start with START token and end with END token
        self.assertEqual(char_ids[0], self.vocab.start_idx)
        self.assertEqual(char_ids[-1], self.vocab.end_idx)
        
        # Decode (without special tokens)
        decoded = self.vocab.decode(char_ids)
        self.assertEqual(decoded, word)
    
    def test_unknown_character(self):
        """Test handling of unknown characters."""
        # Character not in vocabulary
        char_id = self.vocab.char_to_idx.get('щ', self.vocab.unk_idx)
        self.assertEqual(char_id, self.vocab.unk_idx)
    
    def test_vocabulary_size(self):
        """Test vocabulary size."""
        # Should include special tokens + unique characters
        self.assertGreater(len(self.vocab), 4)  # At least 4 special tokens
    
    def test_save_load(self):
        """Test saving and loading vocabulary."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            self.vocab.save(temp_path)
            
            # Load
            loaded_vocab = CharacterVocabulary.load(temp_path)
            
            # Check equality
            self.assertEqual(len(self.vocab), len(loaded_vocab))
            self.assertEqual(self.vocab.char_to_idx, loaded_vocab.char_to_idx)
            self.assertEqual(self.vocab.idx2char, loaded_vocab.idx2char)
            
        finally:
            # Cleanup
            Path(temp_path).unlink()
    
    def test_padding(self):
        """Test sequence padding."""
        word = "ab"
        char_ids = self.vocab.encode(word, max_length=10)
        
        # Should be padded to max_length
        self.assertEqual(len(char_ids), 10)
        
        # Padding should be at the end
        self.assertEqual(char_ids[-1], self.vocab.pad_idx)


if __name__ == '__main__':
    unittest.main()