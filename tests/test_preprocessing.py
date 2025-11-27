"""
Unit tests for preprocessing utilities
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import (
    preprocess_text,
    normalize_azerbaijani_text,
    tokenize_words,
    is_valid_azerbaijani_char
)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    def test_normalize_azerbaijani_text(self):
        """Test Azerbaijani text normalization."""
        # Test basic normalization
        text = "  Mətn   çox   boşluqludur  "
        expected = "Mətn çox boşluqludur"
        self.assertEqual(normalize_azerbaijani_text(text), expected)
        
        # Test special characters
        text = "Kitab—əla, dəftər...yaxşı"
        normalized = normalize_azerbaijani_text(text)
        self.assertNotIn('—', normalized)
        self.assertNotIn('...', normalized)
    
    def test_preprocess_text(self):
        """Test full preprocessing pipeline."""
        text = "  KİTAB  oxuyuram  "
        processed = preprocess_text(text)

        # Should be trimmed (case is preserved in current implementation)
        self.assertIn("oxuyuram", processed.lower())

        # Test empty string
        self.assertEqual(preprocess_text(""), "")
    
    def test_tokenize_words(self):
        """Test word tokenization."""
        text = "Mən kitab oxuyuram"
        tokens = tokenize_words(text)

        self.assertEqual(len(tokens), 3)
        self.assertIn("Mən", tokens)
        self.assertIn("kitab", tokens)
        self.assertIn("oxuyuram", tokens)

        # Test basic tokenization
        text = "Salam necəsən"
        tokens = tokenize_words(text)
        self.assertGreater(len(tokens), 0)
    
    def test_is_valid_azerbaijani_char(self):
        """Test Azerbaijani character validation."""
        # Valid characters
        self.assertTrue(is_valid_azerbaijani_char('a'))
        self.assertTrue(is_valid_azerbaijani_char('ə'))
        self.assertTrue(is_valid_azerbaijani_char('ş'))
        self.assertTrue(is_valid_azerbaijani_char('ç'))
        self.assertTrue(is_valid_azerbaijani_char('ğ'))
        self.assertTrue(is_valid_azerbaijani_char('ö'))
        self.assertTrue(is_valid_azerbaijani_char('ü'))
        self.assertTrue(is_valid_azerbaijani_char('ı'))
        
        # Invalid characters
        self.assertFalse(is_valid_azerbaijani_char('щ'))  # Cyrillic
        self.assertFalse(is_valid_azerbaijani_char('ж'))  # Cyrillic
        self.assertFalse(is_valid_azerbaijani_char('😀'))  # Emoji
    
    def test_azerbaijani_special_characters(self):
        """Test handling of Azerbaijani-specific characters."""
        text = "şəhər, çörək, güllə, ağac, öyrənmək, ıslıq"
        processed = preprocess_text(text)
        
        # All special characters should be preserved
        self.assertIn('ş', processed)
        self.assertIn('ə', processed)
        self.assertIn('ç', processed)
        self.assertIn('ö', processed)
        self.assertIn('ğ', processed)
        self.assertIn('ü', processed)
        self.assertIn('ı', processed)


class TestVowelHarmony(unittest.TestCase):
    """Test vowel harmony detection (if implemented)."""
    
    def test_front_vowels(self):
        """Test front vowel identification."""
        front_vowels = ['e', 'ə', 'i', 'ö', 'ü']
        for vowel in front_vowels:
            # This would test your vowel harmony logic
            pass
    
    def test_back_vowels(self):
        """Test back vowel identification."""
        back_vowels = ['a', 'ı', 'o', 'u']
        for vowel in back_vowels:
            # This would test your vowel harmony logic
            pass


if __name__ == '__main__':
    unittest.main()