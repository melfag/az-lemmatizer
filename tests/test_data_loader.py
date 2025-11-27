"""
Unit tests for data loader
"""

import unittest
import tempfile
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_loader import LemmatizationDataset, create_dataloader
from utils.vocabulary import CharacterVocabulary


class TestLemmatizationDataset(unittest.TestCase):
    """Test lemmatization dataset."""
    
    def setUp(self):
        """Set up test data and vocabulary."""
        # Create test data
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
            },
            {
                "input_word": "gəldim",
                "lemma": "gəlmək",
                "context": "Mən evə gəldim",
                "pos": "VERB"
            }
        ]
        
        # Create vocabulary
        self.vocab = CharacterVocabulary()
        texts = [item['input_word'] for item in self.test_data]
        texts += [item['lemma'] for item in self.test_data]
        self.vocab.build_from_texts(texts)
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False
        )
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary file."""
        Path(self.temp_file.name).unlink()
    
    def test_dataset_length(self):
        """Test dataset length."""
        dataset = LemmatizationDataset(
            data_path=self.temp_file.name,
            vocab=self.vocab
        )
        
        self.assertEqual(len(dataset), len(self.test_data))
    
    def test_dataset_getitem(self):
        """Test getting item from dataset."""
        dataset = LemmatizationDataset(
            data_path=self.temp_file.name,
            vocab=self.vocab
        )
        
        item = dataset[0]
        
        # Check keys
        self.assertIn('input_char_ids', item)
        self.assertIn('target_char_ids', item)
        self.assertIn('input_length', item)
        self.assertIn('context', item)
        self.assertIn('input_word', item)
        
        # Check types
        self.assertIsInstance(item['input_char_ids'], list)
        self.assertIsInstance(item['target_char_ids'], list)
        self.assertIsInstance(item['input_length'], int)
        self.assertIsInstance(item['context'], str)
    
    def test_create_dataloader(self):
        """Test dataloader creation."""
        dataloader = create_dataloader(
            data_path=self.temp_file.name,
            vocab=self.vocab,
            batch_size=2,
            shuffle=False
        )
        
        # Get first batch
        batch = next(iter(dataloader))
        
        # Check batch structure
        self.assertIn('input_char_ids', batch)
        self.assertIn('target_char_ids', batch)
        self.assertIn('input_lengths', batch)
        self.assertIn('context_sentences', batch)
        
        # Check batch size
        self.assertEqual(batch['input_char_ids'].size(0), 2)
        self.assertEqual(len(batch['context_sentences']), 2)


if __name__ == '__main__':
    unittest.main()