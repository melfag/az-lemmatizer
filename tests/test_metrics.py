"""
Unit tests for evaluation metrics
"""

import unittest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.metrics import (
    accuracy,
    levenshtein_distance,
    average_edit_distance,
    character_level_metrics,
    calculate_all_metrics
)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        predictions = ["kitab", "oxumaq", "gəlmək"]
        targets = ["kitab", "oxumaq", "gəlmək"]
        
        acc = accuracy(predictions, targets)
        self.assertEqual(acc, 100.0)
    
    def test_accuracy_partial(self):
        """Test accuracy with partial correctness."""
        predictions = ["kitab", "oxu", "gəlmək"]
        targets = ["kitab", "oxumaq", "gəlmək"]
        
        acc = accuracy(predictions, targets)
        self.assertAlmostEqual(acc, 66.67, places=2)
    
    def test_accuracy_empty(self):
        """Test accuracy with empty lists."""
        acc = accuracy([], [])
        self.assertEqual(acc, 0.0)
    
    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        # Identical strings
        self.assertEqual(levenshtein_distance("kitab", "kitab"), 0)
        
        # One substitution
        self.assertEqual(levenshtein_distance("kitab", "kitap"), 1)
        
        # One deletion
        self.assertEqual(levenshtein_distance("kitab", "kita"), 1)
        
        # One insertion
        self.assertEqual(levenshtein_distance("kita", "kitab"), 1)
        
        # Multiple operations
        self.assertGreater(levenshtein_distance("kitab", "oxumaq"), 3)
    
    def test_average_edit_distance(self):
        """Test average edit distance."""
        predictions = ["kitab", "oxu", "gəl"]
        targets = ["kitab", "oxumaq", "gəlmək"]
        
        avg_ed = average_edit_distance(predictions, targets)
        
        # kitab vs kitab = 0
        # oxu vs oxumaq = 3
        # gəl vs gəlmək = 2
        # Average = (0 + 3 + 2) / 3 = 1.67
        self.assertAlmostEqual(avg_ed, 1.67, places=2)
    
    def test_character_level_metrics(self):
        """Test character-level precision, recall, F1."""
        predictions = ["kitab", "oxu"]
        targets = ["kitab", "oxumaq"]
        
        metrics = character_level_metrics(predictions, targets)
        
        # Check keys
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # Check values are percentages
        self.assertGreaterEqual(metrics['precision'], 0)
        self.assertLessEqual(metrics['precision'], 100)
        self.assertGreaterEqual(metrics['f1'], 0)
        self.assertLessEqual(metrics['f1'], 100)
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics at once."""
        predictions = ["kitab", "oxu", "gəlmək"]
        targets = ["kitab", "oxumaq", "gəlmək"]
        
        metrics = calculate_all_metrics(predictions, targets)
        
        # Check all expected keys
        expected_keys = ['accuracy', 'avg_edit_distance', 'precision', 'recall', 'f1']
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_metrics_with_azerbaijani_characters(self):
        """Test metrics with Azerbaijani-specific characters."""
        predictions = ["şəhər", "çörək", "güllə"]
        targets = ["şəhər", "çörək", "güllə"]
        
        # Should handle Azerbaijani characters correctly
        acc = accuracy(predictions, targets)
        self.assertEqual(acc, 100.0)
        
        # Edit distance should work with special characters
        dist = levenshtein_distance("şəhər", "şəhər")
        self.assertEqual(dist, 0)


if __name__ == '__main__':
    unittest.main()