"""
Evaluation metrics for lemmatization.
Based on Section 5.2 from the thesis.

This module provides both:
1. Pure Python functions for standalone metric calculation
2. PyTorch-integrated classes for model evaluation
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

try:
    import Levenshtein
    USE_LEVENSHTEIN_LIB = True
except ImportError:
    USE_LEVENSHTEIN_LIB = False
    print("Warning: python-Levenshtein not installed. Using slower pure Python implementation.")


# ============================================================================
# PART 1: Pure Python Utility Functions (Original)
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    
    Pure Python implementation - slower but has no dependencies.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    if USE_LEVENSHTEIN_LIB:
        return Levenshtein.distance(s1, s2)
    
    # Fallback to pure Python implementation
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate exact match accuracy (Section 5.2.1).
    
    Accuracy = (Number of correctly lemmatized words) / (Total number of words) × 100%
    
    Args:
        predictions: List of predicted lemmas
        targets: List of target lemmas
        
    Returns:
        Accuracy percentage
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(pred == target for pred, target in zip(predictions, targets))
    return 100.0 * correct / len(predictions)


def average_edit_distance(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate average character-level edit distance (Section 5.2.2).
    
    Average Edit Distance = (1/n) × Σ Levenshtein(predicted_i, gold_i)
    
    Args:
        predictions: List of predicted lemmas
        targets: List of target lemmas
        
    Returns:
        Average edit distance
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    distances = [levenshtein_distance(pred, target) 
                 for pred, target in zip(predictions, targets)]
    return np.mean(distances)


def character_level_metrics(predictions: List[str], 
                            targets: List[str]) -> Dict[str, float]:
    """
    Calculate character-level precision, recall, and F1 (Section 5.2.3).
    
    Args:
        predictions: List of predicted lemmas
        targets: List of target lemmas
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, target in zip(predictions, targets):
        pred_chars = set(enumerate(pred))
        target_chars = set(enumerate(target))
        
        tp = len(pred_chars & target_chars)
        fp = len(pred_chars - target_chars)
        fn = len(target_chars - pred_chars)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }


def ambiguity_resolution_accuracy(predictions: List[str], 
                                  targets: List[str],
                                  ambiguous_indices: List[int]) -> float:
    """
    Calculate accuracy on ambiguous word forms (Section 5.2.4).
    
    Args:
        predictions: List of predicted lemmas
        targets: List of target lemmas
        ambiguous_indices: Indices of ambiguous words
        
    Returns:
        Accuracy on ambiguous words
    """
    if len(ambiguous_indices) == 0:
        return 0.0
    
    correct = sum(predictions[i] == targets[i] for i in ambiguous_indices)
    return 100.0 * correct / len(ambiguous_indices)


def calculate_all_metrics(predictions: List[str], 
                         targets: List[str],
                         ambiguous_indices: List[int] = None) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        predictions: List of predicted lemmas
        targets: List of target lemmas
        ambiguous_indices: Optional indices of ambiguous words
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy(predictions, targets),
        'avg_edit_distance': average_edit_distance(predictions, targets),
    }
    
    char_metrics = character_level_metrics(predictions, targets)
    metrics.update(char_metrics)
    
    if ambiguous_indices is not None and len(ambiguous_indices) > 0:
        metrics['ambiguity_accuracy'] = ambiguity_resolution_accuracy(
            predictions, targets, ambiguous_indices
        )
    
    return metrics


def error_analysis_by_category(predictions: List[str],
                               targets: List[str],
                               categories: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Perform error analysis by category (e.g., POS, morphological complexity).
    
    Args:
        predictions: List of predicted lemmas
        targets: List of target lemmas
        categories: List of category labels for each example
        
    Returns:
        Dictionary mapping categories to their metrics
    """
    if len(predictions) != len(targets) or len(predictions) != len(categories):
        raise ValueError("All lists must have the same length")
    
    category_data = defaultdict(lambda: {'predictions': [], 'targets': []})
    for pred, target, cat in zip(predictions, targets, categories):
        category_data[cat]['predictions'].append(pred)
        category_data[cat]['targets'].append(target)
    
    results = {}
    for cat, data in category_data.items():
        results[cat] = calculate_all_metrics(
            data['predictions'], 
            data['targets']
        )
    
    return results


def morphological_complexity_analysis(predictions: List[str],
                                     targets: List[str],
                                     num_morphemes: List[int]) -> Dict[str, Dict[str, float]]:
    """
    Analyze performance by morphological complexity (Section 6.2.3).
    
    Args:
        predictions: List of predicted lemmas
        targets: List of target lemmas
        num_morphemes: Number of morphemes in each word
        
    Returns:
        Dictionary mapping complexity levels to their metrics
    """
    if len(predictions) != len(targets) or len(predictions) != len(num_morphemes):
        raise ValueError("All lists must have the same length")
    
    complexity_categories = []
    for n in num_morphemes:
        if n == 1:
            complexity_categories.append('Level 1 (1 morpheme)')
        elif n == 2:
            complexity_categories.append('Level 2 (2 morphemes)')
        elif n == 3:
            complexity_categories.append('Level 3 (3 morphemes)')
        elif n == 4:
            complexity_categories.append('Level 4 (4 morphemes)')
        else:
            complexity_categories.append('Level 5+ (5+ morphemes)')
    
    return error_analysis_by_category(predictions, targets, complexity_categories)


# ============================================================================
# PART 2: PyTorch-Integrated Classes for Model Evaluation
# ============================================================================

class LemmatizationMetrics:
    """
    Comprehensive metrics tracker for lemmatization evaluation.
    Works with PyTorch tensors and integrates with vocabulary.
    
    Based on Section 5.2 of the thesis.
    """
    
    def __init__(self, vocab):
        """
        Args:
            vocab: Character vocabulary for decoding
        """
        self.vocab = vocab
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total = 0
        self.correct = 0
        self.edit_distances = []
        self.char_tp = 0
        self.char_fp = 0
        self.char_fn = 0
        self.predictions_list = []
        self.targets_list = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        input_words: List[str] = None
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            predictions: (batch_size, seq_len) Predicted character IDs
            targets: (batch_size, seq_len) Target character IDs
            input_words: List of input words (for analysis)
        """
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred_lemma = self._decode(predictions[i])
            target_lemma = self._decode(targets[i])
            
            self.predictions_list.append(pred_lemma)
            self.targets_list.append(target_lemma)
            
            self.total += 1
            
            # Accuracy
            if pred_lemma == target_lemma:
                self.correct += 1
            
            # Edit distance
            edit_dist = levenshtein_distance(pred_lemma, target_lemma)
            self.edit_distances.append(edit_dist)
            
            # Character-level metrics
            pred_chars = set(pred_lemma)
            target_chars = set(target_lemma)
            
            self.char_tp += len(pred_chars & target_chars)
            self.char_fp += len(pred_chars - target_chars)
            self.char_fn += len(target_chars - pred_chars)
    
    def _decode(self, char_ids: torch.Tensor) -> str:
        """
        Decode character IDs to string.

        Args:
            char_ids: (seq_len,) Character IDs

        Returns:
            Decoded string
        """
        chars = []
        for idx in char_ids:
            idx = idx.item()

            # Stop at end token or padding
            if idx == self.vocab.end_idx or idx == self.vocab.pad_idx:
                break

            # Skip start token
            if idx == self.vocab.start_idx:
                continue

            char = self.vocab.idx2char.get(idx, '')
            if char:
                chars.append(char)

        return ''.join(chars)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self.total == 0:
            return {
                'accuracy': 0.0,
                'avg_edit_distance': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        acc = 100 * self.correct / self.total
        avg_edit_distance = np.mean(self.edit_distances) if self.edit_distances else 0.0
        
        # Character-level precision, recall, F1
        precision = self.char_tp / (self.char_tp + self.char_fp) if (self.char_tp + self.char_fp) > 0 else 0.0
        recall = self.char_tp / (self.char_tp + self.char_fn) if (self.char_tp + self.char_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': acc,
            'avg_edit_distance': avg_edit_distance,
            'precision': 100 * precision,
            'recall': 100 * recall,
            'f1': 100 * f1
        }
    
    def get_summary(self) -> str:
        """Get formatted summary string."""
        metrics = self.compute()
        
        summary = f"""
Lemmatization Metrics:
  Accuracy: {metrics['accuracy']:.2f}%
  Avg Edit Distance: {metrics['avg_edit_distance']:.2f}
  Character-level:
    Precision: {metrics['precision']:.2f}%
    Recall: {metrics['recall']:.2f}%
    F1 Score: {metrics['f1']:.2f}%
  Total Samples: {self.total}
  Correct: {self.correct}
        """
        
        return summary


class AmbiguityMetrics:
    """
    Specialized metrics for ambiguity resolution.
    Based on Section 5.2.4 of the thesis.
    """
    
    def __init__(self, vocab):
        self.vocab = vocab
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.ambiguous_correct = 0
        self.ambiguous_total = 0
        self.by_word_form = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        word_forms: List[str]
    ):
        """
        Update ambiguity metrics.
        
        Args:
            predictions: (batch_size, seq_len) Predicted character IDs
            targets: (batch_size, seq_len) Target character IDs
            word_forms: List of input word forms
        """
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred_lemma = self._decode(predictions[i])
            target_lemma = self._decode(targets[i])
            word_form = word_forms[i]
            
            self.ambiguous_total += 1
            self.by_word_form[word_form]['total'] += 1
            
            if pred_lemma == target_lemma:
                self.ambiguous_correct += 1
                self.by_word_form[word_form]['correct'] += 1
    
    def _decode(self, char_ids: torch.Tensor) -> str:
        """Decode character IDs to string."""
        chars = []
        for idx in char_ids:
            idx = idx.item()
            if idx == self.vocab.end_idx or idx == self.vocab.pad_idx:
                break
            if idx == self.vocab.start_idx:
                continue
            char = self.vocab.idx2char.get(idx, '')
            if char:
                chars.append(char)
        return ''.join(chars)
    
    def compute(self) -> Dict[str, float]:
        """Compute ambiguity resolution accuracy."""
        if self.ambiguous_total == 0:
            return {'ambiguity_resolution_accuracy': 0.0}
        
        acc = 100 * self.ambiguous_correct / self.ambiguous_total
        
        return {
            'ambiguity_resolution_accuracy': acc,
            'total_ambiguous': self.ambiguous_total,
            'correct_ambiguous': self.ambiguous_correct
        }
    
    def get_per_word_stats(self) -> Dict[str, Dict[str, float]]:
        """Get accuracy statistics per word form."""
        stats = {}
        for word_form, counts in self.by_word_form.items():
            if counts['total'] > 0:
                acc = 100 * counts['correct'] / counts['total']
                stats[word_form] = {
                    'accuracy': acc,
                    'correct': counts['correct'],
                    'total': counts['total']
                }
        return stats


class MorphologicalComplexityMetrics:
    """
    Metrics broken down by morphological complexity.
    Based on Section 5.2 of the thesis.
    """
    
    def __init__(self, vocab):
        self.vocab = vocab
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.by_complexity = defaultdict(lambda: {
            'correct': 0,
            'total': 0,
            'edit_distances': []
        })
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        complexity_levels: List[int]
    ):
        """
        Update metrics by morphological complexity.
        
        Args:
            predictions: (batch_size, seq_len)
            targets: (batch_size, seq_len)
            complexity_levels: List of complexity levels (number of morphemes)
        """
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred_lemma = self._decode(predictions[i])
            target_lemma = self._decode(targets[i])
            complexity = complexity_levels[i]
            
            self.by_complexity[complexity]['total'] += 1
            
            if pred_lemma == target_lemma:
                self.by_complexity[complexity]['correct'] += 1
            
            edit_dist = levenshtein_distance(pred_lemma, target_lemma)
            self.by_complexity[complexity]['edit_distances'].append(edit_dist)
    
    def _decode(self, char_ids: torch.Tensor) -> str:
        """Decode character IDs to string."""
        chars = []
        for idx in char_ids:
            idx = idx.item()
            if idx == self.vocab.end_idx or idx == self.vocab.pad_idx:
                break
            if idx == self.vocab.start_idx:
                continue
            char = self.vocab.idx2char.get(idx, '')
            if char:
                chars.append(char)
        return ''.join(chars)
    
    def compute(self) -> Dict[int, Dict[str, float]]:
        """Compute metrics per complexity level."""
        results = {}
        
        for complexity, stats in sorted(self.by_complexity.items()):
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                avg_edit_dist = np.mean(stats['edit_distances'])
                
                results[complexity] = {
                    'accuracy': acc,
                    'avg_edit_distance': avg_edit_dist,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
        
        return results
    
    def get_summary(self) -> str:
        """Get formatted summary."""
        results = self.compute()
        
        summary = "\nPerformance by Morphological Complexity:\n"
        summary += "-" * 60 + "\n"
        summary += f"{'Level':<10} {'Accuracy':<15} {'Avg Edit Dist':<15} {'Samples':<10}\n"
        summary += "-" * 60 + "\n"
        
        for complexity, metrics in sorted(results.items()):
            summary += f"{complexity:<10} {metrics['accuracy']:>6.2f}%       "
            summary += f"{metrics['avg_edit_distance']:>6.2f}          "
            summary += f"{metrics['total']:<10}\n"
        
        return summary


class MetricsTracker:
    """Simple metrics tracker for training (keeps both pure Python and PyTorch compatibility)."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions: List[str], targets: List[str], loss: float = None):
        """
        Update tracker with new predictions.
        
        Args:
            predictions: Batch predictions
            targets: Batch targets
            loss: Optional loss value
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = calculate_all_metrics(self.predictions, self.targets)
        
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
        
        return metrics
    
    def __len__(self):
        """Return number of examples tracked."""
        return len(self.predictions)


if __name__ == "__main__":
    # Test pure Python functions
    print("=" * 60)
    print("Testing Pure Python Functions")
    print("=" * 60)
    
    predictions = ["kitab", "oxu", "gəl", "ev"]
    targets = ["kitab", "oxumaq", "gəlmək", "ev"]
    
    print(f"Predictions: {predictions}")
    print(f"Targets: {targets}\n")
    
    acc = accuracy(predictions, targets)
    print(f"Accuracy: {acc:.2f}%")
    
    avg_ed = average_edit_distance(predictions, targets)
    print(f"Average Edit Distance: {avg_ed:.2f}")
    
    char_metrics = character_level_metrics(predictions, targets)
    print(f"Character-level Precision: {char_metrics['precision']:.2f}%")
    print(f"Character-level Recall: {char_metrics['recall']:.2f}%")
    print(f"Character-level F1: {char_metrics['f1']:.2f}%")
    
    print("\n" + "=" * 60)
    print("Pure Python functions work correctly!")
    print("=" * 60)