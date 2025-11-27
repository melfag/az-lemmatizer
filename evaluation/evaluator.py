"""
Main evaluator for the lemmatizer model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
from pathlib import Path

from utils.metrics import (
    LemmatizationMetrics,
    AmbiguityMetrics,
    MorphologicalComplexityMetrics
)


class Evaluator:
    """
    Comprehensive evaluator for lemmatization model.
    Based on Chapter 5 (Experimental Setup) of the thesis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        vocab,
        device: torch.device,
        output_dir: str = "evaluation_results"
    ):
        """
        Args:
            model: The trained lemmatizer model
            vocab: Character vocabulary
            device: Device to run evaluation on
            output_dir: Directory to save evaluation results
        """
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model to eval mode
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: DataLoader,
        dataset_name: str = "test",
        save_predictions: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            dataset_name: Name of dataset (for logging)
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating on {dataset_name} set...")
        print(f"Total samples: {len(dataloader.dataset)}")
        
        # Initialize metrics
        metrics = LemmatizationMetrics(self.vocab)
        
        # Storage for predictions
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                # Move to device
                input_char_ids = batch['input_char_ids'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                target_char_ids = batch['target_char_ids'].to(self.device)
                context_sentences = batch['context_sentences']
                target_positions_tensor = batch['target_positions']

                # Convert target_positions from DataLoader format to list of tuples
                if isinstance(target_positions_tensor, list) and len(target_positions_tensor) == 2:
                    starts = target_positions_tensor[0]
                    ends = target_positions_tensor[1]
                    target_positions = [(int(starts[i]), int(ends[i])) for i in range(len(starts))]
                elif hasattr(target_positions_tensor, 'shape') and len(target_positions_tensor.shape) == 2:
                    target_positions = [(int(target_positions_tensor[i][0]), int(target_positions_tensor[i][1]))
                                       for i in range(target_positions_tensor.shape[0])]
                else:
                    target_positions = target_positions_tensor

                # Get predictions
                predictions, _ = self.model.lemmatize(
                    input_char_ids=input_char_ids,
                    input_lengths=input_lengths,
                    context_sentences=context_sentences,
                    target_word_positions=target_positions,
                    max_length=target_char_ids.size(1)
                )

                # Update metrics
                metrics.update(predictions, target_char_ids)

                # Store for analysis
                if save_predictions:
                    for i in range(predictions.size(0)):
                        all_predictions.append(self._decode(predictions[i]))
                        all_targets.append(self._decode(target_char_ids[i]))
                        all_inputs.append(batch.get('word_str', [''] * len(context_sentences))[i])
        
        # Compute metrics
        results = metrics.compute()
        
        # Print summary
        print(metrics.get_summary())
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(
                all_inputs,
                all_predictions,
                all_targets,
                dataset_name
            )
        
        # Save metrics
        self._save_metrics(results, dataset_name)
        
        return results
    
    def evaluate_ambiguity(
        self,
        dataloader: DataLoader,
        dataset_name: str = "ambiguity_test"
    ) -> Dict[str, float]:
        """
        Evaluate ambiguity resolution.
        
        Args:
            dataloader: Ambiguity test data loader
            dataset_name: Name for logging
            
        Returns:
            Ambiguity resolution metrics
        """
        print(f"\nEvaluating ambiguity resolution on {dataset_name}...")
        
        ambiguity_metrics = AmbiguityMetrics(self.vocab)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Ambiguity test"):
                input_char_ids = batch['input_char_ids'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                target_char_ids = batch['target_char_ids'].to(self.device)
                context_sentences = batch['context_sentences']
                target_positions = batch['target_positions']
                word_forms = batch.get('input_words', [])
                
                predictions, _ = self.model.lemmatize(
                    input_char_ids=input_char_ids,
                    input_lengths=input_lengths,
                    context_sentences=context_sentences,
                    target_word_positions=target_positions
                )
                
                ambiguity_metrics.update(predictions, target_char_ids, word_forms)
        
        results = ambiguity_metrics.compute()
        
        print(f"\nAmbiguity Resolution Accuracy: {results['ambiguity_resolution_accuracy']:.2f}%")
        print(f"Total ambiguous forms: {results['total_ambiguous']}")
        print(f"Correctly resolved: {results['correct_ambiguous']}")
        
        # Get per-word statistics
        per_word_stats = ambiguity_metrics.get_per_word_stats()
        
        # Save results
        self._save_metrics(results, f"{dataset_name}_overall")
        self._save_json(per_word_stats, f"{dataset_name}_per_word.json")
        
        return results
    
    def evaluate_morphological_complexity(
        self,
        dataloader: DataLoader,
        dataset_name: str = "morphological_complexity_test"
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate by morphological complexity.
        
        Args:
            dataloader: Complexity test data loader
            dataset_name: Name for logging
            
        Returns:
            Metrics per complexity level
        """
        print(f"\nEvaluating morphological complexity on {dataset_name}...")
        
        complexity_metrics = MorphologicalComplexityMetrics(self.vocab)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Complexity test"):
                input_char_ids = batch['input_char_ids'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                target_char_ids = batch['target_char_ids'].to(self.device)
                context_sentences = batch['context_sentences']
                target_positions = batch['target_positions']
                complexity_levels = batch.get('complexity', [1] * len(context_sentences))
                
                predictions, _ = self.model.lemmatize(
                    input_char_ids=input_char_ids,
                    input_lengths=input_lengths,
                    context_sentences=context_sentences,
                    target_word_positions=target_positions
                )
                
                complexity_metrics.update(predictions, target_char_ids, complexity_levels)
        
        results = complexity_metrics.compute()
        
        print(complexity_metrics.get_summary())
        
        # Save results
        self._save_json(results, f"{dataset_name}_results.json")
        
        return results
    
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
    
    def _save_predictions(
        self,
        inputs: List[str],
        predictions: List[str],
        targets: List[str],
        dataset_name: str
    ):
        """Save predictions to file."""
        results = []
        for inp, pred, tgt in zip(inputs, predictions, targets):
            results.append({
                'input': inp,
                'predicted': pred,
                'target': tgt,
                'correct': pred == tgt
            })
        
        filepath = self.output_dir / f"{dataset_name}_predictions.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Predictions saved to: {filepath}")
    
    def _save_metrics(self, metrics: Dict, dataset_name: str):
        """Save metrics to file."""
        filepath = self.output_dir / f"{dataset_name}_metrics.json"
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")
    
    def _save_json(self, data: Dict, filename: str):
        """Save generic JSON data."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {filepath}")