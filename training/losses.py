"""
Loss functions for training the lemmatizer
Includes main cross-entropy loss and auxiliary losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.transformation_loss import TransformationLoss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    Prevents the model from becoming overconfident.
    
    Based on Section 4.3.4 of the thesis.
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        """
        Args:
            smoothing: Label smoothing factor (0.1 = 10% smoothing)
            ignore_index: Index to ignore in loss calculation (padding)
        """
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, seq_len, vocab_size) Logits
            targets: (batch_size, seq_len) Target indices
            
        Returns:
            loss: Scalar loss value
        """
        batch_size, seq_len, vocab_size = predictions.size()
        
        # Reshape for loss computation
        predictions = predictions.reshape(-1, vocab_size)  # (batch * seq_len, vocab_size)
        targets = targets.reshape(-1)  # (batch * seq_len)
        
        # Create smoothed labels
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (vocab_size - 1)
        
        # One-hot encoding
        one_hot = torch.zeros_like(predictions).scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        smoothed_labels = one_hot * confidence + (1 - one_hot) * smooth_value
        
        # Compute log probabilities
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Compute loss
        loss = -(smoothed_labels * log_probs).sum(dim=1)
        
        # Mask out padding
        mask = (targets != self.ignore_index).float()
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


class CharacterClassificationLoss(nn.Module):
    """
    Auxiliary loss for character classification.
    Encourages the character encoder to distinguish between stems and affixes.
    
    Based on Section 4.3.4 of the thesis.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Dimension of character encoder hidden states
        """
        super().__init__()
        
        # Binary classifier for each character position
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # Binary: stem vs affix
        )
        
    def forward(
        self,
        char_hidden_states: torch.Tensor,
        is_stem: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            char_hidden_states: (batch_size, seq_len, hidden_dim)
            is_stem: (batch_size, seq_len) Binary labels (1=stem, 0=affix)
            
        Returns:
            loss: Scalar loss value
        """
        batch_size, seq_len, hidden_dim = char_hidden_states.size()
        
        # Classify each position
        logits = self.classifier(char_hidden_states)  # (batch, seq_len, 2)
        
        # Reshape for loss
        logits = logits.reshape(-1, 2)  # (batch * seq_len, 2)
        targets = is_stem.reshape(-1)  # (batch * seq_len)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets.long(), reduction='mean')
        
        return loss


class AttentionCoverageLoss(nn.Module):
    """
    Auxiliary loss to encourage attention coverage.
    Prevents the model from repeatedly focusing on the same positions.
    
    Based on Section 4.3.4 of the thesis.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_weights: (batch_size, tgt_seq_len, src_seq_len)
            
        Returns:
            loss: Scalar loss value
        """
        # Sum attention over target positions
        coverage = attention_weights.sum(dim=1)  # (batch, src_seq_len)
        
        # Penalize deviations from uniform coverage
        # Ideal coverage would be close to tgt_seq_len / src_seq_len for each position
        target_coverage = attention_weights.size(1) / attention_weights.size(2)
        
        # L2 penalty
        loss = torch.mean((coverage - target_coverage) ** 2)
        
        return loss


class CompositeLoss(nn.Module):
    """
    Combined loss function with main and auxiliary losses.
    
    Based on Section 4.3.4 of the thesis.
    """
    
    def __init__(
        self,
        char_hidden_dim: int,
        smoothing: float = 0.1,
        char_classification_weight: float = 0.1,
        attention_coverage_weight: float = 0.05,
        transformation_weight: float = 0.0
    ):
        """
        Args:
            char_hidden_dim: Dimension of character encoder hidden states
            smoothing: Label smoothing factor
            char_classification_weight: Weight for character classification loss
            attention_coverage_weight: Weight for attention coverage loss
            transformation_weight: Weight for transformation loss (encourages transformations)
        """
        super().__init__()

        self.char_classification_weight = char_classification_weight
        self.attention_coverage_weight = attention_coverage_weight
        self.transformation_weight = transformation_weight

        # Main loss
        self.main_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)

        # Auxiliary losses
        self.char_classification_loss = CharacterClassificationLoss(char_hidden_dim)
        self.attention_coverage_loss = AttentionCoverageLoss()

        # Transformation loss (NEW)
        if transformation_weight > 0:
            self.transformation_loss = TransformationLoss(weight=transformation_weight)
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        char_hidden_states: torch.Tensor = None,
        is_stem: torch.Tensor = None,
        attention_weights: torch.Tensor = None,
        inputs: torch.Tensor = None
    ) -> dict:
        """
        Args:
            predictions: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
            char_hidden_states: (batch_size, src_seq_len, hidden_dim) - optional
            is_stem: (batch_size, src_seq_len) - optional
            attention_weights: (batch_size, tgt_seq_len, src_seq_len) - optional
            inputs: (batch_size, seq_len) Input character IDs - optional (for transformation loss)

        Returns:
            Dictionary with total loss and individual components
        """
        # Main loss
        main_loss = self.main_loss(predictions, targets)

        total_loss = main_loss
        losses = {'main_loss': main_loss.item()}

        # Character classification loss
        if char_hidden_states is not None and is_stem is not None:
            char_class_loss = self.char_classification_loss(char_hidden_states, is_stem)
            total_loss = total_loss + self.char_classification_weight * char_class_loss
            losses['char_classification_loss'] = char_class_loss.item()

        # Attention coverage loss
        if attention_weights is not None:
            coverage_loss = self.attention_coverage_loss(attention_weights)
            total_loss = total_loss + self.attention_coverage_weight * coverage_loss
            losses['attention_coverage_loss'] = coverage_loss.item()

        # Transformation loss (NEW)
        if self.transformation_weight > 0 and inputs is not None:
            transform_loss = self.transformation_loss(predictions, targets, inputs)
            total_loss = total_loss + transform_loss
            losses['transformation_loss'] = transform_loss.item()
        else:
            losses['transformation_loss'] = 0.0

        losses['total_loss'] = total_loss.item()

        return total_loss, losses