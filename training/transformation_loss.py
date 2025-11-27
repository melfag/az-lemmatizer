"""
Transformation loss - encourage model to transform inputs
"""
import torch
import torch.nn as nn


class TransformationLoss(nn.Module):
    """
    Penalize identity predictions when transformation is needed
    """

    def __init__(self, weight=0.1):
        """
        Args:
            weight: Loss weight (higher = more transformation encouraged)
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute transformation loss

        Args:
            predictions: (batch, seq_len, vocab_size) Model outputs
            targets: (batch, seq_len) Target character IDs
            inputs: (batch, seq_len) Input character IDs

        Returns:
            Transformation loss
        """
        # Get predicted character IDs
        pred_ids = predictions.argmax(dim=-1)

        # Check if prediction matches input (identity)
        is_identity = (pred_ids == inputs).float()

        # Check if target differs from input (requires transformation)
        needs_transform = (targets != inputs).float()

        # Penalty for identity when transformation needed
        # penalty = is_identity * needs_transform
        # Average over sequence and batch
        penalty = (is_identity * needs_transform).mean()

        return self.weight * penalty


if __name__ == '__main__':
    # Test
    loss_fn = TransformationLoss(weight=0.1)

    batch_size, seq_len, vocab_size = 2, 10, 100
    predictions = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(predictions, targets, inputs)
    print(f"Transformation loss: {loss.item():.4f}")
