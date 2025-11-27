"""
Fusion mechanism for combining character-level and contextual representations
Uses gated fusion with vowel harmony awareness
"""

import torch
import torch.nn as nn

from models.components.vowel_harmony import VowelHarmonyAwareness


class FusionMechanism(nn.Module):
    """
    Gated fusion mechanism to combine character-level and contextual information.
    Includes vowel harmony awareness for Azerbaijani.
    
    Based on Section 4.1.5 of the thesis.
    """
    
    def __init__(
        self,
        char_hidden_dim: int,
        contextual_hidden_dim: int,
        output_dim: int,
        char_vocab,
        char_embedding_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Args:
            char_hidden_dim: Dimension of character-level representation
            contextual_hidden_dim: Dimension of contextual representation
            output_dim: Dimension of fused output
            char_vocab: Character vocabulary for vowel harmony
            char_embedding_dim: Dimension of character embeddings (for vowel harmony)
            dropout: Dropout probability
        """
        super().__init__()

        self.char_hidden_dim = char_hidden_dim
        self.contextual_hidden_dim = contextual_hidden_dim
        self.output_dim = output_dim

        # Projection for contextual representation to match char dimension
        self.contextual_projection = nn.Linear(contextual_hidden_dim, char_hidden_dim)

        # Gate for controlling fusion
        self.gate = nn.Sequential(
            nn.Linear(char_hidden_dim + contextual_hidden_dim, char_hidden_dim),
            nn.Sigmoid()
        )

        # Vowel harmony awareness
        self.vowel_harmony = VowelHarmonyAwareness(
            char_vocab=char_vocab,
            char_embedding_dim=char_embedding_dim,
            hidden_dim=char_hidden_dim
        )
        
        # Final projection to output dimension
        self.output_projection = nn.Linear(char_hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        char_representation: torch.Tensor,
        contextual_representation: torch.Tensor,
        char_embedding_layer: nn.Embedding = None
    ) -> torch.Tensor:
        """
        Args:
            char_representation: (batch_size, char_hidden_dim)
            contextual_representation: (batch_size, contextual_hidden_dim)
            char_embedding_layer: Character embedding layer (nn.Embedding) - optional for vowel harmony

        Returns:
            fused_representation: (batch_size, output_dim)
        """
        # Project contextual representation
        contextual_projected = self.contextual_projection(contextual_representation)

        # Compute gate
        concat = torch.cat([char_representation, contextual_representation], dim=1)
        gate = self.gate(concat)  # (batch, char_hidden_dim)

        # Gated fusion
        fused = gate * char_representation + (1 - gate) * contextual_projected
        fused = self.dropout(fused)

        # Add vowel harmony awareness if char_embedding_layer provided
        if char_embedding_layer is not None:
            # Pass in correct order: (h_fused, char_embeddings)
            fused = self.vowel_harmony(fused, char_embedding_layer)

        # Project to output dimension
        output = self.output_projection(fused)

        return output