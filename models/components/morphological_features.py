"""
Morphological Feature Extractor using CNNs
Extracts morphological patterns at different scales (2, 3, 4, 5-character n-grams)
"""

import torch
import torch.nn as nn


class MorphologicalFeatureExtractor(nn.Module):
    """
    CNN-based morphological feature extractor.
    Uses parallel CNNs with different filter sizes to capture n-gram patterns.
    
    Based on Section 4.1.2 of the thesis.
    """
    
    def __init__(self, char_embedding_dim: int, feature_dim: int = 128):
        """
        Args:
            char_embedding_dim: Dimension of character embeddings
            feature_dim: Dimension of output features
        """
        super().__init__()
        
        self.char_embedding_dim = char_embedding_dim
        self.feature_dim = feature_dim
        
        # CNN filters of different sizes (2, 3, 4, 5) to capture different n-gram patterns
        self.conv_filters = nn.ModuleList([
            nn.Conv1d(
                in_channels=char_embedding_dim,
                out_channels=feature_dim,
                kernel_size=k,
                padding=k // 2  # Keep sequence length
            )
            for k in [2, 3, 4, 5]
        ])
        
        # Activation
        self.activation = nn.ReLU()
        
    def forward(self, char_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_embeddings: (batch_size, seq_len, char_embedding_dim)
            
        Returns:
            morphological_features: (batch_size, feature_dim * 4)
        """
        # Conv1d expects (batch, channels, length)
        char_embeddings = char_embeddings.transpose(1, 2)  # (batch, char_embedding_dim, seq_len)
        
        # Apply each CNN filter and max pool
        pooled_features = []
        for conv in self.conv_filters:
            # Apply convolution
            conv_out = self.activation(conv(char_embeddings))  # (batch, feature_dim, seq_len)
            
            # Max pool over sequence length
            pooled = torch.max(conv_out, dim=2)[0]  # (batch, feature_dim)
            pooled_features.append(pooled)
        
        # Concatenate all pooled features
        morphological_features = torch.cat(pooled_features, dim=1)  # (batch, feature_dim * 4)
        
        return morphological_features