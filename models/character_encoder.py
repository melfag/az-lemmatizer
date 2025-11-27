"""
Character-level encoder using BiLSTM
Processes character sequences and extracts morphological features
"""

import torch
import torch.nn as nn
from typing import Tuple

from models.components.morphological_features import MorphologicalFeatureExtractor


class CharacterEncoder(nn.Module):
    """
    Character-level BiLSTM encoder with morphological feature extraction.
    
    Based on Section 4.1.2 of the thesis.
    """
    
    def __init__(
        self,
        vocab_size: int,
        char_embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        morphological_feature_dim: int = 128
    ):
        """
        Args:
            vocab_size: Size of character vocabulary
            char_embedding_dim: Dimension of character embeddings
            hidden_dim: Hidden dimension of BiLSTM (per direction)
            num_layers: Number of BiLSTM layers
            dropout: Dropout probability
            morphological_feature_dim: Dimension of morphological features
        """
        super().__init__()
        
        self.char_embedding_dim = char_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Character embeddings
        self.char_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=char_embedding_dim,
            padding_idx=0
        )
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=char_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Morphological feature extractor
        self.morphological_extractor = MorphologicalFeatureExtractor(
            char_embedding_dim=char_embedding_dim,
            feature_dim=morphological_feature_dim
        )
        
        # Projection for morphological features to match BiLSTM output dimension
        self.morphological_projection = nn.Linear(
            morphological_feature_dim * 4,  # 4 filter sizes
            hidden_dim * 2  # BiLSTM output is bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        char_ids: torch.Tensor,
        lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            char_ids: (batch_size, max_seq_len) Character indices
            lengths: (batch_size,) Actual lengths of sequences
            
        Returns:
            char_hidden_states: (batch_size, max_seq_len, hidden_dim * 2)
            char_final_representation: (batch_size, hidden_dim * 2)
        """
        batch_size, max_seq_len = char_ids.size()
        
        # Embed characters
        char_embeddings = self.char_embedding(char_ids)  # (batch, seq_len, char_embedding_dim)
        char_embeddings = self.dropout(char_embeddings)
        
        # Pack padded sequences if lengths provided
        if lengths is not None:
            char_embeddings = nn.utils.rnn.pack_padded_sequence(
                char_embeddings,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
        
        # BiLSTM encoding
        lstm_out, (hidden, cell) = self.bilstm(char_embeddings)
        
        # Unpack if we packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out,
                batch_first=True,
                total_length=max_seq_len
            )
        
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        
        # Extract morphological features
        morphological_features = self.morphological_extractor(
            self.char_embedding(char_ids)
        )  # (batch, morphological_feature_dim * 4)
        
        # Project morphological features
        morphological_features = self.morphological_projection(
            morphological_features
        )  # (batch, hidden_dim * 2)
        
        # Combine final hidden states from both directions
        # hidden: (num_layers * 2, batch, hidden_dim)
        final_hidden_forward = hidden[-2, :, :]  # (batch, hidden_dim)
        final_hidden_backward = hidden[-1, :, :]  # (batch, hidden_dim)
        final_hidden = torch.cat([final_hidden_forward, final_hidden_backward], dim=1)  # (batch, hidden_dim * 2)
        
        # Add morphological features to final representation
        char_final_representation = final_hidden + morphological_features
        
        return lstm_out, char_final_representation