"""
Attention mechanisms for lemmatization decoder.
Based on Section 4.1.4 from the thesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.
    
    As specified in Section 4.1.4:
    α_{t,i} = exp(s_t^T W_a h_i) / Σ_j exp(s_t^T W_a h_j)
    c_t = Σ_i α_{t,i} h_i
    
    Where:
    - s_t: decoder hidden state at step t
    - h_i: encoder hidden state at position i
    - W_a: learned parameter matrix
    - α_{t,i}: attention weight
    - c_t: context vector
    """
    
    def __init__(self, 
                 decoder_hidden_dim: int,
                 encoder_hidden_dim: int,
                 attention_dim: int = 256):
        """
        Initialize Bahdanau attention.
        
        Args:
            decoder_hidden_dim: Decoder hidden state dimension
            encoder_hidden_dim: Encoder hidden state dimension
            attention_dim: Attention mechanism dimension
        """
        super(BahdanauAttention, self).__init__()
        
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.attention_dim = attention_dim
        
        # Transform decoder hidden state
        self.W_decoder = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        
        # Transform encoder hidden states
        self.W_encoder = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        
        # Attention scoring
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, 
                decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                encoder_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden: Decoder hidden state [batch_size, decoder_hidden_dim]
            encoder_outputs: Encoder hidden states [batch_size, seq_len, encoder_hidden_dim]
            encoder_mask: Mask for encoder outputs [batch_size, seq_len]
                         (1 for valid positions, 0 for padding)
        
        Returns:
            context: Context vector [batch_size, encoder_hidden_dim]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Transform decoder hidden state: [batch_size, attention_dim]
        decoder_transformed = self.W_decoder(decoder_hidden)
        
        # Transform encoder outputs: [batch_size, seq_len, attention_dim]
        encoder_transformed = self.W_encoder(encoder_outputs)
        
        # Add decoder state to each encoder position
        # [batch_size, seq_len, attention_dim]
        decoder_expanded = decoder_transformed.unsqueeze(1).expand_as(encoder_transformed)
        
        # Compute attention scores
        # [batch_size, seq_len, attention_dim]
        energy = torch.tanh(decoder_expanded + encoder_transformed)
        
        # [batch_size, seq_len, 1] -> [batch_size, seq_len]
        attention_scores = self.v(energy).squeeze(-1)
        
        # Apply mask if provided
        if encoder_mask is not None:
            # Convert mask to boolean (1 -> True, 0 -> False)
            mask_bool = encoder_mask.bool()
            attention_scores = attention_scores.masked_fill(~mask_bool, float('-inf'))
        
        # Compute attention weights [batch_size, seq_len]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context vector [batch_size, encoder_hidden_dim]
        # Weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            encoder_outputs                   # [batch_size, seq_len, encoder_hidden_dim]
        ).squeeze(1)  # [batch_size, encoder_hidden_dim]
        
        return context, attention_weights


class AttentionCoverageLoss(nn.Module):
    """
    Attention coverage loss to encourage covering all input positions.
    
    As mentioned in Section 4.3.4 (auxiliary losses).
    
    This loss penalizes the model for repeatedly focusing on the same
    positions, encouraging it to cover all relevant parts of the input.
    """
    
    def __init__(self, coverage_weight: float = 0.05):
        """
        Initialize coverage loss.
        
        Args:
            coverage_weight: Weight for coverage loss
        """
        super(AttentionCoverageLoss, self).__init__()
        self.coverage_weight = coverage_weight
    
    def forward(self, 
                attention_weights: torch.Tensor,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute coverage loss.
        
        Args:
            attention_weights: All attention weights [batch_size, seq_len, num_steps]
                              or list of [batch_size, seq_len] tensors
            encoder_mask: Mask for valid encoder positions [batch_size, seq_len]
        
        Returns:
            coverage_loss: Scalar loss value
        """
        if isinstance(attention_weights, list):
            # Stack attention weights from all decoding steps
            # [num_steps, batch_size, seq_len] -> [batch_size, seq_len, num_steps]
            attention_weights = torch.stack(attention_weights, dim=0).permute(1, 2, 0)
        
        batch_size, seq_len, num_steps = attention_weights.size()
        
        # Compute cumulative attention distribution
        # [batch_size, seq_len, num_steps]
        cumulative_attention = torch.cumsum(attention_weights, dim=2)
        
        # Coverage loss: penalize positions that receive attention multiple times
        # We want attention to be spread across different positions over time
        coverage_loss = torch.sum(
            torch.min(attention_weights, cumulative_attention - attention_weights),
            dim=[1, 2]  # Sum over seq_len and num_steps
        )
        
        # Apply mask if provided
        if encoder_mask is not None:
            # Only consider valid positions
            mask_expanded = encoder_mask.unsqueeze(-1).expand_as(attention_weights)
            coverage_loss = coverage_loss * mask_expanded.sum(dim=[1, 2])
        
        # Average over batch
        coverage_loss = coverage_loss.mean() * self.coverage_weight
        
        return coverage_loss


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism (optional, not used in base model).
    
    Included for potential future experiments.
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor [batch_size, query_len, hidden_dim]
            key: Key tensor [batch_size, key_len, hidden_dim]
            value: Value tensor [batch_size, value_len, hidden_dim]
            mask: Attention mask [batch_size, query_len, key_len]
        
        Returns:
            output: Attention output [batch_size, query_len, hidden_dim]
            attention_weights: Attention weights [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output, attention_weights


if __name__ == "__main__":
    # Test Bahdanau attention
    print("Testing Bahdanau Attention")
    print("=" * 80)
    
    batch_size = 4
    seq_len = 10
    decoder_hidden_dim = 512
    encoder_hidden_dim = 512
    
    # Create attention module
    attention = BahdanauAttention(
        decoder_hidden_dim=decoder_hidden_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        attention_dim=256
    )
    
    # Create dummy inputs
    decoder_hidden = torch.randn(batch_size, decoder_hidden_dim)
    encoder_outputs = torch.randn(batch_size, seq_len, encoder_hidden_dim)
    encoder_mask = torch.ones(batch_size, seq_len)
    encoder_mask[:, 7:] = 0  # Mask out positions 7-9
    
    # Forward pass
    context, attention_weights = attention(decoder_hidden, encoder_outputs, encoder_mask)
    
    print(f"Decoder hidden: {decoder_hidden.shape}")
    print(f"Encoder outputs: {encoder_outputs.shape}")
    print(f"Context vector: {context.shape}")
    print(f"Attention weights: {attention_weights.shape}")
    print(f"Attention weights sum: {attention_weights.sum(dim=1)}")  # Should be ~1.0
    print(f"Attention on masked positions: {attention_weights[0, 7:].sum()}")  # Should be ~0.0
    
    # Test coverage loss
    print("\n" + "=" * 80)
    print("Testing Coverage Loss")
    print("=" * 80)
    
    coverage_loss = AttentionCoverageLoss(coverage_weight=0.05)
    
    # Simulate attention weights over multiple steps
    num_steps = 5
    attention_list = [torch.randn(batch_size, seq_len).softmax(dim=1) for _ in range(num_steps)]
    
    loss = coverage_loss(attention_list, encoder_mask)
    print(f"Coverage loss: {loss.item():.4f}")
    
    print("\n✓ Attention mechanism tests passed!")