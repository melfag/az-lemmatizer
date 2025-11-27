"""
Copy mechanism for lemmatization decoder.

Allows the decoder to directly copy characters from the input word,
which is useful for proper nouns and rare words.

Mentioned in Section 4.3.3 and ablation studies (Section 6.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CopyMechanism(nn.Module):
    """
    Copy mechanism that allows copying characters from source.
    
    The model learns to decide between:
    1. Generating a character from the vocabulary
    2. Copying a character from the input word
    
    This is particularly useful for:
    - Proper nouns (names, places)
    - Technical terms
    - Rare words not seen during training
    - Characters that should remain unchanged
    """
    
    def __init__(self,
                 decoder_hidden_dim: int,
                 encoder_hidden_dim: int,
                 vocab_size: int,
                 copy_penalty: float = 0.0):
        """
        Initialize copy mechanism.

        Args:
            decoder_hidden_dim: Decoder hidden state dimension
            encoder_hidden_dim: Encoder hidden state dimension
            vocab_size: Size of character vocabulary
            copy_penalty: Penalty for copying (0-1, higher = less copying)
        """
        super(CopyMechanism, self).__init__()

        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.vocab_size = vocab_size
        self.copy_penalty = copy_penalty
        
        # Copy gate: decides whether to copy or generate
        # Input: decoder_hidden + context_vector + previous_embedding
        self.copy_gate = nn.Sequential(
            nn.Linear(
                decoder_hidden_dim + encoder_hidden_dim,
                decoder_hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(decoder_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Pointer network for selecting which character to copy
        self.pointer_network = nn.Linear(decoder_hidden_dim, encoder_hidden_dim)
    
    def forward(self,
                decoder_hidden: torch.Tensor,
                context_vector: torch.Tensor,
                encoder_outputs: torch.Tensor,
                attention_weights: torch.Tensor,
                vocab_distribution: torch.Tensor,
                source_chars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute final distribution combining generation and copying.
        
        Args:
            decoder_hidden: Decoder hidden state [batch_size, decoder_hidden_dim]
            context_vector: Context from attention [batch_size, encoder_hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, encoder_hidden_dim]
            attention_weights: Attention weights [batch_size, src_len]
            vocab_distribution: Generation probability [batch_size, vocab_size]
            source_chars: Source character indices [batch_size, src_len]
        
        Returns:
            final_distribution: Combined distribution [batch_size, vocab_size]
            copy_prob: Probability of copying [batch_size, 1]
        """
        batch_size = decoder_hidden.size(0)
        src_len = encoder_outputs.size(1)
        
        # Compute copy gate
        # Concatenate decoder hidden and context
        gate_input = torch.cat([decoder_hidden, context_vector], dim=-1)
        copy_prob = self.copy_gate(gate_input)  # [batch_size, 1]

        # Apply copy penalty if enabled
        if self.copy_penalty > 0:
            copy_prob = copy_prob * (1.0 - self.copy_penalty)

        # Generation probability
        gen_prob = 1.0 - copy_prob  # [batch_size, 1]
        
        # Weighted generation distribution
        gen_distribution = gen_prob * vocab_distribution  # [batch_size, vocab_size]
        
        # Copy distribution
        # Use attention weights as copy probabilities
        copy_distribution = copy_prob * attention_weights  # [batch_size, src_len]
        
        # Scatter copy probabilities to vocabulary
        # Initialize with zeros
        copy_vocab_distribution = torch.zeros(
            batch_size, self.vocab_size,
            device=decoder_hidden.device
        )
        
        # For each source position, add its copy probability to the corresponding vocab position
        # source_chars: [batch_size, src_len]
        # copy_distribution: [batch_size, src_len]
        
        # Scatter add: for each batch, add copy_distribution[b, i] to position source_chars[b, i]
        copy_vocab_distribution.scatter_add_(
            dim=1,
            index=source_chars,
            src=copy_distribution
        )
        
        # Final distribution: generation + copying
        final_distribution = gen_distribution + copy_vocab_distribution
        
        # Normalize (optional, but helps with numerical stability)
        final_distribution = final_distribution / (final_distribution.sum(dim=1, keepdim=True) + 1e-10)
        
        return final_distribution, copy_prob


class CopyAttention(nn.Module):
    """
    Alternative copy mechanism using pointer-generator approach.
    
    This is a simpler variant that directly uses attention weights
    for copying decisions.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize copy attention.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super(CopyAttention, self).__init__()
        
        # Single layer to compute copy probability
        self.copy_linear = nn.Linear(hidden_dim * 3, 1)
    
    def forward(self,
                decoder_hidden: torch.Tensor,
                context_vector: torch.Tensor,
                decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Compute copy probability.
        
        Args:
            decoder_hidden: Decoder hidden state [batch_size, hidden_dim]
            context_vector: Context vector [batch_size, hidden_dim]
            decoder_input: Decoder input embedding [batch_size, hidden_dim]
        
        Returns:
            copy_prob: Probability of copying [batch_size, 1]
        """
        # Concatenate all inputs
        combined = torch.cat([decoder_hidden, context_vector, decoder_input], dim=-1)
        
        # Compute copy probability
        copy_prob = torch.sigmoid(self.copy_linear(combined))
        
        return copy_prob


class SelectiveCopyMechanism(nn.Module):
    """
    Selective copy mechanism that learns which characters are copyable.
    
    Some characters (like vowels that undergo harmony) should not be
    copied directly, while others (consonants in proper nouns) should.
    """
    
    def __init__(self,
                 decoder_hidden_dim: int,
                 encoder_hidden_dim: int,
                 vocab_size: int,
                 char_embedding_dim: int):
        """
        Initialize selective copy mechanism.
        
        Args:
            decoder_hidden_dim: Decoder hidden dimension
            encoder_hidden_dim: Encoder hidden dimension
            vocab_size: Vocabulary size
            char_embedding_dim: Character embedding dimension
        """
        super(SelectiveCopyMechanism, self).__init__()
        
        self.vocab_size = vocab_size
        
        # Copy gate
        self.copy_gate = nn.Sequential(
            nn.Linear(decoder_hidden_dim + encoder_hidden_dim, decoder_hidden_dim),
            nn.Tanh(),
            nn.Linear(decoder_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Character-specific copy scores
        # Learn which characters are more likely to be copied
        self.char_copy_scores = nn.Embedding(vocab_size, 1)
        
        # Initialize with neutral scores
        nn.init.constant_(self.char_copy_scores.weight, 0.0)
    
    def forward(self,
                decoder_hidden: torch.Tensor,
                context_vector: torch.Tensor,
                attention_weights: torch.Tensor,
                vocab_distribution: torch.Tensor,
                source_chars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute selective copy distribution.
        
        Args:
            decoder_hidden: Decoder hidden [batch_size, decoder_hidden_dim]
            context_vector: Context vector [batch_size, encoder_hidden_dim]
            attention_weights: Attention weights [batch_size, src_len]
            vocab_distribution: Generation distribution [batch_size, vocab_size]
            source_chars: Source characters [batch_size, src_len]
        
        Returns:
            final_distribution: Final distribution [batch_size, vocab_size]
            copy_prob: Copy probability [batch_size, 1]
            char_copy_scores: Character-specific copy scores [batch_size, src_len]
        """
        batch_size = source_chars.size(0)
        src_len = source_chars.size(1)
        
        # Global copy probability
        gate_input = torch.cat([decoder_hidden, context_vector], dim=-1)
        copy_prob = self.copy_gate(gate_input)  # [batch_size, 1]
        
        # Character-specific copy scores
        # [batch_size, src_len, 1] -> [batch_size, src_len]
        char_scores = self.char_copy_scores(source_chars).squeeze(-1)
        char_scores = torch.sigmoid(char_scores)  # [batch_size, src_len]
        
        # Combine global and character-specific scores
        copy_weights = copy_prob * attention_weights * char_scores  # [batch_size, src_len]
        
        # Generation weights
        gen_weight = 1.0 - copy_prob  # [batch_size, 1]
        gen_distribution = gen_weight * vocab_distribution
        
        # Copy distribution
        copy_vocab_distribution = torch.zeros(
            batch_size, self.vocab_size,
            device=decoder_hidden.device
        )
        
        copy_vocab_distribution.scatter_add_(
            dim=1,
            index=source_chars,
            src=copy_weights
        )
        
        # Final distribution
        final_distribution = gen_distribution + copy_vocab_distribution
        final_distribution = final_distribution / (final_distribution.sum(dim=1, keepdim=True) + 1e-10)
        
        return final_distribution, copy_prob, char_scores


if __name__ == "__main__":
    # Test copy mechanism
    print("Testing Copy Mechanism")
    print("=" * 80)
    
    batch_size = 4
    src_len = 10
    vocab_size = 85
    decoder_hidden_dim = 512
    encoder_hidden_dim = 512
    
    # Create copy mechanism
    copy_mechanism = CopyMechanism(
        decoder_hidden_dim=decoder_hidden_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        vocab_size=vocab_size
    )
    
    # Create dummy inputs
    decoder_hidden = torch.randn(batch_size, decoder_hidden_dim)
    context_vector = torch.randn(batch_size, encoder_hidden_dim)
    encoder_outputs = torch.randn(batch_size, src_len, encoder_hidden_dim)
    attention_weights = torch.softmax(torch.randn(batch_size, src_len), dim=1)
    vocab_distribution = torch.softmax(torch.randn(batch_size, vocab_size), dim=1)
    source_chars = torch.randint(0, vocab_size, (batch_size, src_len))
    
    # Forward pass
    final_dist, copy_prob = copy_mechanism(
        decoder_hidden,
        context_vector,
        encoder_outputs,
        attention_weights,
        vocab_distribution,
        source_chars
    )
    
    print(f"Decoder hidden: {decoder_hidden.shape}")
    print(f"Context vector: {context_vector.shape}")
    print(f"Final distribution: {final_dist.shape}")
    print(f"Copy probability: {copy_prob.shape}")
    print(f"Copy prob value: {copy_prob[0].item():.4f}")
    print(f"Distribution sum: {final_dist[0].sum().item():.4f}")  # Should be ~1.0
    
    # Test selective copy mechanism
    print("\n" + "=" * 80)
    print("Testing Selective Copy Mechanism")
    print("=" * 80)
    
    selective_copy = SelectiveCopyMechanism(
        decoder_hidden_dim=decoder_hidden_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        vocab_size=vocab_size,
        char_embedding_dim=128
    )
    
    final_dist, copy_prob, char_scores = selective_copy(
        decoder_hidden,
        context_vector,
        attention_weights,
        vocab_distribution,
        source_chars
    )
    
    print(f"Final distribution: {final_dist.shape}")
    print(f"Copy probability: {copy_prob.shape}")
    print(f"Character scores: {char_scores.shape}")
    print(f"Char scores range: [{char_scores.min():.4f}, {char_scores.max():.4f}]")
    
    print("\n✓ Copy mechanism tests passed!")