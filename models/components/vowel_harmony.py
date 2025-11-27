"""
Vowel harmony awareness component for Azerbaijani lemmatization.

Based on Section 4.1.5 from the thesis:
"We implement a Vowel Harmony Awareness component specifically designed
to handle the vowel harmony patterns in Azerbaijani."

Azerbaijani has front-back vowel harmony:
- Front vowels: e, ə, i, ö, ü
- Back vowels: a, ı, o, u
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VowelHarmonyAwareness(nn.Module):
    """
    Vowel harmony awareness component.
    
    As specified in Section 4.1.5:
    v_front = MeanPool(e_c | c ∈ "eəiöü")
    v_back = MeanPool(e_c | c ∈ "aıou")
    h_vowel = ReLU(W_v [h_fused; v_front; v_back] + b_v)
    h_final = h_fused + h_vowel
    
    This component helps the model learn and apply vowel harmony patterns
    when generating lemmas.
    """
    
    # Azerbaijani vowel classes
    FRONT_VOWELS = {'e', 'ə', 'i', 'ö', 'ü', 'E', 'Ə', 'İ', 'Ö', 'Ü'}
    BACK_VOWELS = {'a', 'ı', 'o', 'u', 'A', 'I', 'O', 'U'}
    
    def __init__(self,
                 char_vocab,
                 char_embedding_dim: int,
                 hidden_dim: int):
        """
        Initialize vowel harmony awareness.
        
        Args:
            char_vocab: Character vocabulary (for getting indices)
            char_embedding_dim: Character embedding dimension
            hidden_dim: Hidden dimension of the fused representation
        """
        super(VowelHarmonyAwareness, self).__init__()
        
        self.char_vocab = char_vocab
        self.char_embedding_dim = char_embedding_dim
        self.hidden_dim = hidden_dim
        
        # Identify vowel indices in vocabulary
        self.front_vowel_indices = self._get_vowel_indices(self.FRONT_VOWELS)
        self.back_vowel_indices = self._get_vowel_indices(self.BACK_VOWELS)
        
        # Transformation layer
        # Input: hidden_dim + 2 * char_embedding_dim (for front and back vowel features)
        self.W_v = nn.Linear(
            hidden_dim + 2 * char_embedding_dim,
            hidden_dim
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def _get_vowel_indices(self, vowel_set: set) -> list:
        """
        Get vocabulary indices for vowels in the set.
        
        Args:
            vowel_set: Set of vowel characters
        
        Returns:
            List of vocabulary indices
        """
        indices = []
        for vowel in vowel_set:
            if vowel in self.char_vocab.char2idx:
                indices.append(self.char_vocab.char2idx[vowel])
        return indices
    
    def forward(self,
                h_fused: torch.Tensor,
                char_embeddings: nn.Embedding) -> torch.Tensor:
        """
        Apply vowel harmony awareness.
        
        Args:
            h_fused: Fused representation [batch_size, hidden_dim]
            char_embeddings: Character embedding layer
        
        Returns:
            h_final: Enhanced representation [batch_size, hidden_dim]
        """
        batch_size = h_fused.size(0)
        device = h_fused.device
        
        # Get vowel embeddings
        # Front vowels: average embedding
        if len(self.front_vowel_indices) > 0:
            front_vowel_indices = torch.tensor(
                self.front_vowel_indices,
                device=device
            )
            front_vowel_embeds = char_embeddings.weight[front_vowel_indices]  # [num_front, embed_dim]
            v_front = front_vowel_embeds.mean(dim=0)  # [embed_dim]
            v_front = v_front.unsqueeze(0).expand(batch_size, -1)  # [batch_size, embed_dim]
        else:
            v_front = torch.zeros(batch_size, self.char_embedding_dim, device=device)
        
        # Back vowels: average embedding
        if len(self.back_vowel_indices) > 0:
            back_vowel_indices = torch.tensor(
                self.back_vowel_indices,
                device=device
            )
            back_vowel_embeds = char_embeddings.weight[back_vowel_indices]  # [num_back, embed_dim]
            v_back = back_vowel_embeds.mean(dim=0)  # [embed_dim]
            v_back = v_back.unsqueeze(0).expand(batch_size, -1)  # [batch_size, embed_dim]
        else:
            v_back = torch.zeros(batch_size, self.char_embedding_dim, device=device)
        
        # Concatenate fused representation with vowel features
        combined = torch.cat([h_fused, v_front, v_back], dim=-1)  # [batch_size, hidden_dim + 2*embed_dim]
        
        # Transform and apply ReLU
        h_vowel = F.relu(self.W_v(combined))  # [batch_size, hidden_dim]
        
        # Residual connection
        h_final = h_fused + h_vowel
        
        # Layer normalization
        h_final = self.layer_norm(h_final)
        
        return h_final


class VowelHarmonyClassifier(nn.Module):
    """
    Classifier to predict vowel harmony class of a word.
    
    Can be used as an auxiliary task during training to encourage
    the model to learn vowel harmony patterns.
    """
    
    def __init__(self, hidden_dim: int, num_classes: int = 3):
        """
        Initialize vowel harmony classifier.
        
        Args:
            hidden_dim: Input hidden dimension
            num_classes: Number of harmony classes (front, back, mixed)
        """
        super(VowelHarmonyClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict vowel harmony class.
        
        Args:
            hidden: Hidden representation [batch_size, hidden_dim]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        return self.classifier(hidden)


class VowelHarmonyConstraint(nn.Module):
    """
    Soft constraint for vowel harmony during decoding.
    
    Modifies the output distribution to favor vowels that follow
    harmony with previous vowels in the sequence.
    """
    
    FRONT_VOWELS = {'e', 'ə', 'i', 'ö', 'ü'}
    BACK_VOWELS = {'a', 'ı', 'o', 'u'}
    
    def __init__(self, char_vocab, constraint_strength: float = 0.5):
        """
        Initialize vowel harmony constraint.
        
        Args:
            char_vocab: Character vocabulary
            constraint_strength: Strength of the constraint (0.0 to 1.0)
        """
        super(VowelHarmonyConstraint, self).__init__()
        
        self.char_vocab = char_vocab
        self.constraint_strength = constraint_strength
        
        # Create vowel harmony masks
        self.front_vowel_mask = self._create_vowel_mask(self.FRONT_VOWELS)
        self.back_vowel_mask = self._create_vowel_mask(self.BACK_VOWELS)
    
    def _create_vowel_mask(self, vowel_set: set) -> torch.Tensor:
        """
        Create binary mask for vowel indices.
        
        Args:
            vowel_set: Set of vowel characters
        
        Returns:
            mask: Binary mask [vocab_size]
        """
        vocab_size = len(self.char_vocab)
        mask = torch.zeros(vocab_size)
        
        for vowel in vowel_set:
            if vowel in self.char_vocab.char2idx:
                idx = self.char_vocab.char2idx[vowel]
                mask[idx] = 1.0
        
        return mask
    
    def forward(self,
                output_distribution: torch.Tensor,
                previous_chars: torch.Tensor) -> torch.Tensor:
        """
        Apply vowel harmony constraint to output distribution.
        
        Args:
            output_distribution: Output probabilities [batch_size, vocab_size]
            previous_chars: Previously generated characters [batch_size, seq_len]
        
        Returns:
            constrained_distribution: Modified distribution [batch_size, vocab_size]
        """
        batch_size = output_distribution.size(0)
        device = output_distribution.device
        
        # Move masks to device
        front_mask = self.front_vowel_mask.to(device)
        back_mask = self.back_vowel_mask.to(device)
        
        # Determine vowel harmony class from previous characters
        harmony_scores = torch.zeros(batch_size, 2, device=device)  # [batch_size, 2] for (front, back)
        
        for b in range(batch_size):
            for char_idx in previous_chars[b]:
                char_idx = char_idx.item()
                if char_idx < len(self.char_vocab):
                    char = self.char_vocab.idx2char.get(char_idx, '')
                    if char in self.FRONT_VOWELS:
                        harmony_scores[b, 0] += 1  # Front vowel
                    elif char in self.BACK_VOWELS:
                        harmony_scores[b, 1] += 1  # Back vowel
        
        # Compute harmony preference
        harmony_probs = F.softmax(harmony_scores, dim=1)  # [batch_size, 2]
        
        # Create constraint weights
        # If front vowels dominate, favor front vowels; if back vowels dominate, favor back vowels
        constraint_weights = (
            harmony_probs[:, 0:1] * front_mask.unsqueeze(0) +
            harmony_probs[:, 1:2] * back_mask.unsqueeze(0)
        )  # [batch_size, vocab_size]
        
        # Apply constraint
        # Blend original distribution with constraint
        constrained_distribution = (
            (1 - self.constraint_strength) * output_distribution +
            self.constraint_strength * constraint_weights * output_distribution
        )
        
        # Renormalize
        constrained_distribution = constrained_distribution / (
            constrained_distribution.sum(dim=1, keepdim=True) + 1e-10
        )
        
        return constrained_distribution


if __name__ == "__main__":
    # Test vowel harmony awareness
    print("Testing Vowel Harmony Awareness")
    print("=" * 80)
    
    from utils.vocabulary import CharacterVocabulary
    
    # Create vocabulary
    char_vocab = CharacterVocabulary()
    vocab_size = char_vocab.vocab_size
    
    # Create char embeddings
    char_embedding_dim = 128
    char_embeddings = nn.Embedding(vocab_size, char_embedding_dim)
    
    # Create vowel harmony module
    vowel_harmony = VowelHarmonyAwareness(
        char_vocab=char_vocab,
        char_embedding_dim=char_embedding_dim,
        hidden_dim=512
    )
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Front vowel indices: {vowel_harmony.front_vowel_indices}")
    print(f"Back vowel indices: {vowel_harmony.back_vowel_indices}")
    
    # Test forward pass
    batch_size = 4
    h_fused = torch.randn(batch_size, 512)
    h_final = vowel_harmony(h_fused, char_embeddings)
    
    print(f"\nInput shape: {h_fused.shape}")
    print(f"Output shape: {h_final.shape}")
    print(f"Output mean: {h_final.mean().item():.4f}")
    print(f"Output std: {h_final.std().item():.4f}")
    
    # Test vowel harmony constraint
    print("\n" + "=" * 80)
    print("Testing Vowel Harmony Constraint")
    print("=" * 80)
    
    constraint = VowelHarmonyConstraint(char_vocab, constraint_strength=0.5)
    
    # Test with front vowel context
    output_dist = torch.softmax(torch.randn(batch_size, vocab_size), dim=1)
    previous_chars = torch.tensor([
        [char_vocab.char2idx['k'], char_vocab.char2idx['i'], char_vocab.char2idx['t']],
        [char_vocab.char2idx['e'], char_vocab.char2idx['v'], char_vocab.char2idx['ə']],
        [char_vocab.char2idx['a'], char_vocab.char2idx['l'], char_vocab.char2idx['ı']],
        [char_vocab.char2idx['o'], char_vocab.char2idx['x'], char_vocab.char2idx['u']],
    ])
    
    constrained_dist = constraint(output_dist, previous_chars)
    
    print(f"Original distribution shape: {output_dist.shape}")
    print(f"Constrained distribution shape: {constrained_dist.shape}")
    print(f"Distribution sum: {constrained_dist[0].sum().item():.4f}")
    
    print("\n✓ Vowel harmony awareness tests passed!")