"""
Main Lemmatizer model
Combines all components: character encoder, contextual encoder, fusion, and decoder
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from models.character_encoder import CharacterEncoder
from models.contextual_encoder import ContextualEncoder
from models.fusion_mechanism import FusionMechanism
from models.decoder import Decoder


class ContextAwareLemmatizer(nn.Module):
    """
    Complete context-aware lemmatizer for Azerbaijani.
    Combines BiLSTM character encoding with AllmaBERT contextual embeddings.
    
    Main model described in Chapter 4 of the thesis.
    """
    
    def __init__(
        self,
        char_vocab_size: int,
        char_vocab=None,
        char_embedding_dim: int = 128,
        char_hidden_dim: int = 256,
        char_num_layers: int = 2,
        bert_model_name: str = "allmalab/bert-base-aze",
        bert_freeze_layers: int = 4,
        fusion_output_dim: int = 512,
        decoder_hidden_dim: int = 512,
        decoder_num_layers: int = 1,
        dropout: float = 0.3,
        use_copy: bool = True,
        copy_penalty: float = 0.0
    ):
        """
        Args:
            char_vocab_size: Size of character vocabulary
            char_vocab: Character vocabulary object (for vowel harmony)
            char_embedding_dim: Dimension of character embeddings
            char_hidden_dim: Hidden dimension of character BiLSTM (per direction)
            char_num_layers: Number of BiLSTM layers
            bert_model_name: HuggingFace model name for AllmaBERT
            bert_freeze_layers: Number of BERT layers to freeze
            fusion_output_dim: Output dimension of fusion mechanism
            decoder_hidden_dim: Hidden dimension of decoder LSTM
            decoder_num_layers: Number of decoder LSTM layers
            dropout: Dropout probability
            use_copy: Whether to use copy mechanism in decoder
            copy_penalty: Penalty for copy mechanism (0-1, higher = less copying)
        """
        super().__init__()

        self.char_vocab_size = char_vocab_size
        self.char_vocab = char_vocab
        self.char_embedding_dim = char_embedding_dim
        
        # Character-level encoder
        self.character_encoder = CharacterEncoder(
            vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            hidden_dim=char_hidden_dim,
            num_layers=char_num_layers,
            dropout=dropout
        )
        
        # Contextual encoder (AllmaBERT)
        self.contextual_encoder = ContextualEncoder(
            model_name=bert_model_name,
            freeze_first_n_layers=bert_freeze_layers
        )
        
        # Fusion mechanism
        self.fusion = FusionMechanism(
            char_hidden_dim=char_hidden_dim * 2,  # BiLSTM is bidirectional
            contextual_hidden_dim=self.contextual_encoder.hidden_size,
            output_dim=fusion_output_dim,
            char_vocab=char_vocab,
            char_embedding_dim=char_embedding_dim,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            encoder_hidden_dim=fusion_output_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            dropout=dropout,
            use_copy=use_copy,
            copy_penalty=copy_penalty
        )

        # Share character embeddings between encoder and decoder
        self.decoder.char_embedding = self.character_encoder.char_embedding

        # Projection layer to align char hidden states with fusion output dim
        self.char_to_fusion_projection = nn.Linear(char_hidden_dim * 2, fusion_output_dim)
        
    def forward(
        self,
        input_char_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        context_sentences: List[str],
        target_word_positions: List[Tuple[int, int]],
        target_char_ids: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_char_ids: (batch_size, max_input_len) Input word characters
            input_lengths: (batch_size,) Actual lengths
            context_sentences: List of context sentences
            target_word_positions: List of (start, end) char positions in sentences
            target_char_ids: (batch_size, max_target_len) Target lemma characters
            teacher_forcing_ratio: Probability of teacher forcing
            
        Returns:
            Dictionary containing:
                - outputs: (batch_size, max_target_len, vocab_size)
                - attentions: (batch_size, max_target_len, max_input_len)
        """
        batch_size = input_char_ids.size(0)
        
        # 1. Character-level encoding
        char_hidden_states, char_final_representation = self.character_encoder(
            char_ids=input_char_ids,
            lengths=input_lengths
        )
        # char_hidden_states: (batch, max_input_len, char_hidden_dim * 2)
        # char_final_representation: (batch, char_hidden_dim * 2)
        
        # 2. Contextual encoding
        contextual_representation = self.contextual_encoder(
            sentences=context_sentences,
            target_word_positions=target_word_positions
        )
        # contextual_representation: (batch, bert_hidden_size)
        
        # 3. Fusion (pass embedding layer for vowel harmony, not tensor)
        fused_representation = self.fusion(
            char_representation=char_final_representation,
            contextual_representation=contextual_representation,
            char_embedding_layer=self.character_encoder.char_embedding
        )
        # fused_representation: (batch, fusion_output_dim)
        
        # 5. Expand fused representation to match input sequence length for attention
        max_input_len = input_char_ids.size(1)
        encoder_hidden_states = fused_representation.unsqueeze(1).expand(
            batch_size, max_input_len, -1
        )
        # encoder_hidden_states: (batch, max_input_len, fusion_output_dim)

        # For better attention, we can also use the character hidden states
        # Project them to fusion output dim
        encoder_hidden_states = self.char_to_fusion_projection(char_hidden_states) + encoder_hidden_states
        
        # 6. Decoding
        outputs, attentions = self.decoder(
            encoder_hidden_states=encoder_hidden_states,
            encoder_final_state=fused_representation,
            target_char_ids=target_char_ids,
            input_char_ids=input_char_ids,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        return {
            'outputs': outputs,
            'attentions': attentions,
            'char_representation': char_final_representation,
            'contextual_representation': contextual_representation,
            'fused_representation': fused_representation
        }
    
    def lemmatize(
        self,
        input_char_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        context_sentences: List[str],
        target_word_positions: List[Tuple[int, int]],
        max_length: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference method for lemmatization.
        
        Args:
            input_char_ids: (batch_size, max_input_len)
            input_lengths: (batch_size,)
            context_sentences: List of context sentences
            target_word_positions: List of (start, end) positions
            max_length: Maximum output length
            
        Returns:
            predictions: (batch_size, max_length) Predicted character IDs
            attentions: (batch_size, max_length, max_input_len)
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = input_char_ids.size(0)
            
            # Encode
            char_hidden_states, char_final_representation = self.character_encoder(
                char_ids=input_char_ids,
                lengths=input_lengths
            )
            
            contextual_representation = self.contextual_encoder(
                sentences=context_sentences,
                target_word_positions=target_word_positions
            )

            fused_representation = self.fusion(
                char_representation=char_final_representation,
                contextual_representation=contextual_representation,
                char_embedding_layer=self.character_encoder.char_embedding
            )
            
            # Prepare encoder states for decoder
            max_input_len = input_char_ids.size(1)
            encoder_hidden_states = fused_representation.unsqueeze(1).expand(
                batch_size, max_input_len, -1
            )

            encoder_hidden_states = self.char_to_fusion_projection(char_hidden_states) + encoder_hidden_states
            
            # Decode
            predictions, attentions = self.decoder.decode_greedy(
                encoder_hidden_states=encoder_hidden_states,
                encoder_final_state=fused_representation,
                input_char_ids=input_char_ids,
                max_length=max_length
            )
            
            return predictions, attentions
    
    def get_parameter_groups(self, bert_lr: float = 1e-5, other_lr: float = 3e-4):
        """
        Get parameter groups with different learning rates.
        
        Args:
            bert_lr: Learning rate for BERT parameters
            other_lr: Learning rate for other parameters
            
        Returns:
            List of parameter groups
        """
        bert_params = list(self.contextual_encoder.bert.parameters())
        other_params = [
            p for n, p in self.named_parameters()
            if not n.startswith('contextual_encoder.bert')
        ]
        
        return [
            {'params': bert_params, 'lr': bert_lr},
            {'params': other_params, 'lr': other_lr}
        ]