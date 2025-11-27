"""
LSTM decoder with attention and copy mechanism
Generates lemma character-by-character
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from models.components.attention import BahdanauAttention
from models.components.copy_mechanism import CopyMechanism


class Decoder(nn.Module):
    """
    LSTM decoder with attention and copy mechanism.
    Generates lemma one character at a time.
    
    Based on Section 4.1.4 of the thesis.
    """
    
    def __init__(
        self,
        vocab_size: int,
        char_embedding_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        use_copy: bool = True,
        copy_penalty: float = 0.0
    ):
        """
        Args:
            vocab_size: Size of character vocabulary
            char_embedding_dim: Dimension of character embeddings
            encoder_hidden_dim: Dimension of encoder hidden states
            decoder_hidden_dim: Hidden dimension of decoder LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            use_copy: Whether to use copy mechanism
            copy_penalty: Penalty for copy mechanism (0-1, higher = less copying)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.use_copy = use_copy
        self.copy_penalty = copy_penalty
        
        # Share character embeddings with encoder
        self.char_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=char_embedding_dim,
            padding_idx=0
        )
        
        # Initial state projection
        self.init_state_projection = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=char_embedding_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = BahdanauAttention(
            encoder_hidden_dim=encoder_hidden_dim,
            decoder_hidden_dim=decoder_hidden_dim
        )
        
        # Attentional hidden state
        self.attentional_projection = nn.Linear(
            decoder_hidden_dim + encoder_hidden_dim,
            decoder_hidden_dim
        )
        
        # Output projection
        self.output_projection = nn.Linear(decoder_hidden_dim, vocab_size)
        
        # Copy mechanism
        if use_copy:
            self.copy_mechanism = CopyMechanism(
                encoder_hidden_dim=encoder_hidden_dim,
                decoder_hidden_dim=decoder_hidden_dim,
                vocab_size=vocab_size,
                copy_penalty=copy_penalty
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_final_state: torch.Tensor,
        target_char_ids: torch.Tensor = None,
        input_char_ids: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
        max_length: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_hidden_states: (batch_size, src_seq_len, encoder_hidden_dim)
            encoder_final_state: (batch_size, encoder_hidden_dim)
            target_char_ids: (batch_size, tgt_seq_len) - ground truth for training
            input_char_ids: (batch_size, src_seq_len) - input word characters for copy
            teacher_forcing_ratio: Probability of using teacher forcing
            max_length: Maximum generation length
            
        Returns:
            outputs: (batch_size, max_length, vocab_size)
            attentions: (batch_size, max_length, src_seq_len)
        """
        batch_size = encoder_hidden_states.size(0)
        src_seq_len = encoder_hidden_states.size(1)
        device = encoder_hidden_states.device
        
        # Initialize decoder state
        hidden = self.init_state_projection(encoder_final_state).unsqueeze(0)  # (1, batch, decoder_hidden_dim)
        cell = torch.zeros_like(hidden)
        
        # Determine sequence length
        if target_char_ids is not None:
            max_length = target_char_ids.size(1)
        
        # Storage for outputs
        outputs = torch.zeros(batch_size, max_length, self.vocab_size, device=device)
        attentions = torch.zeros(batch_size, max_length, src_seq_len, device=device)
        
        # First input is <START> token (assuming index 2)
        decoder_input = torch.full((batch_size, 1), fill_value=2, dtype=torch.long, device=device)
        
        for t in range(max_length):
            # Embed current input
            embedded = self.char_embedding(decoder_input)  # (batch, 1, char_embedding_dim)
            embedded = self.dropout(embedded)
            
            # LSTM step
            lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            # lstm_out: (batch, 1, decoder_hidden_dim)
            
            # Attention
            context, attention_weights = self.attention(
                lstm_out.squeeze(1),  # (batch, decoder_hidden_dim)
                encoder_hidden_states
            )
            # context: (batch, encoder_hidden_dim)
            # attention_weights: (batch, src_seq_len)
            
            # Attentional hidden state
            attentional_hidden = torch.tanh(
                self.attentional_projection(
                    torch.cat([lstm_out.squeeze(1), context], dim=1)
                )
            )  # (batch, decoder_hidden_dim)
            
            # Generate output distribution
            output = self.output_projection(attentional_hidden)  # (batch, vocab_size)
            
            # Apply copy mechanism if enabled
            if self.use_copy and input_char_ids is not None:
                output, _ = self.copy_mechanism(
                    decoder_hidden=attentional_hidden,
                    context_vector=context,
                    encoder_outputs=encoder_hidden_states,
                    attention_weights=attention_weights,
                    vocab_distribution=output,
                    source_chars=input_char_ids
                )
            
            outputs[:, t, :] = output
            attentions[:, t, :] = attention_weights
            
            # Teacher forcing
            use_teacher_forcing = target_char_ids is not None and torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                decoder_input = target_char_ids[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(dim=1).unsqueeze(1)
        
        return outputs, attentions
    
    def decode_greedy(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_final_state: torch.Tensor,
        input_char_ids: torch.Tensor = None,
        max_length: int = 50,
        end_token_id: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding for inference.
        
        Args:
            encoder_hidden_states: (batch_size, src_seq_len, encoder_hidden_dim)
            encoder_final_state: (batch_size, encoder_hidden_dim)
            input_char_ids: (batch_size, src_seq_len) - for copy mechanism
            max_length: Maximum generation length
            end_token_id: Token ID for <END>
            
        Returns:
            predictions: (batch_size, max_length)
            attentions: (batch_size, max_length, src_seq_len)
        """
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        
        predictions = []
        all_attentions = []
        
        # Initialize
        hidden = self.init_state_projection(encoder_final_state).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        decoder_input = torch.full((batch_size, 1), fill_value=2, dtype=torch.long, device=device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for t in range(max_length):
            embedded = self.char_embedding(decoder_input)
            embedded = self.dropout(embedded)
            
            lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            
            context, attention_weights = self.attention(
                lstm_out.squeeze(1),
                encoder_hidden_states
            )
            
            attentional_hidden = torch.tanh(
                self.attentional_projection(
                    torch.cat([lstm_out.squeeze(1), context], dim=1)
                )
            )
            
            output = self.output_projection(attentional_hidden)

            if self.use_copy and input_char_ids is not None:
                output, _ = self.copy_mechanism(
                    decoder_hidden=attentional_hidden,
                    context_vector=context,
                    encoder_outputs=encoder_hidden_states,
                    attention_weights=attention_weights,
                    vocab_distribution=output,
                    source_chars=input_char_ids
                )
            
            # Greedy selection
            next_token = output.argmax(dim=1)
            
            # Mark finished sequences
            finished = finished | (next_token == end_token_id)
            
            predictions.append(next_token.unsqueeze(1))
            all_attentions.append(attention_weights.unsqueeze(1))
            
            if finished.all():
                break
            
            decoder_input = next_token.unsqueeze(1)
        
        predictions = torch.cat(predictions, dim=1)
        attentions = torch.cat(all_attentions, dim=1)
        
        return predictions, attentions