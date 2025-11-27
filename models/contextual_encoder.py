"""
Contextual encoder using AllmaBERT
Extracts contextual word representations from sentences
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, List


class ContextualEncoder(nn.Module):
    """
    Contextual encoder using AllmaBERT for Azerbaijani.
    Handles subword tokenization and alignment.
    
    Based on Section 4.1.3 of the thesis.
    """
    
    def __init__(
        self,
        model_name: str = "allmalab/bert-base-aze",
        freeze_first_n_layers: int = 4,
        output_layer: int = -1
    ):
        """
        Args:
            model_name: HuggingFace model name for AllmaBERT
            freeze_first_n_layers: Number of initial layers to freeze during training
            output_layer: Which layer to extract features from (-1 for last layer)
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_layer = output_layer
        
        # Load AllmaBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze first n layers
        if freeze_first_n_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_first_n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.hidden_size = self.bert.config.hidden_size
        
    def forward(
        self,
        sentences: List[str],
        target_word_positions: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Args:
            sentences: List of sentences containing target words
            target_word_positions: List of (start_char, end_char) positions for each target word
            
        Returns:
            contextual_representations: (batch_size, hidden_size)
        """
        batch_size = len(sentences)
        device = next(self.bert.parameters()).device
        
        # Tokenize sentences
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        offset_mapping = encoded['offset_mapping']
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract from specified layer
        if self.output_layer == -1:
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        else:
            hidden_states = outputs.hidden_states[self.output_layer]
        
        # Extract target word representations
        target_representations = []
        
        for i in range(batch_size):
            start_char, end_char = target_word_positions[i]
            
            # Find which tokens correspond to the target word
            token_indices = []
            for token_idx, (token_start, token_end) in enumerate(offset_mapping[i]):
                # Check if token overlaps with target word
                if token_start < end_char and token_end > start_char:
                    token_indices.append(token_idx)
            
            if not token_indices:
                # Fallback: use [CLS] token
                token_indices = [0]
            
            # Mean pool over target word tokens
            target_token_embeddings = hidden_states[i, token_indices, :]  # (num_tokens, hidden_size)
            target_representation = target_token_embeddings.mean(dim=0)  # (hidden_size,)
            
            target_representations.append(target_representation)
        
        # Stack into batch
        contextual_representations = torch.stack(target_representations, dim=0)  # (batch, hidden_size)
        
        return contextual_representations
    
    def encode_batch(
        self,
        input_words: List[str],
        contexts: List[str]
    ) -> torch.Tensor:
        """
        Convenience method to encode a batch of words with their contexts.
        
        Args:
            input_words: List of target words
            contexts: List of context sentences
            
        Returns:
            contextual_representations: (batch_size, hidden_size)
        """
        # Find word positions in contexts
        target_positions = []
        for word, context in zip(input_words, contexts):
            start_pos = context.lower().find(word.lower())
            if start_pos == -1:
                # Word not found, use middle of sentence
                start_pos = len(context) // 2
                end_pos = start_pos + len(word)
            else:
                end_pos = start_pos + len(word)
            target_positions.append((start_pos, end_pos))
        
        return self.forward(contexts, target_positions)