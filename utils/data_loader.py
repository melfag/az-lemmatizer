"""
Data loading utilities for Azerbaijani lemmatization.
Based on Section 4.2 from the thesis.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from utils.vocabulary import CharacterVocabulary
from utils.preprocessing import AzerbaijaniPreprocessor


class LemmatizationDataset(Dataset):
    """
    Dataset for lemmatization training.
    
    Each example contains:
    - word: inflected word form
    - context: sentence containing the word
    - lemma: target lemma
    - word_indices: character indices for the word
    - lemma_indices: character indices for the lemma
    """
    
    def __init__(self,
                 examples: List[Dict],
                 char_vocab: CharacterVocabulary,
                 bert_tokenizer,
                 max_word_length: int = 50,
                 max_context_length: int = 128):
        """
        Initialize dataset.
        
        Args:
            examples: List of (word, context, lemma) examples
            char_vocab: Character vocabulary
            bert_tokenizer: BERT tokenizer for context
            max_word_length: Maximum word length in characters
            max_context_length: Maximum context length in tokens
        """
        self.examples = examples
        self.char_vocab = char_vocab
        self.bert_tokenizer = bert_tokenizer
        self.max_word_length = max_word_length
        self.max_context_length = max_context_length
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single example.
        
        Returns:
            Dictionary with:
            - word_chars: character indices for word (input)
            - word_chars_length: actual length of word
            - lemma_chars: character indices for lemma (target)
            - lemma_chars_length: actual length of lemma
            - context_input_ids: BERT input IDs for context
            - context_attention_mask: BERT attention mask
            - target_word_mask: mask indicating target word position in context
        """
        example = self.examples[idx]
        
        word = example['word']
        context = example['context']
        lemma = example['lemma']
        
        # Encode word as character sequence
        word_chars = self.char_vocab.encode(
            word, 
            max_length=self.max_word_length,
            add_special_tokens=True
        )
        word_length = min(len(word) + 2, self.max_word_length)  # +2 for START/END
        
        # Encode lemma as character sequence
        lemma_chars = self.char_vocab.encode(
            lemma,
            max_length=self.max_word_length,
            add_special_tokens=True
        )
        lemma_length = min(len(lemma) + 2, self.max_word_length)  # +2 for START/END
        
        # Encode context with BERT tokenizer
        # Mark the target word position
        context_encoding = self.bert_tokenizer(
            context,
            max_length=self.max_context_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create target word mask (identify which tokens correspond to target word)
        target_word_mask = self._create_target_word_mask(
            context, word, context_encoding
        )

        # Find target word position in context (character offsets)
        word_start = context.find(word)
        if word_start == -1:
            word_start = 0
        word_end = word_start + len(word)

        return {
            # Keys expected by trainer/model
            'input_char_ids': torch.LongTensor(word_chars),
            'input_lengths': word_length,
            'target_char_ids': torch.LongTensor(lemma_chars),
            'target_lengths': lemma_length,
            'context_sentences': context,  # Raw sentence for contextual encoder
            'target_positions': (word_start, word_end),  # Character positions

            # Additional useful keys
            'context_input_ids': context_encoding['input_ids'].squeeze(0),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(0),
            'target_word_mask': torch.FloatTensor(target_word_mask),
            'word_str': word,
            'lemma_str': lemma,
            'context_str': context,

            # Legacy keys for backward compatibility
            'word_chars': torch.LongTensor(word_chars),
            'word_chars_length': word_length,
            'lemma_chars': torch.LongTensor(lemma_chars),
            'lemma_chars_length': lemma_length
        }
    
    def _create_target_word_mask(self, 
                                 context: str, 
                                 word: str,
                                 context_encoding) -> List[float]:
        """
        Create mask indicating target word position in tokenized context.
        
        Args:
            context: Context sentence
            word: Target word
            context_encoding: BERT tokenization output
            
        Returns:
            Binary mask (1.0 for target word tokens, 0.0 otherwise)
        """
        # Find word position in context
        word_start = context.find(word)
        if word_start == -1:
            # Word not found, return zero mask
            return [0.0] * self.max_context_length
        
        word_end = word_start + len(word)
        
        # Get character-to-token mapping
        char_to_token = context_encoding.char_to_token(0)  # batch_index=0
        
        # Create mask
        mask = [0.0] * self.max_context_length
        
        for char_idx in range(word_start, word_end):
            token_idx = context_encoding.char_to_token(0, char_idx)
            if token_idx is not None and token_idx < self.max_context_length:
                mask[token_idx] = 1.0
        
        return mask


class DOLLMADataProcessor:
    """
    Process DOLLMA dataset for lemmatization.
    
    The DOLLMA dataset contains raw Azerbaijani text. We need to:
    1. Extract word-lemma pairs (potentially using a morphological analyzer)
    2. Create context windows
    3. Split into train/val/test sets
    """
    
    def __init__(self,
                 dataset_name: str = "allmalab/DOLLMA",
                 cache_dir: Optional[str] = None):
        """
        Initialize DOLLMA processor.
        
        Args:
            dataset_name: HuggingFace dataset name
            cache_dir: Cache directory for downloaded data
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.preprocessor = AzerbaijaniPreprocessor()
        
    def load_raw_dataset(self):
        """
        Load raw DOLLMA dataset from HuggingFace.
        
        Returns:
            Loaded dataset
        """
        print(f"Loading {self.dataset_name} dataset...")
        dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        return dataset
    
    def create_synthetic_examples(self, 
                                  texts: List[str],
                                  num_examples: int = 10000) -> List[Dict]:
        """
        Create synthetic lemmatization examples from raw text.
        
        NOTE: This is a temporary solution. In the real thesis, they use
        the pre-annotated DOLLMA dataset with (word, context, lemma) triples.
        
        For this implementation, we'll create synthetic examples by:
        1. Extracting words from sentences
        2. Using simple heuristics to generate lemmas
        3. Creating proper training examples
        
        In production, you should use properly annotated data.
        
        Args:
            texts: List of text documents
            num_examples: Number of examples to create
            
        Returns:
            List of (word, context, lemma) examples
        """
        examples = []
        
        print(f"Creating synthetic examples from {len(texts)} texts...")
        
        for text in tqdm(texts[:num_examples // 10]):  # Rough estimate
            # Split into sentences
            sentences = self._split_sentences(text)
            
            for sentence in sentences:
                # Extract words
                words = sentence.split()
                
                for word in words:
                    if len(word) < 3 or len(examples) >= num_examples:
                        continue
                    
                    # Clean word
                    word_clean = self.preprocessor.preprocess(word)
                    
                    # Generate synthetic lemma (simplified)
                    lemma = self._generate_synthetic_lemma(word_clean)
                    
                    # Create example
                    example = {
                        'word': word_clean,
                        'context': sentence,
                        'lemma': lemma
                    }
                    
                    examples.append(example)
                    
                    if len(examples) >= num_examples:
                        break
        
        return examples
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        import re
        # Split on period, exclamation, question mark followed by space/newline
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _generate_synthetic_lemma(self, word: str) -> str:
        """
        Generate synthetic lemma using simple heuristics.
        
        NOTE: This is a placeholder. Real lemmatization requires
        proper morphological analysis.
        
        Common Azerbaijani suffixes to remove:
        - Plural: -lar, -lər
        - Case markers: -da, -də, -dan, -dən, -a, -ə, -ı, -i, -ın, -in
        - Possessive: -ım, -im, -um, -üm, -ımız, -imiz, etc.
        - Verb: -dı, -di, -du, -dü, -maq, -mək, -ır, -ir, -ur, -ür
        
        Args:
            word: Input word
            
        Returns:
            Synthetic lemma
        """
        lemma = word.lower()
        
        # Remove common suffixes (very simplified)
        suffixes = [
            'larımızda', 'lərimizə', 'larından', 'lərindən',
            'larımız', 'lərimiz', 'lardan', 'lərdan',
            'ların', 'lərin', 'larım', 'lərim',
            'lar', 'lər', 'dan', 'dən', 'ın', 'in',
            'da', 'də', 'ım', 'im', 'um', 'üm',
            'dı', 'di', 'du', 'dü', 'maq', 'mək',
            'ır', 'ir', 'ur', 'ür', 'a', 'ə'
        ]
        
        for suffix in suffixes:
            if lemma.endswith(suffix) and len(lemma) > len(suffix) + 2:
                return lemma[:-len(suffix)]
        
        return lemma
    
    def load_annotated_examples(self, filepath: str) -> List[Dict]:
        """
        Load pre-annotated examples from file.
        
        Expected format: JSON with list of objects containing
        'word', 'context', and 'lemma' fields.
        
        Args:
            filepath: Path to annotated data file
            
        Returns:
            List of examples
        """
        print(f"Loading annotated examples from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        print(f"Loaded {len(examples)} examples")
        return examples
    
    def split_data(self, 
                   examples: List[Dict],
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split examples into train/val/test sets (Section 4.2.3).
        
        Args:
            examples: List of examples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed
            
        Returns:
            (train_examples, val_examples, test_examples)
        """
        import random
        random.seed(seed)
        
        # Shuffle examples
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Calculate split points
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_examples = shuffled[:train_end]
        val_examples = shuffled[train_end:val_end]
        test_examples = shuffled[val_end:]
        
        print(f"Split data:")
        print(f"  Train: {len(train_examples)} examples ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(val_examples)} examples ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(test_examples)} examples ({test_ratio*100:.0f}%)")
        
        return train_examples, val_examples, test_examples
    
    def save_split_data(self,
                       train_examples: List[Dict],
                       val_examples: List[Dict],
                       test_examples: List[Dict],
                       output_dir: str):
        """
        Save split data to files.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            test_examples: Test examples
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        for split_name, examples in [
            ('train', train_examples),
            ('val', val_examples),
            ('test', test_examples)
        ]:
            filepath = output_path / f'{split_name}.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(examples)} examples to {filepath}")


def create_data_loaders(train_examples: List[Dict],
                       val_examples: List[Dict],
                       test_examples: List[Dict],
                       char_vocab: CharacterVocabulary,
                       bert_tokenizer,
                       batch_size: int = 64,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_examples: Training examples
        val_examples: Validation examples
        test_examples: Test examples
        char_vocab: Character vocabulary
        bert_tokenizer: BERT tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = LemmatizationDataset(
        train_examples, char_vocab, bert_tokenizer
    )
    val_dataset = LemmatizationDataset(
        val_examples, char_vocab, bert_tokenizer
    )
    test_dataset = LemmatizationDataset(
        test_examples, char_vocab, bert_tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Not needed for MPS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def get_dataloaders(train_examples: List[Dict],
                   val_examples: List[Dict],
                   test_examples: List[Dict],
                   char_vocab: CharacterVocabulary,
                   bert_tokenizer,
                   batch_size: int = 64,
                   num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Alias for create_data_loaders for backward compatibility.

    Args:
        train_examples: Training examples
        val_examples: Validation examples
        test_examples: Test examples
        char_vocab: Character vocabulary
        bert_tokenizer: BERT tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        (train_loader, val_loader, test_loader)
    """
    return create_data_loaders(
        train_examples, val_examples, test_examples,
        char_vocab, bert_tokenizer, batch_size, num_workers
    )


def create_dataloader(examples: List[Dict],
                     char_vocab: CharacterVocabulary,
                     bert_tokenizer,
                     batch_size: int = 64,
                     shuffle: bool = False,
                     num_workers: int = 4) -> DataLoader:
    """
    Create a single data loader.

    Args:
        examples: List of examples
        char_vocab: Character vocabulary
        bert_tokenizer: BERT tokenizer
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers

    Returns:
        DataLoader
    """
    dataset = LemmatizationDataset(examples, char_vocab, bert_tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )

    return loader


if __name__ == "__main__":
    # Test data loading
    from transformers import AutoTokenizer
    
    # Initialize
    char_vocab = CharacterVocabulary()
    bert_tokenizer = AutoTokenizer.from_pretrained("allmalab/bert-base-aze")
    
    # Create sample examples
    examples = [
        {
            'word': 'kitabları',
            'context': 'Mən kitabları oxuyuram.',
            'lemma': 'kitab'
        },
        {
            'word': 'oxuyuram',
            'context': 'Mən kitabları oxuyuram.',
            'lemma': 'oxumaq'
        }
    ]
    
    # Create dataset
    dataset = LemmatizationDataset(examples, char_vocab, bert_tokenizer)
    
    # Test __getitem__
    item = dataset[0]
    print("Sample item:")
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")