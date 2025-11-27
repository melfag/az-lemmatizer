"""
Character vocabulary for Azerbaijani lemmatization.
Based on Section 4.3.1 from the thesis.
"""

import json
from typing import List, Dict, Set
from collections import Counter


class CharacterVocabulary:
    """
    Character vocabulary for Azerbaijani.
    
    The vocabulary includes:
    - Azerbaijani alphabet (34 letters, both upper and lowercase)
    - Digits (0-9)
    - Punctuation marks
    - Special tokens (<PAD>, <START>, <END>, <UNK>)
    
    Total: 85 unique characters as mentioned in Section 4.3.1
    """
    
    # Azerbaijani alphabet (Latin script)
    # https://en.wikipedia.org/wiki/Azerbaijani_alphabet
    AZERBAIJANI_LETTERS = [
        'a', 'b', 'c', 'ç', 'd', 'e', 'ə', 'f', 'g', 'ğ', 'h', 'x', 
        'ı', 'i', 'j', 'k', 'q', 'l', 'm', 'n', 'o', 'ö', 'p', 'r',
        's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z'
    ]
    
    # Front vowels and back vowels for vowel harmony
    FRONT_VOWELS = {'e', 'ə', 'i', 'ö', 'ü'}
    BACK_VOWELS = {'a', 'ı', 'o', 'u'}
    
    def __init__(self, special_tokens: Dict[str, str] = None):
        """
        Initialize character vocabulary.
        
        Args:
            special_tokens: Dictionary of special tokens (pad, start, end, unk)
        """
        if special_tokens is None:
            special_tokens = {
                'pad': '<PAD>',
                'start': '<START>',
                'end': '<END>',
                'unk': '<UNK>'
            }
        
        self.special_tokens = special_tokens
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self.char_counts: Counter = Counter()
        
        # Build initial vocabulary
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        """Build the character vocabulary."""
        chars = []
        
        # Add special tokens first
        chars.extend([
            self.special_tokens['pad'],
            self.special_tokens['start'],
            self.special_tokens['end'],
            self.special_tokens['unk']
        ])
        
        # Add Azerbaijani letters (lowercase and uppercase)
        for letter in self.AZERBAIJANI_LETTERS:
            chars.append(letter.lower())
            chars.append(letter.upper())
        
        # Add digits
        chars.extend([str(i) for i in range(10)])
        
        # Add common punctuation
        punctuation = [
            ' ', '.', ',', '!', '?', ':', ';', '-', '–', '—',
            '(', ')', '[', ']', '{', '}', '"', "'", '«', '»',
            '/', '\\', '@', '#', '$', '%', '&', '*', '+', '=',
            '<', '>', '~', '`', '^', '|', '_'
        ]
        chars.extend(punctuation)
        
        # Create mappings
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
    def add_from_data(self, texts: List[str]):
        """
        Update vocabulary from data (to capture any missing characters).

        Args:
            texts: List of text strings
        """
        for text in texts:
            for char in text:
                self.char_counts[char] += 1
                if char not in self.char2idx:
                    idx = len(self.char2idx)
                    self.char2idx[char] = idx
                    self.idx2char[idx] = char

    def build_from_texts(self, texts: List[str]):
        """
        Compatibility method for building vocabulary from texts.
        Alias for add_from_data for backward compatibility.

        Args:
            texts: List of text strings
        """
        self.add_from_data(texts)

    @property
    def char_to_idx(self) -> Dict[str, int]:
        """
        Compatibility property for accessing char2idx.
        Alias for char2idx for backward compatibility.

        Returns:
            Character to index mapping
        """
        return self.char2idx
    
    def encode(self, text: str, max_length: int = None, 
               add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to character indices.
        
        Args:
            text: Input text
            max_length: Maximum length (truncate or pad)
            add_special_tokens: Whether to add START and END tokens
            
        Returns:
            List of character indices
        """
        indices = []
        
        if add_special_tokens:
            indices.append(self.char2idx[self.special_tokens['start']])
        
        for char in text:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            else:
                indices.append(self.char2idx[self.special_tokens['unk']])
        
        if add_special_tokens:
            indices.append(self.char2idx[self.special_tokens['end']])
        
        # Padding or truncation
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length-1] + [self.char2idx[self.special_tokens['end']]]
            else:
                indices.extend([self.char2idx[self.special_tokens['pad']]] * (max_length - len(indices)))
        
        return indices
    
    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """
        Decode character indices to text.
        
        Args:
            indices: List of character indices
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text
        """
        chars = []
        special_token_values = set(self.special_tokens.values())
        
        for idx in indices:
            if idx in self.idx2char:
                char = self.idx2char[idx]
                if remove_special_tokens and char in special_token_values:
                    if char == self.special_tokens['end']:
                        break  # Stop at END token
                    continue  # Skip other special tokens
                chars.append(char)
        
        return ''.join(chars)
    
    def is_front_vowel(self, char: str) -> bool:
        """Check if character is a front vowel."""
        return char.lower() in self.FRONT_VOWELS
    
    def is_back_vowel(self, char: str) -> bool:
        """Check if character is a back vowel."""
        return char.lower() in self.BACK_VOWELS
    
    def get_vowel_harmony_class(self, word: str) -> str:
        """
        Determine vowel harmony class of a word.
        
        Args:
            word: Input word
            
        Returns:
            'front', 'back', or 'mixed'
        """
        has_front = any(self.is_front_vowel(c) for c in word)
        has_back = any(self.is_back_vowel(c) for c in word)
        
        if has_front and has_back:
            return 'mixed'
        elif has_front:
            return 'front'
        elif has_back:
            return 'back'
        else:
            return 'none'
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.char2idx)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.char2idx)
    
    @property
    def pad_idx(self) -> int:
        """Return padding token index."""
        return self.char2idx[self.special_tokens['pad']]
    
    @property
    def start_idx(self) -> int:
        """Return start token index."""
        return self.char2idx[self.special_tokens['start']]
    
    @property
    def end_idx(self) -> int:
        """Return end token index."""
        return self.char2idx[self.special_tokens['end']]
    
    @property
    def unk_idx(self) -> int:
        """Return unknown token index."""
        return self.char2idx[self.special_tokens['unk']]
    
    def save(self, filepath: str):
        """Save vocabulary to file."""
        data = {
            'char2idx': self.char2idx,
            'idx2char': {int(k): v for k, v in self.idx2char.items()},
            'special_tokens': self.special_tokens,
            'char_counts': dict(self.char_counts)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CharacterVocabulary':
        """Load vocabulary from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(special_tokens=data['special_tokens'])
        vocab.char2idx = data['char2idx']
        vocab.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        vocab.char_counts = Counter(data.get('char_counts', {}))
        
        return vocab


if __name__ == "__main__":
    # Test the vocabulary
    vocab = CharacterVocabulary()
    
    # Test encoding/decoding
    test_word = "kitablarımda"
    encoded = vocab.encode(test_word)
    decoded = vocab.decode(encoded)
    
    print(f"Original: {test_word}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {vocab.vocab_size}")
    
    # Test vowel harmony
    print(f"\nVowel harmony class: {vocab.get_vowel_harmony_class(test_word)}")