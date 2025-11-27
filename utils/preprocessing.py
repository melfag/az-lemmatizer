"""
Text preprocessing utilities for Azerbaijani lemmatization.
Based on Section 4.2.2 from the thesis.
"""

import re
import unicodedata
from typing import List, Tuple, Optional


class AzerbaijaniPreprocessor:
    """
    Preprocessor for Azerbaijani text.
    
    Handles:
    - Text normalization (Section 4.2.2)
    - Character encoding standardization
    - Special character handling
    - Whitespace normalization
    """
    
    # Azerbaijani-specific character mappings for normalization
    CHAR_MAPPINGS = {
        # Handle alternative representations
        'ə': 'ə',  # U+0259
        'ğ': 'ğ',  # U+011F
        'ş': 'ş',  # U+015F
        'ç': 'ç',  # U+00E7
        'ö': 'ö',  # U+00F6
        'ü': 'ü',  # U+00FC
        'ı': 'ı',  # U+0131
    }
    
    def __init__(self, 
                 normalize_unicode: bool = True,
                 standardize_quotes: bool = True,
                 normalize_whitespace: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            normalize_unicode: Normalize Unicode representation
            standardize_quotes: Convert various quote marks to standard forms
            normalize_whitespace: Normalize whitespace characters
        """
        self.normalize_unicode = normalize_unicode
        self.standardize_quotes = standardize_quotes
        self.normalize_whitespace = normalize_whitespace
    
    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return text
        
        # Unicode normalization (Section 4.2.2)
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Character normalization
        text = self._normalize_characters(text)
        
        # Quote standardization
        if self.standardize_quotes:
            text = self._standardize_quotes(text)
        
        # Whitespace normalization
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode representation to NFC form.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # NFC normalization ensures consistent representation
        return unicodedata.normalize('NFC', text)
    
    def _normalize_characters(self, text: str) -> str:
        """
        Normalize Azerbaijani-specific characters.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        for old_char, new_char in self.CHAR_MAPPINGS.items():
            text = text.replace(old_char, new_char)
        
        return text
    
    def _standardize_quotes(self, text: str) -> str:
        """
        Standardize various quote marks.
        
        Converts:
        - Curly quotes to straight quotes
        - Guillemets to standard quotes
        - Various dash types
        
        Args:
            text: Input text
            
        Returns:
            Text with standardized quotes
        """
        # Quote marks
        quote_mappings = {
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            ''': "'",  # Left single quotation mark
            ''': "'",  # Right single quotation mark
            '«': '"',  # Left-pointing guillemet
            '»': '"',  # Right-pointing guillemet
            '„': '"',  # Double low-9 quotation mark
        }
        
        for old_quote, new_quote in quote_mappings.items():
            text = text.replace(old_quote, new_quote)
        
        # Dashes
        dash_mappings = {
            '—': '-',  # Em dash
            '–': '-',  # En dash
            '−': '-',  # Minus sign
        }
        
        for old_dash, new_dash in dash_mappings.items():
            text = text.replace(old_dash, new_dash)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_context_window(self, 
                               sentence: str, 
                               target_word: str,
                               window_size: Optional[int] = None,
                               max_length: int = 128) -> Tuple[str, int, int]:
        """
        Extract context window around target word.
        
        Args:
            sentence: Full sentence
            target_word: Target word to lemmatize
            window_size: Number of words before/after (None for full sentence)
            max_length: Maximum context length in tokens
            
        Returns:
            (context, start_idx, end_idx) - context string and word position
        """
        # Tokenize sentence
        words = sentence.split()
        
        # Find target word position
        try:
            target_idx = words.index(target_word)
        except ValueError:
            # If exact match not found, try to find partial match
            target_idx = self._find_partial_match(words, target_word)
            if target_idx is None:
                # Return full sentence if target not found
                return sentence, 0, len(sentence)
        
        if window_size is None:
            # Use full sentence (Section 4.2.2: max 128 tokens)
            context_words = words
            word_start_in_context = target_idx
        else:
            # Extract window
            start_idx = max(0, target_idx - window_size)
            end_idx = min(len(words), target_idx + window_size + 1)
            context_words = words[start_idx:end_idx]
            word_start_in_context = target_idx - start_idx
        
        # Reconstruct context
        context = ' '.join(context_words)
        
        # Calculate character positions
        char_start = len(' '.join(context_words[:word_start_in_context]))
        if char_start > 0:
            char_start += 1  # Add space
        char_end = char_start + len(target_word)
        
        return context, char_start, char_end
    
    def _find_partial_match(self, words: List[str], target: str) -> Optional[int]:
        """
        Find partial match for target word in word list.
        
        Args:
            words: List of words
            target: Target word
            
        Returns:
            Index of matching word or None
        """
        target_lower = target.lower()
        for i, word in enumerate(words):
            if target_lower in word.lower() or word.lower() in target_lower:
                return i
        return None
    
    def clean_lemma(self, lemma: str) -> str:
        """
        Clean predicted lemma.
        
        Args:
            lemma: Predicted lemma
            
        Returns:
            Cleaned lemma
        """
        # Remove special tokens if present
        special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        for token in special_tokens:
            lemma = lemma.replace(token, '')
        
        # Remove extra whitespace
        lemma = lemma.strip()
        
        return lemma


def create_lemmatization_example(word: str, 
                                 context: str, 
                                 lemma: str,
                                 preprocessor: Optional[AzerbaijaniPreprocessor] = None) -> dict:
    """
    Create a lemmatization example with preprocessing.
    
    Args:
        word: Inflected word
        context: Context sentence
        lemma: Target lemma
        preprocessor: Preprocessor instance
        
    Returns:
        Dictionary with preprocessed example
    """
    if preprocessor is None:
        preprocessor = AzerbaijaniPreprocessor()
    
    # Preprocess
    word_clean = preprocessor.preprocess(word)
    context_clean = preprocessor.preprocess(context)
    lemma_clean = preprocessor.preprocess(lemma)
    
    # Extract context window
    context_window, start_idx, end_idx = preprocessor.extract_context_window(
        context_clean, word_clean
    )
    
    return {
        'word': word_clean,
        'context': context_window,
        'lemma': lemma_clean,
        'word_start_idx': start_idx,
        'word_end_idx': end_idx
    }


# Standalone helper functions for backward compatibility

_default_preprocessor = None

def get_default_preprocessor() -> AzerbaijaniPreprocessor:
    """Get or create default preprocessor instance."""
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = AzerbaijaniPreprocessor()
    return _default_preprocessor


def preprocess_text(text: str) -> str:
    """
    Standalone function to preprocess text.
    Uses default preprocessor for backward compatibility.

    Args:
        text: Input text

    Returns:
        Preprocessed text
    """
    preprocessor = get_default_preprocessor()
    return preprocessor.preprocess(text)


def normalize_azerbaijani_text(text: str) -> str:
    """
    Normalize Azerbaijani text.
    Alias for preprocess_text for backward compatibility.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    return preprocess_text(text)


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Input text

    Returns:
        List of words
    """
    preprocessor = get_default_preprocessor()
    normalized = preprocessor.preprocess(text)
    return normalized.split()


def is_valid_azerbaijani_char(char: str) -> bool:
    """
    Check if character is valid Azerbaijani character.

    Args:
        char: Character to check

    Returns:
        True if valid, False otherwise
    """
    from utils.vocabulary import CharacterVocabulary

    # Check if character is in Azerbaijani alphabet or common punctuation
    valid_chars = set(CharacterVocabulary.AZERBAIJANI_LETTERS)
    valid_chars.update([c.upper() for c in CharacterVocabulary.AZERBAIJANI_LETTERS])
    valid_chars.update([str(i) for i in range(10)])  # Digits
    valid_chars.update([' ', '.', ',', '!', '?', ':', ';', '-', '"', "'"])

    return char in valid_chars


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = AzerbaijaniPreprocessor()

    # Test text normalization
    test_text = "Mən  kitabları   oxuyuram."  # Multiple spaces
    normalized = preprocessor.preprocess(test_text)
    print(f"Original: '{test_text}'")
    print(f"Normalized: '{normalized}'")
    
    # Test context extraction
    sentence = "Mən kitabları oxuyuram və onları çox sevirəm."
    target_word = "kitabları"
    context, start, end = preprocessor.extract_context_window(sentence, target_word)
    print(f"\nSentence: {sentence}")
    print(f"Target: {target_word}")
    print(f"Context: {context}")
    print(f"Position: {start}-{end}")
    print(f"Extracted word: '{context[start:end]}'")