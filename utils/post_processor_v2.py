"""
Improved post-processing for lemmatization predictions
Addresses issues found in v1: case normalization and over-aggressive suffix removal
"""


class ImprovedLemmaPostProcessor:
    """
    Improved post-processor with smarter case handling and validation
    """

    def __init__(self):
        # Common Azerbaijani suffixes (from linguistic analysis)
        # Ordered by length (longest first) to match longer suffixes first
        self.suffixes = [
            # Plural + case combinations (try these first)
            'larımızdan', 'lərimizdən', 'larından', 'lərindən',
            'larımıza', 'lərimizə', 'larımda', 'lərimdə',
            'larımız', 'lərimiz', 'lardan', 'lərdan',
            'larına', 'lərinə', 'ların', 'lərin',
            'ları', 'ləri', 'lar', 'lər',

            # Possessive + case combinations
            'ımızdan', 'imizdən', 'umuzdan', 'üm 1üzdən',
            'ımıza', 'imizə', 'umuza', 'ümüzə',
            'ımızda', 'imizdə', 'umuzda', 'ümüzdə',
            'ımdan', 'imdən', 'umdan', 'ümdən',
            'ımıza', 'imizə', 'umuza', 'ümüzə',
            'ımda', 'imdə', 'umda', 'ümdə',

            # Case markers
            'dan', 'dən', 'tan', 'tən',
            'ında', 'ində', 'unda', 'ündə',
            'ına', 'inə', 'una', 'ünə',

            # Possessive markers
            'ımız', 'imiz', 'umuz', 'ümüz',
            'ım', 'im', 'um', 'üm',
            'ın', 'in', 'un', 'ün',
            'ı', 'i', 'u', 'ü',

            # Simple case
            'da', 'də', 'ta', 'tə',
            'a', 'ə',

            # Verb suffixes
            'mışdır', 'mişdir', 'muşdur', 'müşdür',
            'dılar', 'dilər', 'dular', 'dülər',
            'acaq', 'əcək', 'acağ', 'əcəy',
            'dıq', 'dik', 'duq', 'dük',
            'dı', 'di', 'du', 'dü',
            'dır', 'dir', 'dur', 'dür',
            'mış', 'miş', 'muş', 'müş',
            'maq', 'mək',
            'ır', 'ir', 'ur', 'ür',
            'ar', 'ər',
        ]

        # Sort by length (longest first) to match longer suffixes first
        self.suffixes = sorted(set(self.suffixes), key=len, reverse=True)

        # Common Azerbaijani stems that should NOT be further reduced
        # This prevents "bir" → "bi", "siqar" → "siq", etc.
        self.protected_words = {
            'bir', 'iki', 'üç', 'dörd', 'beş',  # Numbers
            'var', 'yox', 'çox', 'az',  # Common words
            'gəl', 'get', 'ver', 'al', 'ol',  # Common verbs (already lemmatized)
            'o', 'bu', 'şu',  # Pronouns
        }

    def normalize_case(self, text, original_word, pos_tag=None):
        """
        Smart case normalization

        Args:
            text: Predicted lemma
            original_word: Original input word
            pos_tag: POS tag if available

        Returns:
            Case-normalized lemma
        """
        # If POS tag indicates proper noun, preserve case
        if pos_tag == 'PROPN':
            return text

        # If original word starts with uppercase, check if it's a proper noun
        # (proper nouns typically stay capitalized in lemma form)
        if original_word[0].isupper() and text[0].isupper():
            # If prediction also starts uppercase, it's likely a proper noun
            return text

        # Otherwise, lowercase
        return text.lower()

    def is_valid_stem(self, stem, original_word):
        """
        Validate if a stem is acceptable

        Args:
            stem: Candidate stem
            original_word: Original word

        Returns:
            True if stem is valid
        """
        # Minimum length
        if len(stem) < 2:
            return False

        # Don't reduce protected words
        if stem in self.protected_words:
            return True

        # Stem should be at least 40% of original word length
        # This prevents "siqar" → "siq" (too aggressive)
        if len(stem) < len(original_word) * 0.4:
            return False

        # Stem should end with a vowel or valid consonant cluster
        # Azerbaijani stems typically don't end with certain consonant clusters
        if len(stem) >= 1:
            last_char = stem[-1]
            # Valid ending characters
            valid_endings = set('aəeioöuüyrlnm')
            if last_char not in valid_endings:
                # If ends with consonant, check if it's a valid single consonant ending
                valid_consonant_endings = set('bcdğfghjkqstvxz')
                if last_char not in valid_consonant_endings:
                    return False

        return True

    def remove_suffix(self, word):
        """
        Remove suffix using linguistic rules with validation

        Args:
            word: Word to lemmatize

        Returns:
            Stem (lemma candidate)
        """
        word_lower = word.lower()

        # Don't process protected words
        if word_lower in self.protected_words:
            return word_lower

        # Try each suffix
        for suffix in self.suffixes:
            if word_lower.endswith(suffix):
                stem = word_lower[:-len(suffix)]

                # Validate stem
                if self.is_valid_stem(stem, word_lower):
                    return stem

        # No valid suffix found - return original
        return word_lower

    def post_process(self, word, prediction, pos_tag=None, use_rules=True):
        """
        Main post-processing pipeline

        Args:
            word: Original input word
            prediction: Model prediction
            pos_tag: POS tag if available
            use_rules: Whether to apply rule-based fallback

        Returns:
            Post-processed lemma
        """
        # Step 1: Smart case normalization
        lemma = self.normalize_case(prediction, word, pos_tag)

        # Step 2: If model copied input (identity), try rules
        if use_rules and lemma == word.lower():
            lemma = self.remove_suffix(word)

        return lemma
