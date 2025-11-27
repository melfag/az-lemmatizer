"""
Post-processing for lemmatization predictions
"""


class LemmaPostProcessor:
    """
    Post-process model predictions for better accuracy
    """

    def __init__(self):
        # Common Azerbaijani suffixes (from linguistic analysis)
        # Ordered by frequency and length
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
            'dan', 'dən', 'dan', 'dən',
            'ında', 'ində', 'unda', 'ündə',
            'ına', 'inə', 'una', 'ünə',
            'dan', 'dən', 'tan', 'tən',

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
            'ar', 'ər', 'r',
        ]

        # Sort by length (longest first) to match longer suffixes first
        self.suffixes = sorted(set(self.suffixes), key=len, reverse=True)

    def normalize_case(self, text):
        """Normalize to lowercase"""
        return text.lower()

    def remove_suffix(self, word):
        """
        Remove suffix using linguistic rules

        Args:
            word: Word to lemmatize

        Returns:
            Stem (lemma candidate)
        """
        word_lower = word.lower()

        # Try each suffix
        for suffix in self.suffixes:
            if word_lower.endswith(suffix):
                stem = word_lower[:-len(suffix)]

                # Validate stem
                if len(stem) >= 2:  # Minimum stem length
                    return stem

        # No suffix found - return original
        return word_lower

    def post_process(self, word, prediction, use_rules=True):
        """
        Main post-processing pipeline

        Args:
            word: Original input word
            prediction: Model prediction
            use_rules: Whether to apply rule-based fallback

        Returns:
            Post-processed lemma
        """
        # Step 1: Normalize case
        lemma = self.normalize_case(prediction)

        # Step 2: If model copied input (identity), try rules
        if use_rules and lemma == word.lower():
            lemma = self.remove_suffix(word)

        return lemma
