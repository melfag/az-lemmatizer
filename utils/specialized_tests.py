"""
Generate specialized test sets for detailed evaluation.
Based on Section 5.3 from the thesis.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict


class SpecializedTestSetGenerator:
    """
    Generate specialized test sets for evaluating specific aspects of lemmatization.
    
    Test sets:
    1. Ambiguity Test Set (Section 5.3.1)
    2. Morphological Complexity Test Set (Section 5.3.2)
    3. Linguistic Phenomena Test Sets (Section 5.3.3)
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            seed: Random seed
        """
        random.seed(seed)
        self.seed = seed
    
    def create_ambiguity_test_set(self, 
                                  examples: List[Dict],
                                  target_size: int = 10000) -> List[Dict]:
        """
        Create ambiguity test set (Section 5.3.1).
        
        Identifies word forms that can have different lemmas depending on context.
        
        Examples from thesis:
        - "gözlər" → "göz" (noun, "eyes") or "gözlə" (verb, "wait")
        - "qaçdı" → "qaç" (verb, "ran") or "qaçdı" (noun, "fugitive")
        - "alma" → "alma" (noun, "apple") or "al" (verb, "take" + negation)
        
        Args:
            examples: All available examples
            target_size: Target number of examples
            
        Returns:
            Ambiguity test set
        """
        print(f"Creating ambiguity test set...")
        
        # Group examples by word form
        word_to_examples = defaultdict(list)
        for ex in examples:
            word_to_examples[ex['word']].append(ex)
        
        # Find ambiguous word forms (same word, different lemmas)
        ambiguous_examples = []
        
        for word, word_examples in word_to_examples.items():
            # Get unique lemmas for this word
            lemmas = set(ex['lemma'] for ex in word_examples)
            
            # If word has multiple possible lemmas, it's ambiguous
            if len(lemmas) > 1:
                # Add all examples of this ambiguous word
                ambiguous_examples.extend(word_examples)
        
        # Sample if we have too many
        if len(ambiguous_examples) > target_size:
            ambiguous_examples = random.sample(ambiguous_examples, target_size)
        
        print(f"  Found {len(ambiguous_examples)} ambiguous examples")
        
        # Add metadata
        for ex in ambiguous_examples:
            ex['test_type'] = 'ambiguity'
            ex['is_ambiguous'] = True
        
        return ambiguous_examples
    
    def create_morphological_complexity_test_set(self,
                                                examples: List[Dict]) -> List[Dict]:
        """
        Create morphological complexity test set (Section 5.3.2).
        
        Categorizes examples by number of morphemes:
        - Level 1: 1 morpheme (just root)
        - Level 2: 2 morphemes (root + 1 suffix)
        - Level 3: 3 morphemes (root + 2 suffixes)
        - Level 4: 4 morphemes (root + 3 suffixes)
        - Level 5+: 5+ morphemes (root + 4+ suffixes)
        
        Args:
            examples: All available examples
            
        Returns:
            Complexity test set with morpheme counts
        """
        print(f"Creating morphological complexity test set...")
        
        complexity_examples = []
        
        for ex in examples:
            word = ex['word']
            lemma = ex['lemma']
            
            # Estimate number of morphemes (simplified heuristic)
            num_morphemes = self._estimate_morpheme_count(word, lemma)
            
            # Create example with complexity info
            complexity_ex = ex.copy()
            complexity_ex['num_morphemes'] = num_morphemes
            complexity_ex['complexity_level'] = self._get_complexity_level(num_morphemes)
            complexity_ex['test_type'] = 'morphological_complexity'
            
            complexity_examples.append(complexity_ex)
        
        # Balance across complexity levels
        balanced_examples = self._balance_by_complexity(complexity_examples)
        
        print(f"  Created {len(balanced_examples)} complexity examples")
        self._print_complexity_distribution(balanced_examples)
        
        return balanced_examples
    
    def create_linguistic_phenomena_test_sets(self,
                                            examples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create test sets for specific linguistic phenomena (Section 5.3.3).
        
        Phenomena:
        1. Vowel Harmony: Words exhibiting vowel harmony patterns
        2. Consonant Alternation: Words with consonant changes (k→y, q→ğ, etc.)
        3. Vowel Elision: Words with dropped vowels (ağız → ağzım)
        4. Loanwords: Borrowed words from Arabic, Persian, Russian, etc.
        
        Args:
            examples: All available examples
            
        Returns:
            Dictionary mapping phenomenon name to test set
        """
        print(f"Creating linguistic phenomena test sets...")
        
        phenomena_sets = {
            'vowel_harmony': [],
            'consonant_alternation': [],
            'vowel_elision': [],
            'loanwords': []
        }
        
        for ex in examples:
            word = ex['word']
            lemma = ex['lemma']
            
            # Check for vowel harmony
            if self._has_vowel_harmony(word):
                vh_ex = ex.copy()
                vh_ex['test_type'] = 'vowel_harmony'
                phenomena_sets['vowel_harmony'].append(vh_ex)
            
            # Check for consonant alternation
            if self._has_consonant_alternation(word, lemma):
                ca_ex = ex.copy()
                ca_ex['test_type'] = 'consonant_alternation'
                phenomena_sets['consonant_alternation'].append(ca_ex)
            
            # Check for vowel elision
            if self._has_vowel_elision(word, lemma):
                ve_ex = ex.copy()
                ve_ex['test_type'] = 'vowel_elision'
                phenomena_sets['vowel_elision'].append(ve_ex)
            
            # Check for loanwords (simple heuristic)
            if self._is_loanword(word):
                lw_ex = ex.copy()
                lw_ex['test_type'] = 'loanword'
                phenomena_sets['loanwords'].append(lw_ex)
        
        for phenomenon, test_set in phenomena_sets.items():
            print(f"  {phenomenon}: {len(test_set)} examples")
        
        return phenomena_sets
    
    def _estimate_morpheme_count(self, word: str, lemma: str) -> int:
        """
        Estimate number of morphemes in a word.
        
        Simplified heuristic based on:
        - Length difference between word and lemma
        - Number of removed suffixes
        
        Args:
            word: Inflected word
            lemma: Base lemma
            
        Returns:
            Estimated morpheme count
        """
        if word == lemma:
            return 1  # Just the root
        
        # Count approximate suffixes based on length difference
        length_diff = len(word) - len(lemma)
        
        # Typical Azerbaijani suffix length: 2-3 characters
        estimated_suffixes = max(1, length_diff // 2)
        
        # Total morphemes = root + suffixes
        return 1 + estimated_suffixes
    
    def _get_complexity_level(self, num_morphemes: int) -> str:
        """Get complexity level label."""
        if num_morphemes == 1:
            return 'Level 1'
        elif num_morphemes == 2:
            return 'Level 2'
        elif num_morphemes == 3:
            return 'Level 3'
        elif num_morphemes == 4:
            return 'Level 4'
        else:
            return 'Level 5+'
    
    def _balance_by_complexity(self, 
                               examples: List[Dict],
                               min_per_level: int = 100) -> List[Dict]:
        """
        Balance examples across complexity levels.
        
        Args:
            examples: All complexity examples
            min_per_level: Minimum examples per level
            
        Returns:
            Balanced examples
        """
        # Group by level
        level_examples = defaultdict(list)
        for ex in examples:
            level_examples[ex['complexity_level']].append(ex)
        
        # Sample from each level
        balanced = []
        for level, level_exs in level_examples.items():
            sample_size = min(len(level_exs), max(min_per_level, len(level_exs) // 2))
            balanced.extend(random.sample(level_exs, sample_size))
        
        return balanced
    
    def _print_complexity_distribution(self, examples: List[Dict]):
        """Print distribution of complexity levels."""
        level_counts = defaultdict(int)
        for ex in examples:
            level_counts[ex['complexity_level']] += 1
        
        print("  Complexity distribution:")
        for level in ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5+']:
            count = level_counts[level]
            if count > 0:
                print(f"    {level}: {count}")
    
    def _has_vowel_harmony(self, word: str) -> bool:
        """
        Check if word exhibits vowel harmony patterns.
        
        Azerbaijani has front-back vowel harmony.
        """
        from utils.vocabulary import CharacterVocabulary
        
        vocab = CharacterVocabulary()
        harmony_class = vocab.get_vowel_harmony_class(word)
        
        # Pure front or back vowels indicate harmony
        return harmony_class in ['front', 'back']
    
    def _has_consonant_alternation(self, word: str, lemma: str) -> bool:
        """
        Check for consonant alternations.
        
        Common alternations in Azerbaijani:
        - k ↔ y: kitab → kitabı (k→b), but also directional changes
        - q ↔ ğ: ağac → ağacı
        - Final consonant devoicing: b→p, d→t, g→k
        """
        alternations = [
            ('k', 'y'), ('y', 'k'),
            ('q', 'ğ'), ('ğ', 'q'),
            ('b', 'p'), ('p', 'b'),
            ('d', 't'), ('t', 'd'),
            ('g', 'k'), ('k', 'g')
        ]
        
        # Check if word contains any of these alternations compared to lemma
        for char1, char2 in alternations:
            if char1 in word and char2 in lemma:
                return True
            if char2 in word and char1 in lemma:
                return True
        
        return False
    
    def _has_vowel_elision(self, word: str, lemma: str) -> bool:
        """
        Check for vowel elision.
        
        Example: ağız → ağzım (vowel ı is dropped)
        """
        # Count vowels
        vowels = set('aəeıioöuü')
        
        word_vowels = sum(1 for c in word.lower() if c in vowels)
        lemma_vowels = sum(1 for c in lemma.lower() if c in vowels)
        
        # If lemma has more vowels than word (relative to length), elision occurred
        if len(lemma) > 0 and len(word) > 0:
            word_vowel_ratio = word_vowels / len(word)
            lemma_vowel_ratio = lemma_vowels / len(lemma)
            return lemma_vowel_ratio > word_vowel_ratio + 0.1
        
        return False
    
    def _is_loanword(self, word: str) -> bool:
        """
        Check if word is likely a loanword (simple heuristic).
        
        Common patterns:
        - Arabic/Persian: often start with specific consonant clusters
        - Russian: may contain 'ы', 'э', specific patterns
        - English: specific patterns
        """
        word_lower = word.lower()
        
        # Some loanword indicators (very simplified)
        loanword_patterns = [
            'tele', 'auto', 'foto', 'radio', 'video',  # Modern loanwords
            'müh', 'həy', 'məh',  # Arabic patterns
        ]
        
        return any(pattern in word_lower for pattern in loanword_patterns)
    
    def save_test_sets(self, 
                      test_sets: Dict[str, List[Dict]],
                      output_dir: str):
        """
        Save specialized test sets to files.
        
        Args:
            test_sets: Dictionary mapping test set name to examples
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for test_name, examples in test_sets.items():
            filepath = output_path / f'{test_name}_test.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(examples)} examples to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Specialized Test Set Generator")
    print("=" * 80)
    
    # Load test examples (this would come from your actual data)
    print("\nNote: This is a demo. In practice, load from your processed data.")
    
    # Create sample examples for demonstration
    sample_examples = [
        {'word': 'gözlər', 'context': 'Onun gözləri göz', 'lemma': 'göz'},
        {'word': 'gözlər', 'context': 'Sən gözlə', 'lemma': 'gözlə'},
        {'word': 'kitablar', 'context': 'Kitablar', 'lemma': 'kitab'},
        {'word': 'kitablarımda', 'context': 'Mənim kitablarımda', 'lemma': 'kitab'},
    ]
    
    # Create generator
    generator = SpecializedTestSetGenerator()
    
    # Generate test sets
    ambiguity_set = generator.create_ambiguity_test_set(sample_examples, target_size=100)
    print(f"\nAmbiguity test set: {len(ambiguity_set)} examples")
    
    complexity_set = generator.create_morphological_complexity_test_set(sample_examples)
    print(f"\nComplexity test set: {len(complexity_set)} examples")
    
    phenomena_sets = generator.create_linguistic_phenomena_test_sets(sample_examples)
    print(f"\nLinguistic phenomena test sets:")
    for name, test_set in phenomena_sets.items():
        print(f"  {name}: {len(test_set)} examples")