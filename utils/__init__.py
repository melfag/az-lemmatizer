"""
Utility functions and classes for Azerbaijani lemmatization.
"""

from .vocabulary import CharacterVocabulary
from .preprocessing import AzerbaijaniPreprocessor, create_lemmatization_example
from .metrics import (
    accuracy,
    average_edit_distance,
    levenshtein_distance,
    character_level_metrics,
    ambiguity_resolution_accuracy,
    calculate_all_metrics,
    error_analysis_by_category,
    morphological_complexity_analysis,
    MetricsTracker
)
from .data_loader import (
    LemmatizationDataset,
    DOLLMADataProcessor,
    create_data_loaders
)
from .specialized_tests import SpecializedTestSetGenerator

__all__ = [
    # Vocabulary
    'CharacterVocabulary',
    
    # Preprocessing
    'AzerbaijaniPreprocessor',
    'create_lemmatization_example',
    
    # Metrics
    'accuracy',
    'average_edit_distance',
    'levenshtein_distance',
    'character_level_metrics',
    'ambiguity_resolution_accuracy',
    'calculate_all_metrics',
    'error_analysis_by_category',
    'morphological_complexity_analysis',
    'MetricsTracker',
    
    # Data loading
    'LemmatizationDataset',
    'DOLLMADataProcessor',
    'create_data_loaders',
    
    # Specialized tests
    'SpecializedTestSetGenerator',
]