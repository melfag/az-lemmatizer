"""
Inference script for interactive lemmatization
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.lemmatizer import ContextAwareLemmatizer
from utils.vocabulary import CharacterVocabulary
from utils.preprocessing import preprocess_text


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class LemmatizerInference:
    """
    Interactive lemmatization inference.
    """
    
    def __init__(
        self,
        model: ContextAwareLemmatizer,
        vocab: CharacterVocabulary,
        device: torch.device
    ):
        """
        Args:
            model: Trained lemmatizer model
            vocab: Character vocabulary
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device
    
    def lemmatize_word(
        self,
        word: str,
        context: str
    ) -> tuple:
        """
        Lemmatize a single word in context.
        
        Args:
            word: The word to lemmatize
            context: The sentence context containing the word
            
        Returns:
            (lemma, confidence) tuple
        """
        # Preprocess
        word = preprocess_text(word)
        context = preprocess_text(context)
        
        # Find word position in context
        word_start = context.lower().find(word.lower())
        if word_start == -1:
            # Word not found in context, use word as context
            context = word
            word_start = 0
        
        word_end = word_start + len(word)
        target_position = (word_start, word_end)
        
        # Convert word to character IDs
        char_ids = [self.vocab.start_idx]
        for char in word:
            char_ids.append(self.vocab.char_to_idx.get(char, self.vocab.unk_idx))
        char_ids.append(self.vocab.end_idx)
        
        # Pad to reasonable length
        max_len = 50
        while len(char_ids) < max_len:
            char_ids.append(self.vocab.pad_idx)
        
        # Convert to tensor
        input_char_ids = torch.tensor([char_ids[:max_len]], dtype=torch.long).to(self.device)
        input_lengths = torch.tensor([len(word) + 2], dtype=torch.long).to(self.device)
        
        # Lemmatize
        with torch.no_grad():
            predictions, attentions = self.model.lemmatize(
                input_char_ids=input_char_ids,
                input_lengths=input_lengths,
                context_sentences=[context],
                target_word_positions=[target_position],
                max_length=max_len
            )
        
        # Decode prediction
        lemma = self._decode(predictions[0])
        
        # Calculate confidence (average attention entropy)
        attention = attentions[0]  # (seq_len, src_len)
        entropy = -(attention * torch.log(attention + 1e-10)).sum(dim=1).mean()
        confidence = float(torch.exp(-entropy))
        
        return lemma, confidence
    
    def lemmatize_sentence(
        self,
        sentence: str
    ) -> list:
        """
        Lemmatize all words in a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of (word, lemma, confidence) tuples
        """
        words = sentence.split()
        results = []
        
        for word in words:
            lemma, confidence = self.lemmatize_word(word, sentence)
            results.append((word, lemma, confidence))
        
        return results
    
    def _decode(self, char_ids: torch.Tensor) -> str:
        """Decode character IDs to string."""
        chars = []
        for idx in char_ids:
            idx = idx.item()
            if idx == self.vocab.end_idx or idx == self.vocab.pad_idx:
                break
            if idx == self.vocab.start_idx:
                continue
            char = self.vocab.idx_to_char.get(idx, '')
            if char:
                chars.append(char)
        return ''.join(chars)


def interactive_mode(lemmatizer: LemmatizerInference):
    """Interactive lemmatization mode."""
    print("\n" + "="*60)
    print("Interactive Azerbaijani Lemmatization")
    print("="*60)
    print("\nCommands:")
    print("  - Enter a word and context (separated by '|')")
    print("  - Enter a sentence to lemmatize all words")
    print("  - Type 'quit' or 'exit' to quit")
    print("\nExamples:")
    print("  kitablar | Mən kitablar oxuyuram")
    print("  Mən kitablar oxuyuram")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Check if input has word|context format
            if '|' in user_input:
                parts = user_input.split('|')
                word = parts[0].strip()
                context = parts[1].strip() if len(parts) > 1 else word
                
                lemma, confidence = lemmatizer.lemmatize_word(word, context)
                
                print(f"\nInput: {word}")
                print(f"Context: {context}")
                print(f"Lemma: {lemma}")
                print(f"Confidence: {confidence:.2%}\n")
            
            else:
                # Lemmatize sentence
                results = lemmatizer.lemmatize_sentence(user_input)
                
                print("\nLemmatization Results:")
                print("-" * 60)
                print(f"{'Word':<20} {'Lemma':<20} {'Confidence':<15}")
                print("-" * 60)
                
                for word, lemma, confidence in results:
                    print(f"{word:<20} {lemma:<20} {confidence:>6.2%}")
                
                print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def batch_mode(lemmatizer: LemmatizerInference, input_file: str, output_file: str):
    """Batch lemmatization from file."""
    print(f"\nProcessing file: {input_file}")
    
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check format
            if '|' in line:
                parts = line.split('|')
                word = parts[0].strip()
                context = parts[1].strip() if len(parts) > 1 else word
                
                lemma, confidence = lemmatizer.lemmatize_word(word, context)
                
                results.append({
                    'input': word,
                    'context': context,
                    'lemma': lemma,
                    'confidence': confidence
                })
            else:
                # Treat as sentence
                sentence_results = lemmatizer.lemmatize_sentence(line)
                for word, lemma, confidence in sentence_results:
                    results.append({
                        'input': word,
                        'context': line,
                        'lemma': lemma,
                        'confidence': confidence
                    })
    
    # Save results
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"Processed {len(results)} words")


def main(args):
    """Main inference function."""
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load vocabulary
    print("Loading vocabulary...")
    vocab = CharacterVocabulary.load(config['vocab_path'])
    
    # Create model
    print("Creating model...")
    model = ContextAwareLemmatizer(
        char_vocab_size=len(vocab),
        char_embedding_dim=config['model']['char_embedding_dim'],
        char_hidden_dim=config['model']['char_hidden_dim'],
        char_num_layers=config['model']['char_num_layers'],
        bert_model_name=config['model']['bert_model_name'],
        bert_freeze_layers=config['model']['bert_freeze_layers'],
        fusion_output_dim=config['model']['fusion_output_dim'],
        decoder_hidden_dim=config['model']['decoder_hidden_dim'],
        decoder_num_layers=config['model']['decoder_num_layers'],
        dropout=config['model']['dropout'],
        use_copy=config['model']['use_copy']
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully\n")
    
    # Create lemmatizer
    lemmatizer = LemmatizerInference(model, vocab, device)
    
    # Choose mode
    if args.input_file:
        batch_mode(lemmatizer, args.input_file, args.output_file)
    else:
        interactive_mode(lemmatizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lemmatize Azerbaijani text')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Input file for batch processing (optional)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='lemmatization_results.json',
        help='Output file for batch processing'
    )
    
    args = parser.parse_args()
    main(args)