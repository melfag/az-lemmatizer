"""
Evaluate model with post-processing
"""
import sys
import torch
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.lemmatizer import ContextAwareLemmatizer
from utils.vocabulary import CharacterVocabulary
from utils.post_processor import LemmaPostProcessor
from utils.data_loader import create_dataloader
from transformers import AutoTokenizer
from tqdm import tqdm


def evaluate_with_postprocessing(
    checkpoint_path,
    test_data_path,
    config_path,
    vocab_path,
    output_dir,
    batch_size=32
):
    """Evaluate with post-processing"""

    # Load model (reuse existing evaluation code)
    import yaml

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint.get('config', {})

    # Load vocabulary
    vocab = CharacterVocabulary.load(vocab_path)
    bert_tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_model_name'])

    # Create model
    model = ContextAwareLemmatizer(
        char_vocab_size=len(vocab),
        char_vocab=vocab,
        **config['model']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data
    with open(test_data_path) as f:
        test_examples = json.load(f)

    # Create dataloader
    test_loader = create_dataloader(
        test_examples,
        vocab,
        bert_tokenizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize post-processor
    post_processor = LemmaPostProcessor()

    # Evaluate
    results_raw = []  # Without post-processing
    results_processed = []  # With post-processing

    print(f"\nEvaluating {len(test_examples)} examples...")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_char_ids = batch['input_char_ids'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            context_sentences = batch['context_sentences']
            target_positions_tensor = batch['target_positions']

            # Convert target positions
            if isinstance(target_positions_tensor, list) and len(target_positions_tensor) == 2:
                starts, ends = target_positions_tensor
                target_positions = [(int(starts[i]), int(ends[i])) for i in range(len(starts))]
            else:
                target_positions = target_positions_tensor

            # Get predictions
            predictions, _ = model.lemmatize(
                input_char_ids,
                input_lengths,
                context_sentences,
                target_positions,
                max_length=50
            )

            # Decode and post-process
            for i in range(len(batch['word_str'])):
                word = batch['word_str'][i]
                target = batch['lemma_str'][i]

                # Raw prediction
                pred_ids = predictions[i].cpu().tolist()
                pred_raw = vocab.decode(pred_ids, remove_special_tokens=True)

                # Post-processed prediction
                pred_processed = post_processor.post_process(word, pred_raw)

                results_raw.append({
                    'word': word,
                    'predicted': pred_raw,
                    'target': target,
                    'correct': pred_raw == target
                })

                results_processed.append({
                    'word': word,
                    'predicted': pred_processed,
                    'target': target,
                    'correct': pred_processed == target
                })

    # Calculate metrics
    acc_raw = sum(r['correct'] for r in results_raw) / len(results_raw) * 100
    acc_processed = sum(r['correct'] for r in results_processed) / len(results_processed) * 100

    improvement = acc_processed - acc_raw

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nWithout Post-Processing: {acc_raw:.2f}%")
    print(f"With Post-Processing:    {acc_processed:.2f}%")
    print(f"Improvement:             +{improvement:.2f}%")
    print("="*60)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'results_raw.json', 'w') as f:
        json.dump(results_raw, f, ensure_ascii=False, indent=2)

    with open(output_path / 'results_processed.json', 'w') as f:
        json.dump(results_processed, f, ensure_ascii=False, indent=2)

    metrics = {
        'accuracy_raw': acc_raw,
        'accuracy_processed': acc_processed,
        'improvement': improvement
    }

    with open(output_path / 'metrics_comparison.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--vocab', default='data/processed/char_vocab.json')
    parser.add_argument('--output-dir', default='evaluation_results/postprocessed')
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    evaluate_with_postprocessing(
        args.checkpoint,
        args.test_data,
        args.config,
        args.vocab,
        args.output_dir,
        args.batch_size
    )
