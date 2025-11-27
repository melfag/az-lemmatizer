"""
Compare results across multiple evaluation runs
"""
import json
import glob
from pathlib import Path
import argparse


def compare_evaluations(base_dir='evaluation_results', output_file=None):
    """
    Compare all evaluations in base directory.

    Args:
        base_dir: Base directory containing evaluation results
        output_file: Optional file to save comparison table
    """
    results = []

    # Find all metrics files
    metrics_files = glob.glob(f'{base_dir}/*/test_metrics.json')

    if not metrics_files:
        print(f"No evaluation results found in {base_dir}")
        return

    print(f"\nFound {len(metrics_files)} evaluation results")

    # Load all results
    for metrics_file in metrics_files:
        dataset_name = Path(metrics_file).parent.name

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Count predictions if available
            predictions_file = Path(metrics_file).parent / 'test_predictions.json'
            num_examples = 0
            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    predictions = json.load(f)
                    num_examples = len(predictions)

            results.append({
                'dataset': dataset_name,
                'num_examples': num_examples,
                'accuracy': metrics.get('accuracy', 0),
                'edit_distance': metrics.get('avg_edit_distance', 0),
                'char_precision': metrics.get('precision', 0),
                'char_recall': metrics.get('recall', 0),
                'char_f1': metrics.get('f1', 0)
            })
        except Exception as e:
            print(f"  Warning: Could not load {metrics_file}: {e}")
            continue

    if not results:
        print("No valid results found")
        return

    # Sort by accuracy (descending)
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    # Print comparison table
    print("\n" + "=" * 120)
    print("Evaluation Results Comparison")
    print("=" * 120)
    print(f"{'Dataset':<35} {'Examples':<12} {'Accuracy':<12} {'Edit Dist':<12} "
          f"{'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 120)

    output_lines = []
    for r in results:
        line = (f"{r['dataset']:<35} "
                f"{r['num_examples']:>10}  "
                f"{r['accuracy']:>9.2f}%  "
                f"{r['edit_distance']:>9.2f}  "
                f"{r['char_precision']:>9.2f}%  "
                f"{r['char_recall']:>9.2f}%  "
                f"{r['char_f1']:>9.2f}%")
        print(line)
        output_lines.append(line)

    print("=" * 120)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 120)
    accuracies = [r['accuracy'] for r in results]
    print(f"  Average Accuracy: {sum(accuracies)/len(accuracies):.2f}%")
    print(f"  Best Accuracy:    {max(accuracies):.2f}% ({results[0]['dataset']})")
    print(f"  Worst Accuracy:   {min(accuracies):.2f}% ({results[-1]['dataset']})")

    edit_dists = [r['edit_distance'] for r in results]
    print(f"  Average Edit Distance: {sum(edit_dists)/len(edit_dists):.2f}")

    print("=" * 120)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write("Evaluation Results Comparison\n")
            f.write("=" * 120 + "\n")
            f.write(f"{'Dataset':<35} {'Examples':<12} {'Accuracy':<12} {'Edit Dist':<12} "
                   f"{'Precision':<12} {'Recall':<12} {'F1':<12}\n")
            f.write("-" * 120 + "\n")
            for line in output_lines:
                f.write(line + "\n")
            f.write("=" * 120 + "\n")
        print(f"\n✅ Comparison saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare evaluation results')
    parser.add_argument(
        '--base-dir',
        type=str,
        default='evaluation_results',
        help='Base directory containing evaluation results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save comparison table'
    )

    args = parser.parse_args()
    compare_evaluations(args.base_dir, args.output)


if __name__ == '__main__':
    main()
