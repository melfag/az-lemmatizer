"""
Analyze what post-processing fixed
"""
import json
import argparse


def analyze_improvements(raw_file, processed_file):
    """Compare raw vs processed predictions"""

    with open(raw_file) as f:
        raw = json.load(f)

    with open(processed_file) as f:
        processed = json.load(f)

    # Find cases where post-processing helped
    improvements = []
    regressions = []

    for r, p in zip(raw, processed):
        if not r['correct'] and p['correct']:
            # Post-processing fixed an error
            improvements.append({
                'word': r['word'],
                'raw_prediction': r['predicted'],
                'processed_prediction': p['predicted'],
                'target': r['target'],
                'improvement_type': classify_improvement(r, p)
            })
        elif r['correct'] and not p['correct']:
            # Post-processing broke something
            regressions.append({
                'word': r['word'],
                'raw_prediction': r['predicted'],
                'processed_prediction': p['predicted'],
                'target': r['target']
            })

    # Categorize improvements
    by_type = {}
    for imp in improvements:
        typ = imp['improvement_type']
        by_type[typ] = by_type.get(typ, 0) + 1

    # Print summary
    print("\n" + "="*60)
    print("POST-PROCESSING ANALYSIS")
    print("="*60)
    print(f"\nTotal examples: {len(raw)}")
    print(f"Improvements: {len(improvements)}")
    print(f"Regressions: {len(regressions)}")
    print(f"Net gain: {len(improvements) - len(regressions)}")

    if improvements:
        print("\n" + "-"*60)
        print("IMPROVEMENTS BY TYPE:")
        print("-"*60)
        for typ, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(improvements) * 100
            print(f"  {typ}: {count} ({pct:.1f}%)")

        print("\n" + "-"*60)
        print("EXAMPLE IMPROVEMENTS:")
        print("-"*60)
        for i, imp in enumerate(improvements[:15], 1):
            print(f"\n{i}. Word: {imp['word']}")
            print(f"   Raw:  {imp['raw_prediction']} ❌")
            print(f"   Fixed: {imp['processed_prediction']} ✓")
            print(f"   Target: {imp['target']}")
            print(f"   Type: {imp['improvement_type']}")

    if regressions:
        print("\n" + "-"*60)
        print("REGRESSIONS (Post-processing broke these):")
        print("-"*60)
        for i, reg in enumerate(regressions[:10], 1):
            print(f"\n{i}. Word: {reg['word']}")
            print(f"   Raw (correct): {reg['raw_prediction']} ✓")
            print(f"   Processed (wrong): {reg['processed_prediction']} ❌")
            print(f"   Target: {reg['target']}")

    print("\n" + "="*60)

    return {
        'total': len(raw),
        'improvements': len(improvements),
        'regressions': len(regressions),
        'net_gain': len(improvements) - len(regressions),
        'by_type': by_type
    }


def classify_improvement(raw, processed):
    """Classify type of improvement"""
    raw_pred = raw['predicted']
    proc_pred = processed['predicted']
    word = raw['word']

    # Check if it's case normalization
    if raw_pred.lower() == proc_pred and raw_pred != proc_pred:
        return 'Case normalization'

    # Check if suffix was removed
    elif len(raw_pred) > len(proc_pred):
        removed = raw_pred[len(proc_pred):]
        return f'Suffix removal (-{removed})'

    # Check if model just copied input
    elif raw_pred == word or raw_pred == word.lower():
        return 'Identity → Lemma (rule-based)'

    else:
        return 'Other transformation'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='evaluation_results/postprocessed_ud/results_raw.json',
                        help='Path to raw results JSON')
    parser.add_argument('--processed', default='evaluation_results/postprocessed_ud/results_processed.json',
                        help='Path to processed results JSON')

    args = parser.parse_args()

    analyze_improvements(args.raw, args.processed)
