#!/usr/bin/env python3
"""
Automated AI verification using Gemini CLI for 100k examples.

Usage:
    python scripts/auto_verify_gemini.py \
        --input data/processed/moraz_900k.json \
        --output-dir data/processed/ai_verified \
        --num-examples 100000
"""

import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import Counter
import sys

def create_batches(data: List[Dict], num_examples: int, batch_size: int, seed: int = 42) -> List[List[Dict]]:
    """Create batches from randomly sampled data."""
    random.seed(seed)
    sampled_data = random.sample(data, min(num_examples, len(data)))

    batches = []
    for i in range(0, len(sampled_data), batch_size):
        batch = sampled_data[i:i + batch_size]
        batches.append(batch)

    return batches

def create_verification_prompt(batch: List[Dict], batch_num: int) -> str:
    """Create prompt for Gemini."""
    examples_text = "\n".join([
        f"{i+1}. Word: '{ex['word']}', Lemma: '{ex['lemma']}', Context: '{ex.get('context', '')[:100]}'"
        for i, ex in enumerate(batch)
    ])

    return f"""Review Azerbaijani word-lemma pairs. Respond ONLY with JSON array, no markdown.

RULES:
1. Verbs: Remove suffixes (gəlirəm → gəl, oxuyurdum → oxu)
2. Nouns: Remove case/plural/possessive (kitablar → kitab)
3. Pronouns: Standard forms (onun → o)
4. Adjectives: Usually unchanged
5. Proper nouns: Keep as-is

BATCH {batch_num}:
{examples_text}

JSON OUTPUT:
[
  {{"index": 1, "correct": true, "corrected_lemma": null, "reason": ""}},
  {{"index": 2, "correct": false, "corrected_lemma": "oxu", "reason": "Over-stemmed"}},
  ...
]"""

def call_gemini(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini CLI with retry."""
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['gemini', 'ask', prompt],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                return result.stdout.strip()

            error = result.stderr.strip()
            if 'rate limit' in error.lower() or 'quota' in error.lower():
                wait = 60 * (2 ** attempt)
                print(f"    ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                time.sleep(5 * (attempt + 1))

        except subprocess.TimeoutExpired:
            print(f"    ⚠️  Timeout (attempt {attempt + 1})")
            time.sleep(10)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            time.sleep(10)

    return None

def parse_response(text: str) -> List[Dict]:
    """Parse Gemini JSON response."""
    if not text:
        return None

    # Remove markdown
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    # Find JSON array
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        text = text[start:end+1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def apply_corrections(batch: List[Dict], corrections: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Apply corrections to batch."""
    corrected_batch = []
    stats = Counter()
    corrections_map = {c['index']: c for c in corrections}

    for i, ex in enumerate(batch):
        correction = corrections_map.get(i + 1)

        if not correction:
            corrected_batch.append(ex)
            stats['no_info'] += 1
            continue

        if correction['correct']:
            corrected_batch.append(ex)
            stats['correct'] += 1
        else:
            new_lemma = correction.get('corrected_lemma')
            if new_lemma and new_lemma != ex['lemma']:
                corrected_ex = {
                    **ex,
                    'lemma': new_lemma,
                    'ai_corrected': True,
                    'ai_reason': correction.get('reason', ''),
                    'original_lemma': ex['lemma']
                }
                corrected_batch.append(corrected_ex)
                stats['corrected'] += 1
            else:
                corrected_batch.append(ex)
                stats['no_correction'] += 1

    return corrected_batch, dict(stats)

def save_checkpoint(batch: List[Dict], batch_num: int, checkpoint_dir: Path, stats: Dict):
    """Save checkpoint."""
    checkpoint_file = checkpoint_dir / f"checkpoint_{batch_num:06d}.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump({'batch_num': batch_num, 'stats': stats, 'data': batch}, f, ensure_ascii=False, indent=2)

def load_progress(checkpoint_dir: Path) -> int:
    """Get last completed batch."""
    files = list(checkpoint_dir.glob("checkpoint_*.json"))
    if not files:
        return 0
    return max(int(f.stem.split('_')[1]) for f in files)

def combine_checkpoints(checkpoint_dir: Path, output_file: Path) -> Dict:
    """Combine all checkpoints."""
    files = sorted(checkpoint_dir.glob("checkpoint_*.json"))
    combined_data = []
    combined_stats = Counter()

    for f in files:
        with open(f, 'r') as fp:
            checkpoint = json.load(fp)
        combined_data.extend(checkpoint['data'])
        for k, v in checkpoint['stats'].items():
            combined_stats[k] += v

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    return dict(combined_stats)

def main():
    parser = argparse.ArgumentParser(description='AI verification (100k examples)')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='data/processed/ai_verified')
    parser.add_argument('--num-examples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--delay', type=int, default=3)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("AUTOMATED AI VERIFICATION (100K SAMPLE)")
    print("=" * 80)

    # Check gemini CLI
    try:
        subprocess.run(['gemini', '--version'], capture_output=True, timeout=5, check=True)
    except:
        print("\n❌ Error: 'gemini' CLI not found!")
        print("Install: npm install -g @google/generative-ai-cli")
        sys.exit(1)

    # Load data
    print(f"\nLoading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} examples")

    # Create batches
    batches = create_batches(data, args.num_examples, args.batch_size)
    total = len(batches)
    print(f"\nCreated {total:,} batches ({args.num_examples:,} examples)")

    # Resume check
    start = load_progress(checkpoint_dir) if args.resume else 0
    if start > 0:
        print(f"✓ Resuming from batch {start + 1}")

    # Process
    print("\n" + "=" * 80)
    print("PROCESSING")
    print("=" * 80)
    print(f"Estimated time: {(total - start) * args.delay / 3600:.1f}h\n")

    stats = Counter()
    failed = []
    start_time = time.time()

    for i in range(start, total):
        batch = batches[i]
        progress = ((i + 1) / total) * 100
        elapsed = time.time() - start_time
        eta = (elapsed / (i - start + 1)) * (total - i - 1) if i > start else 0

        print(f"[{i+1}/{total}] ({progress:.1f}%) ETA: {eta/3600:.1f}h")

        prompt = create_verification_prompt(batch, i + 1)
        response = call_gemini(prompt)

        if not response:
            print(f"  ❌ No response")
            failed.append(i + 1)
            save_checkpoint(batch, i + 1, checkpoint_dir, {'failed': 1})
            time.sleep(args.delay * 2)
            continue

        corrections = parse_response(response)
        if not corrections:
            print(f"  ❌ Parse error")
            failed.append(i + 1)
            save_checkpoint(batch, i + 1, checkpoint_dir, {'failed': 1})
            time.sleep(args.delay)
            continue

        corrected, batch_stats = apply_corrections(batch, corrections)
        save_checkpoint(corrected, i + 1, checkpoint_dir, batch_stats)

        for k, v in batch_stats.items():
            stats[k] += v

        print(f"  ✓ Correct: {batch_stats.get('correct', 0)}, Corrected: {batch_stats.get('corrected', 0)}")
        time.sleep(args.delay)

        if (i + 1) % 100 == 0:
            print(f"\n  💾 Checkpoint: {i+1}/{total}\n")

    # Combine
    print("\n" + "=" * 80)
    print("COMBINING")
    print("=" * 80)

    output_file = output_dir / f'verified_{args.num_examples}.json'
    final_stats = combine_checkpoints(checkpoint_dir, output_file)

    # Summary
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Time: {(time.time() - start_time)/3600:.1f}h")
    print(f"Failed: {len(failed)}")
    print(f"Correct: {stats.get('correct', 0):,}")
    print(f"Corrected: {stats.get('corrected', 0):,}")

    if stats.get('corrected', 0) > 0:
        error_rate = (stats['corrected'] / (stats.get('correct', 0) + stats.get('corrected', 0))) * 100
        print(f"Error rate: {error_rate:.2f}%")

    print(f"\n✅ Saved: {output_file}")

if __name__ == "__main__":
    main()
