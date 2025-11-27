import json
from collections import Counter

# Load verified data
with open('data/processed/ai_verified/verified_1000.json', 'r') as f:
    data = json.load(f)

# Calculate statistics
stats = Counter()
corrections_by_type = []

for ex in data:
    if 'ai_corrected' in ex and ex['ai_corrected']:
        stats['corrected'] += 1
        corrections_by_type.append({
            'word': ex['word'],
            'old_lemma': ex['original_lemma'],
            'new_lemma': ex['lemma'],
            'reason': ex.get('ai_reason', '')
        })
    else:
        stats['correct'] += 1

total = len(data)
corrected = stats['corrected']
correct = stats['correct']
error_rate = (corrected / total) * 100 if total > 0 else 0

print("=" * 70)
print("VERIFICATION RESULTS (1,000 examples)")
print("=" * 70)
print(f"\nTotal examples:     {total:,}")
print(f"Correct lemmas:     {correct:,} ({correct/total*100:.1f}%)")
print(f"Corrected by AI:    {corrected:,} ({error_rate:.1f}%)")
print(f"\nError rate:         {error_rate:.1f}%")

# Show sample corrections
if corrections_by_type:
    print(f"\n" + "=" * 70)
    print("SAMPLE CORRECTIONS (first 20)")
    print("=" * 70)
    for i, corr in enumerate(corrections_by_type[:20], 1):
        print(f"{i}. '{corr['word']}': {corr['old_lemma']} → {corr['new_lemma']}")
        if corr['reason']:
            print(f"   Reason: {corr['reason']}")

# Analyze correction patterns
old_lemmas = Counter(c['old_lemma'] for c in corrections_by_type)
new_lemmas = Counter(c['new_lemma'] for c in corrections_by_type)

if old_lemmas:
    print(f"\n" + "=" * 70)
    print("MOST FREQUENTLY CORRECTED LEMMAS")
    print("=" * 70)
    for lemma, count in old_lemmas.most_common(10):
        print(f"  '{lemma}': corrected {count} times")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if error_rate < 5:
    print("✅ Error rate < 5% - Data quality is EXCELLENT")
    print("   → Proceed to training immediately")
elif error_rate < 10:
    print("✅ Error rate < 10% - Data quality is GOOD")
    print("   → Can proceed to training")
    print("   → Or apply pattern-based fixes first")
elif error_rate < 20:
    print("⚠️  Error rate 10-20% - Data quality is ACCEPTABLE")
    print("   → Recommended: Apply pattern fixes before training")
    print("   → Or verify more examples")
else:
    print("❌ Error rate > 20% - Data quality needs improvement")
    print("   → Apply systematic pattern fixes")
    print("   → Or improve MorAz annotation quality")

print("\n")
