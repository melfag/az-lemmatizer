#!/bin/bash
# Compare all evaluation results
# Run this AFTER all tests are complete

echo "=================================================="
echo "Comparing All Evaluation Results"
echo "=================================================="
echo ""

# Activate environment
source venv/bin/activate

# Run comparison
python scripts/compare_evaluations.py \
  --base-dir evaluation_results \
  --output evaluation_results/full_comparison.txt

echo ""
echo "Detailed comparison saved to:"
echo "  evaluation_results/full_comparison.txt"
echo ""

# Show quick summary
echo "=================================================="
echo "Quick Summary"
echo "=================================================="
echo ""

# Check which tests completed
echo "Tests completed:"
[ -f "evaluation_results/ud_test/test_metrics.json" ] && echo "  ✓ UD Test (726 examples)"
[ -f "evaluation_results/moraz_val_1k/test_metrics.json" ] && echo "  ✓ MorAz 1K"
[ -f "evaluation_results/moraz_val_5k/test_metrics.json" ] && echo "  ✓ MorAz 5K"
[ -f "evaluation_results/moraz_val_10k/test_metrics.json" ] && echo "  ✓ MorAz 10K"
[ -f "evaluation_results/moraz_val_full_50k/test_metrics.json" ] && echo "  ✓ MorAz 50K (FULL)"

echo ""
echo "Individual metrics:"
echo ""

# Show each result if exists
if [ -f "evaluation_results/ud_test/test_metrics.json" ]; then
    ACC=$(cat evaluation_results/ud_test/test_metrics.json | grep -o '"accuracy": [0-9.]*' | grep -o '[0-9.]*')
    echo "  UD Test:      ${ACC}%"
fi

if [ -f "evaluation_results/moraz_val_1k/test_metrics.json" ]; then
    ACC=$(cat evaluation_results/moraz_val_1k/test_metrics.json | grep -o '"accuracy": [0-9.]*' | grep -o '[0-9.]*')
    echo "  MorAz 1K:     ${ACC}%"
fi

if [ -f "evaluation_results/moraz_val_5k/test_metrics.json" ]; then
    ACC=$(cat evaluation_results/moraz_val_5k/test_metrics.json | grep -o '"accuracy": [0-9.]*' | grep -o '[0-9.]*')
    echo "  MorAz 5K:     ${ACC}%"
fi

if [ -f "evaluation_results/moraz_val_10k/test_metrics.json" ]; then
    ACC=$(cat evaluation_results/moraz_val_10k/test_metrics.json | grep -o '"accuracy": [0-9.]*' | grep -o '[0-9.]*')
    echo "  MorAz 10K:    ${ACC}%"
fi

if [ -f "evaluation_results/moraz_val_full_50k/test_metrics.json" ]; then
    ACC=$(cat evaluation_results/moraz_val_full_50k/test_metrics.json | grep -o '"accuracy": [0-9.]*' | grep -o '[0-9.]*')
    echo "  MorAz 50K:    ${ACC}%"
fi

echo ""
echo "=================================================="
echo ""
