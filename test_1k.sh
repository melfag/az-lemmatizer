#!/bin/bash
# Test on 1K examples
# Expected time: 1-2 minutes

echo "=================================================="
echo "Testing on 1K examples (MorAz Validation)"
echo "=================================================="
echo ""
echo "Expected time: 1-2 minutes"
echo "Expected accuracy: ~61-62%"
echo ""

# Activate environment
source venv/bin/activate

# Run evaluation
python scripts/evaluate.py \
  --checkpoint checkpoints/full_training_15epochs/best_model.pt \
  --test-data data/processed/moraz_500k_val_1000.json \
  --config configs/full_training_15epochs.yaml \
  --batch-size 64 \
  --output-dir evaluation_results/moraz_val_1k \
  --save-predictions

echo ""
echo "=================================================="
echo "1K Test Complete!"
echo "=================================================="
echo ""
echo "Results saved to: evaluation_results/moraz_val_1k"
echo ""
echo "View metrics:"
echo "  cat evaluation_results/moraz_val_1k/test_metrics.json"
echo ""
