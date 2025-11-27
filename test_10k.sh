#!/bin/bash
# Test on 10K examples
# Expected time: 15-20 minutes

echo "=================================================="
echo "Testing on 10K examples (MorAz Validation)"
echo "=================================================="
echo ""
echo "Expected time: 15-20 minutes"
echo "Expected accuracy: ~60-62%"
echo ""

# Activate environment
source venv/bin/activate

# Check if sample exists
if [ ! -f "data/processed/moraz_500k_val_10000.json" ]; then
    echo "10K sample not found. Creating it now..."
    python scripts/create_samples.py \
      --input data/processed/moraz_500k_val.json \
      --sizes 10000 \
      --output-dir data/processed
    echo ""
fi

# Run evaluation
python scripts/evaluate.py \
  --checkpoint checkpoints/full_training_15epochs/best_model.pt \
  --test-data data/processed/moraz_500k_val_10000.json \
  --config configs/full_training_15epochs.yaml \
  --batch-size 64 \
  --output-dir evaluation_results/moraz_val_10k \
  --save-predictions

echo ""
echo "=================================================="
echo "10K Test Complete!"
echo "=================================================="
echo ""
echo "Results saved to: evaluation_results/moraz_val_10k"
echo ""
echo "View metrics:"
echo "  cat evaluation_results/moraz_val_10k/test_metrics.json"
echo ""
