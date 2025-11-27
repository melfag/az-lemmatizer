#!/bin/bash
# Test on FULL 50K examples
# Expected time: 30-60 minutes
# WARNING: This is the long one!

echo "=================================================="
echo "Testing on FULL 50K examples (MorAz Validation)"
echo "=================================================="
echo ""
echo "⚠️  WARNING: This will take 30-60 minutes!"
echo ""
echo "Expected time: 30-60 minutes"
echo "Expected accuracy: ~60-62%"
echo "Output file size: ~50MB with predictions"
echo ""
echo "You can monitor progress in real-time."
echo "Press Ctrl+C to cancel if needed."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting full 50K evaluation..."
echo "Started at: $(date)"
echo ""

# Activate environment
source venv/bin/activate

# Run evaluation (with predictions saved)
python scripts/evaluate.py \
  --checkpoint checkpoints/full_training_15epochs/best_model.pt \
  --test-data data/processed/moraz_500k_val.json \
  --config configs/full_training_15epochs.yaml \
  --batch-size 64 \
  --output-dir evaluation_results/moraz_val_full_50k \
  --save-predictions

echo ""
echo "Finished at: $(date)"
echo ""
echo "=================================================="
echo "50K Test Complete!"
echo "=================================================="
echo ""
echo "Results saved to: evaluation_results/moraz_val_full_50k"
echo ""
echo "View metrics:"
echo "  cat evaluation_results/moraz_val_full_50k/test_metrics.json"
echo ""
echo "Predictions file size:"
ls -lh evaluation_results/moraz_val_full_50k/test_predictions.json 2>/dev/null || echo "  (predictions not saved)"
echo ""
