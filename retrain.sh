#!/bin/bash

# Improved model retraining script

echo "=================================================="
echo "Starting Improved Model Retraining"
echo "=================================================="
echo ""
echo "Configuration: configs/improved_training.yaml"
echo "Expected time: 2-3 days (~4 hours/epoch x 20 epochs)"
echo "Target accuracy: 60-70% (UD), 80-90% (MorAz)"
echo ""

# Activate environment
source venv/bin/activate

# Run training
python scripts/train.py \
  --config configs/improved_training.yaml

echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
echo ""
echo "Best model: checkpoints/improved_training/best_model.pt"
echo ""
