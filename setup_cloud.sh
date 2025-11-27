#!/bin/bash

echo "=================================================="
echo "Cloud GPU Setup for Azerbaijani Lemmatizer"
echo "=================================================="
echo ""

# Detect environment
if [ -d "/kaggle" ]; then
    PLATFORM="Kaggle"
    WORKSPACE="/kaggle/working"
elif [ -d "/content" ] && [ -n "$COLAB_GPU" ]; then
    PLATFORM="Google Colab"
    WORKSPACE="/content"
elif [ -d "/workspace" ]; then
    PLATFORM="RunPod/Lambda"
    WORKSPACE="/workspace"
else
    PLATFORM="Unknown"
    WORKSPACE=$(pwd)
fi

echo "Detected platform: $PLATFORM"
echo "Working directory: $WORKSPACE"
echo ""

# Check GPU
echo "Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ GPU Found: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("✗ No GPU found - training will be slow!")
EOF

echo ""
echo "Installing dependencies..."

# Install required packages
pip install -q transformers torch pyyaml tqdm 2>/dev/null || pip3 install -q transformers torch pyyaml tqdm

echo "✓ Dependencies installed"
echo ""

# Check if repo exists
if [ ! -d "models" ] || [ ! -d "scripts" ]; then
    echo "Repository not found in current directory."
    echo ""
    echo "Options:"
    echo "1. Clone from GitHub:"
    echo "   git clone <your-repo-url> && cd azerbaijani-lemmatizer"
    echo ""
    echo "2. Upload files manually to: $WORKSPACE"
    echo ""
    exit 1
fi

echo "✓ Project structure found"
echo ""

# Verify data files
echo "Checking data files..."
DATA_DIR="data/processed"

if [ ! -f "$DATA_DIR/moraz_500k_train_filtered.json" ]; then
    echo "⚠ Warning: Training data not found: $DATA_DIR/moraz_500k_train_filtered.json"
    echo "  Please upload data files to: $WORKSPACE/data/processed/"
fi

if [ ! -f "$DATA_DIR/char_vocab.json" ]; then
    echo "⚠ Warning: Vocabulary not found: $DATA_DIR/char_vocab.json"
fi

if [ ! -f "configs/improved_training.yaml" ]; then
    echo "⚠ Warning: Config not found: configs/improved_training.yaml"
fi

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Platform: $PLATFORM"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""
echo "To start training:"
echo "  python scripts/train.py --config configs/improved_training.yaml"
echo ""
echo "For background training (recommended):"
echo "  tmux new -s training"
echo "  python scripts/train.py --config configs/improved_training.yaml"
echo "  # Press Ctrl+B then D to detach"
echo ""
echo "Good luck! 🚀"
