#!/usr/bin/env python3
"""Quick script to analyze training progress from checkpoints"""

import torch
from pathlib import Path

checkpoint_dir = Path("checkpoints/full_training_15epochs")
checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))

print("=" * 70)
print("Training Progress Analysis - 15 Epoch Training")
print("=" * 70)

for ckpt_file in checkpoint_files:
    try:
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        history = checkpoint.get('history', {})

        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        learning_rates = history.get('learning_rate', [])

        # Get current epoch's metrics (last in history)
        if train_losses and val_losses:
            train_loss = train_losses[-1]
            val_loss = val_losses[-1]
            lr = learning_rates[-1] if learning_rates else 0

            print(f"\nEpoch {epoch + 1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")

    except Exception as e:
        print(f"Error reading {ckpt_file.name}: {e}")

# Load best model info
try:
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location='cpu')
    best_epoch = best_checkpoint['epoch']
    best_val_loss = best_checkpoint['best_val_loss']

    print("\n" + "=" * 70)
    print(f"Best Model: Epoch {best_epoch + 1} | Val Loss: {best_val_loss:.4f}")
    print("=" * 70)

    # Show full training history
    history = best_checkpoint.get('history', {})
    train_losses = history.get('train_loss', [])
    val_losses = history.get('val_loss', [])

    if train_losses and val_losses:
        print("\nTraining History Summary:")
        print("-" * 70)
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            marker = " ← BEST" if i == best_epoch else ""
            print(f"Epoch {i+1:2d}: Train Loss: {tl:.4f} | Val Loss: {vl:.4f}{marker}")

except Exception as e:
    print(f"Error reading best_model.pt: {e}")

print("\n" + "=" * 70)
print("Checkpoint Files Summary:")
print("-" * 70)
print(f"Total epochs completed: {len(checkpoint_files)}")
print(f"Epochs remaining: {15 - len(checkpoint_files)}")
print("=" * 70)
