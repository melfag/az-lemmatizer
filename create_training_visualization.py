#!/usr/bin/env python3
"""Create training visualization and detailed analysis"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load training history
history_file = Path("checkpoints/full_training_15epochs/training_history.json")
with open(history_file, 'r') as f:
    history = json.load(f)

train_losses = history['train_loss']
val_losses = history['val_loss']
learning_rates = history['learning_rate']

epochs = list(range(1, len(train_losses) + 1))

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Training and Validation Loss
ax1 = axes[0]
ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Progress - 15 Epochs (450K train, 50K val)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

# Mark best model
best_epoch = val_losses.index(min(val_losses)) + 1
best_val_loss = min(val_losses)
ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Model (Epoch {best_epoch})')
ax1.text(best_epoch, best_val_loss, f'  Best: {best_val_loss:.4f}',
         fontsize=10, verticalalignment='bottom')

# Plot 2: Learning Rate Schedule
ax2 = axes[1]
ax2.plot(epochs, learning_rates, 'g-^', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Learning Rate', fontsize=12)
ax2.set_title('Learning Rate Schedule (Linear Warmup + Decay)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('checkpoints/full_training_15epochs/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved training curves to: checkpoints/full_training_15epochs/training_curves.png")

# Print detailed statistics
print("\n" + "="*70)
print("TRAINING STATISTICS")
print("="*70)

print(f"\nInitial Performance (Epoch 1):")
print(f"  Train Loss: {train_losses[0]:.4f}")
print(f"  Val Loss:   {val_losses[0]:.4f}")

print(f"\nFinal Performance (Epoch 15):")
print(f"  Train Loss: {train_losses[-1]:.4f}")
print(f"  Val Loss:   {val_losses[-1]:.4f}")

print(f"\nBest Performance (Epoch {best_epoch}):")
print(f"  Train Loss: {train_losses[best_epoch-1]:.4f}")
print(f"  Val Loss:   {val_losses[best_epoch-1]:.4f}")

print(f"\nTotal Improvement:")
print(f"  Train Loss: {train_losses[0] - train_losses[-1]:.4f} ({(train_losses[0] - train_losses[-1])/train_losses[0]*100:.1f}% reduction)")
print(f"  Val Loss:   {val_losses[0] - val_losses[-1]:.4f} ({(val_losses[0] - val_losses[-1])/val_losses[0]*100:.1f}% reduction)")

print(f"\nLearning Rate:")
print(f"  Initial: {learning_rates[0]:.2e}")
print(f"  Final:   {learning_rates[-1]:.2e}")

print(f"\nConvergence Analysis:")
# Check when train/val losses stabilized (< 0.01 change for 3 consecutive epochs)
for i in range(2, len(train_losses)):
    if (abs(train_losses[i] - train_losses[i-1]) < 0.01 and
        abs(train_losses[i-1] - train_losses[i-2]) < 0.01):
        print(f"  Training loss stabilized around epoch {i-1}")
        break

for i in range(2, len(val_losses)):
    if (abs(val_losses[i] - val_losses[i-1]) < 0.01 and
        abs(val_losses[i-1] - val_losses[i-2]) < 0.01):
        print(f"  Validation loss stabilized around epoch {i-1}")
        break

print(f"\nOverfitting Check:")
final_gap = abs(train_losses[-1] - val_losses[-1])
print(f"  Train-Val Gap: {final_gap:.4f}")
if final_gap < 0.01:
    print(f"  Status: ✓ No overfitting (excellent generalization)")
elif final_gap < 0.05:
    print(f"  Status: ✓ Minimal overfitting (good generalization)")
else:
    print(f"  Status: ⚠ Some overfitting detected")

print("\n" + "="*70)
