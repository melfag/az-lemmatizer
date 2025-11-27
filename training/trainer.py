"""
Main training loop with early stopping and checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os
from tqdm import tqdm
import json
from pathlib import Path

from training.losses import CompositeLoss
from training.optimizer import get_optimizer, get_parameter_count
from training.scheduler import get_linear_schedule_with_warmup


class Trainer:
    """
    Trainer class for the lemmatizer model.
    
    Based on Section 4.4 of the thesis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Args:
            model: The lemmatizer model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on (cuda/mps/cpu)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = CompositeLoss(
            char_hidden_dim=config.get('char_hidden_dim', 512),
            smoothing=config.get('label_smoothing', 0.1),
            char_classification_weight=config.get('char_class_weight', 0.1),
            attention_coverage_weight=config.get('attention_coverage_weight', 0.05),
            transformation_weight=config.get('transformation_loss_weight', 0.0)
        )
        
        # Optimizer
        self.optimizer = get_optimizer(
            model=model,
            bert_lr=config.get('bert_lr', 1e-5),
            other_lr=config.get('other_lr', 3e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        num_training_steps = len(train_dataloader) * config.get('num_epochs', 15)
        num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.get('early_stopping_patience', 5)
        self._checkpoint_loaded = False  # Track if we resumed from checkpoint
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Teacher forcing with dynamic schedule
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
        self.teacher_forcing_schedule = config.get('teacher_forcing_schedule', 'constant')
        self.teacher_forcing_end_ratio = config.get('teacher_forcing_end_ratio', 0.0)

        print(f"Teacher forcing: {self.teacher_forcing_ratio} → {self.teacher_forcing_end_ratio}")
        print(f"Schedule: {self.teacher_forcing_schedule}")

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Print parameter count
        param_count = get_parameter_count(model)
        print(f"\nModel Parameters:")
        print(f"  Trainable: {param_count['trainable']:,}")
        print(f"  Frozen: {param_count['frozen']:,}")
        print(f"  Total: {param_count['total']:,}\n")

    def get_teacher_forcing_ratio(self, epoch: int, num_epochs: int) -> float:
        """
        Calculate teacher forcing ratio for current epoch

        Args:
            epoch: Current epoch
            num_epochs: Total epochs

        Returns:
            Teacher forcing ratio (0-1)
        """
        if self.teacher_forcing_schedule == 'constant':
            return self.teacher_forcing_ratio

        elif self.teacher_forcing_schedule == 'linear':
            # Linear decay
            progress = epoch / num_epochs
            ratio = self.teacher_forcing_ratio - progress * (
                self.teacher_forcing_ratio - self.teacher_forcing_end_ratio
            )
            return max(self.teacher_forcing_end_ratio, ratio)

        elif self.teacher_forcing_schedule == 'exponential':
            # Exponential decay
            decay_rate = 0.95
            ratio = self.teacher_forcing_ratio * (decay_rate ** epoch)
            return max(self.teacher_forcing_end_ratio, ratio)

        else:
            return self.teacher_forcing_ratio

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with average losses
        """
        self.model.train()

        # Get teacher forcing ratio for this epoch
        tf_ratio = self.get_teacher_forcing_ratio(
            self.current_epoch,
            self.config.get('num_epochs', 15)
        )

        epoch_losses = {
            'total_loss': 0.0,
            'main_loss': 0.0,
            'char_classification_loss': 0.0,
            'attention_coverage_loss': 0.0,
            'transformation_loss': 0.0
        }

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1} (TF={tf_ratio:.2f})")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_char_ids = batch['input_char_ids'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            target_char_ids = batch['target_char_ids'].to(self.device)
            context_sentences = batch['context_sentences']
            target_positions_tensor = batch['target_positions']

            # Debug: Check what DataLoader returned
            if batch_idx == 0:
                print(f"\nDEBUG BEFORE conversion:")
                print(f"  type: {type(target_positions_tensor)}")
                if isinstance(target_positions_tensor, list):
                    print(f"  list length: {len(target_positions_tensor)}")
                    print(f"  first element: {target_positions_tensor[0]}")
                elif hasattr(target_positions_tensor, 'shape'):
                    print(f"  tensor shape: {target_positions_tensor.shape}")

            # Convert target_positions from DataLoader format to list of tuples
            # DataLoader returns list of 2 tensors: [start_positions, end_positions]
            # where each tensor has shape (batch_size,)
            if isinstance(target_positions_tensor, list) and len(target_positions_tensor) == 2:
                # Format: [tensor([start1, start2, ...]), tensor([end1, end2, ...])]
                starts = target_positions_tensor[0]
                ends = target_positions_tensor[1]
                target_positions = [(int(starts[i]), int(ends[i])) for i in range(len(starts))]
            elif hasattr(target_positions_tensor, 'shape') and len(target_positions_tensor.shape) == 2:
                # Format: tensor([[start, end], [start, end], ...])  shape: (batch_size, 2)
                target_positions = [(int(target_positions_tensor[i][0]), int(target_positions_tensor[i][1]))
                                   for i in range(target_positions_tensor.shape[0])]
            else:
                # Already a list of tuples (shouldn't happen but just in case)
                target_positions = target_positions_tensor

            # Debug: Check data shapes
            if batch_idx == 0:
                print(f"\nDEBUG Batch 0:")
                print(f"  context_sentences type: {type(context_sentences)}")
                print(f"  context_sentences length: {len(context_sentences)}")
                print(f"  target_positions length: {len(target_positions)}")
                if hasattr(context_sentences, 'shape'):
                    print(f"  context_sentences shape: {context_sentences.shape}")

            # Optional: stem labels for auxiliary loss
            is_stem = batch.get('is_stem', None)
            if is_stem is not None:
                is_stem = is_stem.to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_char_ids=input_char_ids,
                input_lengths=input_lengths,
                context_sentences=context_sentences,
                target_word_positions=target_positions,
                target_char_ids=target_char_ids,
                teacher_forcing_ratio=tf_ratio  # Use dynamic ratio
            )

            # Compute loss
            loss, loss_dict = self.criterion(
                predictions=outputs['outputs'],
                targets=target_char_ids,
                char_hidden_states=None,  # Can be added if needed
                is_stem=is_stem,
                attention_weights=outputs['attentions'],
                inputs=input_char_ids  # Add inputs for transformation loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                epoch_losses[key] += value
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Average losses
        num_batches = len(self.train_dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with average validation losses
        """
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'main_loss': 0.0,
            'char_classification_loss': 0.0,
            'attention_coverage_loss': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                input_char_ids = batch['input_char_ids'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                target_char_ids = batch['target_char_ids'].to(self.device)
                context_sentences = batch['context_sentences']
                target_positions_tensor = batch['target_positions']

                # Convert target_positions from DataLoader format to list of tuples
                # DataLoader returns list of 2 tensors: [start_positions, end_positions]
                if isinstance(target_positions_tensor, list) and len(target_positions_tensor) == 2:
                    # Format: [tensor([start1, start2, ...]), tensor([end1, end2, ...])]
                    starts = target_positions_tensor[0]
                    ends = target_positions_tensor[1]
                    target_positions = [(int(starts[i]), int(ends[i])) for i in range(len(starts))]
                elif hasattr(target_positions_tensor, 'shape') and len(target_positions_tensor.shape) == 2:
                    # Format: tensor([[start, end], [start, end], ...])  shape: (batch_size, 2)
                    target_positions = [(int(target_positions_tensor[i][0]), int(target_positions_tensor[i][1]))
                                       for i in range(target_positions_tensor.shape[0])]
                else:
                    # Already a list of tuples (shouldn't happen but just in case)
                    target_positions = target_positions_tensor
                
                is_stem = batch.get('is_stem', None)
                if is_stem is not None:
                    is_stem = is_stem.to(self.device)
                
                # Forward pass (no teacher forcing during validation)
                outputs = self.model(
                    input_char_ids=input_char_ids,
                    input_lengths=input_lengths,
                    context_sentences=context_sentences,
                    target_word_positions=target_positions,
                    target_char_ids=target_char_ids,
                    teacher_forcing_ratio=0.0  # No teacher forcing
                )
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    predictions=outputs['outputs'],
                    targets=target_char_ids,
                    char_hidden_states=None,
                    is_stem=is_stem,
                    attention_weights=outputs['attentions']
                )
                
                # Update metrics
                for key, value in loss_dict.items():
                    val_losses[key] += value
        
        # Average losses
        num_batches = len(self.val_dataloader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def train(self, num_epochs: int = None):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train (overrides config)
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 15)

        # Determine starting epoch based on whether we resumed from checkpoint
        if self._checkpoint_loaded:
            start_epoch = self.current_epoch + 1  # Continue from next epoch
            print(f"Resuming training from epoch {start_epoch + 1}/{num_epochs}...")
        else:
            start_epoch = 0  # Fresh start
            print(f"Starting training for {num_epochs} epochs...")

        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        print(f"Validation samples: {len(self.val_dataloader.dataset)}")
        print(f"Batch size: {self.train_dataloader.batch_size}")
        print(f"Gradient accumulation: {self.config.get('gradient_accumulation_steps', 1)}\n")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['val_loss'].append(val_losses['total_loss'])
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_losses['total_loss']:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save checkpoint if best
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                print(f"  ✓ New best model! (Val Loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.patience})")
            
            # Regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            print()
        
        # Save final checkpoint and history
        self.save_checkpoint(is_best=False, filename='final_model.pt')
        self.save_history()
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, is_best: bool = False, filename: str = None):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model
            filename: Custom filename (optional)
        """
        if filename is None:
            filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{self.current_epoch + 1}.pt'
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"  Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        self._checkpoint_loaded = True  # Mark that we resumed from checkpoint

        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_file = self.checkpoint_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history: {history_file}")