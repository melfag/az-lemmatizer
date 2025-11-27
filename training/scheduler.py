# Warmup + linear decay

"""
Learning rate scheduler with warmup and linear decay
"""

import math
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    Create a learning rate scheduler with linear warmup and linear decay.
    
    Based on Section 4.4.2 of the thesis.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps (usually 10% of training)
        num_training_steps: Total number of training steps
        last_epoch: The index of the last epoch
        
    Returns:
        LambdaLR scheduler
    """
    
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Decay phase
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    Create a learning rate scheduler with cosine annealing and warmup.
    Alternative to linear decay.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (0.5 = cosine to 0)
        last_epoch: The index of the last epoch
        
    Returns:
        LambdaLR scheduler
    """
    
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / \
                  float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)