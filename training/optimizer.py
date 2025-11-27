"""
Optimizer configuration with differential learning rates
"""

import torch
from torch.optim import AdamW


def get_optimizer(
    model,
    bert_lr: float = 1e-5,
    other_lr: float = 3e-4,
    weight_decay: float = 1e-5
) -> AdamW:
    """
    Create AdamW optimizer with differential learning rates.
    Lower learning rate for pre-trained BERT, higher for new components.
    
    Based on Section 4.4.1 of the thesis.
    
    Args:
        model: The lemmatizer model
        bert_lr: Learning rate for BERT parameters
        other_lr: Learning rate for other parameters
        weight_decay: Weight decay coefficient
        
    Returns:
        AdamW optimizer
    """
    # Separate BERT parameters from others
    bert_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'contextual_encoder.bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups with different learning rates
    optimizer = AdamW([
        {
            'params': bert_params,
            'lr': bert_lr,
            'weight_decay': weight_decay
        },
        {
            'params': other_params,
            'lr': other_lr,
            'weight_decay': weight_decay
        }
    ])
    
    return optimizer


def get_parameter_count(model) -> dict:
    """
    Count trainable and frozen parameters.
    
    Args:
        model: The model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    
    return {
        'trainable': trainable,
        'frozen': frozen,
        'total': total
    }