import torch
from typing import Optional

def dice_loss(inputs, targets, smooth=1):      
    """
    dice loss for the binary case (one category)
    """
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
    return 1 - dice