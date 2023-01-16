import numpy as np
import torch
import sklearn as sk

def dice_score_one(inputs, targets, smooth=1e-5):      
    """
    dice loss for the binary case (one category)
    """
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
    return dice
