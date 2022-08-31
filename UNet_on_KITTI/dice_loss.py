import torch
from typing import Optional

def dice_loss(inputs, targets, smooth=1):
        
    #comment out if your model contains a sigmoid or equivalent activation layer
    #inputs = F.sigmoid(inputs)       
        
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
    return 1 - dice