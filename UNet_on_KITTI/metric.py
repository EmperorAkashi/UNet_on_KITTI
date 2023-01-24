import numpy as np
import torch
import sklearn as sk
from typing import Optional, Callable
import torch.nn as nn
import segmentation_models_pytorch as smp

def dice_score_one(inputs:torch.Tensor, targets:torch.Tensor, smooth=1e-5) -> torch.Tensor:      
    """
    dice loss for the binary case (one category)
    """
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
    return dice

def m_dice_score(mask:torch.Tensor, pred:torch.Tensor) -> torch.Tensor:
    """
    calculate dice for all classes, then average
    input:
    mask(tensor): labels in one hot format
    pred(tensor): predicted map
    """
    channel_num, _, _ = mask.shape
    
    score = torch.zeros(channel_num)

    for i in range(channel_num):
        curr_score = dice_score_one(mask[i],pred[i])
        score[i] = curr_score
        
    return torch.mean(score)

def dice_loss(mask:torch.Tensor, pred:torch.Tensor) -> torch.Tensor:
    """
    dice loss is 1 - dice score
    """
    softmax = nn.Softmax(dim=1)
    pred_soft = softmax(pred)
        
    dice = smp.losses.DiceLoss(mode="multilabel")
    return dice(pred_soft,mask)

def jaccard_one(mask:torch.Tensor, pred:torch.Tensor, smooth=1e-5) -> torch.Tensor:
    """
    calculate jaccard index/IoU of one class
    """
    inputs = pred.view(-1)
    targets = mask.view(-1)
    
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    
    IoU = (intersection + smooth)/(union + smooth)

    return IoU

def m_jaccard(mask:torch.Tensor, pred:torch.Tensor, smooth=1e-5) -> torch.Tensor:
    channel_num, _, _ = mask.shape
    
    score = torch.zeros(channel_num)

    for i in range(channel_num):
        curr_score = jaccard_one(mask[i],pred[i])
        score[i] = curr_score
        
    return torch.mean(score)

def metric_every_batch(mask:torch.Tensor, pred:torch.Tensor, criterion:Callable) -> torch.Tensor:
    n = len(mask)
    softmax = nn.Softmax(dim=1)
    pred_soft = softmax(pred)
    curr_batch_loss = torch.zeros(n)

    for i in range(n):
        curr_metric = criterion(mask[i], pred_soft[i])
        curr_batch_loss[i] = curr_metric

    return torch.mean(curr_batch_loss)