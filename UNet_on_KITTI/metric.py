import numpy as np
import torch
import sklearn as sk


def mIoU(predict, mask):
    return sk.metrics.jaccard_score(mask, predict)