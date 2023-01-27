import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import os
import cv2 as cv
from file_utils import read_from_folder
import torch
import labels as lb


class kitti_dataset(Dataset):
    """
    Dataset to load semantic data from Kitti
    the dir is the top folder of dataset; the read_from_folder 
    will outputs a dict with {folder: [list file names]}
    """

    def __init__(self, dir, transform = None, norm = None, debug = None):
        self.transform = transform #transforms will be specified in trainer
        self.norm = norm
        self.dir = dir
        self.df =  read_from_folder(dir)
        self.debug = debug

    def __len__(self):
        return len(self.df['image_2'])

    def __getitem__(self, index):
        img_path = self.df['image_2'][index]
        seg_path = self.df['semantic_rgb'][index]
        image_ = cv.imread(img_path)
        segment = cv.imread(seg_path)
        segment = lb.rgb_to_onehot(segment)
        augment = self.transform(image=image_, mask=segment) #transform used here are from albulmentations
        img_aug = augment['image']
        msk_aug = augment['mask']
        #normalize image&mask separately
        norm_img = self.norm(image=img_aug)
        img_norm = norm_img['image']
        
        return torch.Tensor(img_norm).permute(2,0,1), torch.Tensor(msk_aug).permute(2,0,1)
        #totensor and permutation maybe specified in transforms?