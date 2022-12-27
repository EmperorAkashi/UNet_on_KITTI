import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import os
import cv2 as cv
from file_utils import read_from_folder
import torch


class kitti_dataset(Dataset):
    """
    Dataset to load semantic data from Kitti
    the dir is the top folder of dataset; the read_from_folder 
    will outputs a dict with {folder: [list file names]}
    """

    def __init__(self, dir, transform = None):
        self.transform = transform
        self.dir = dir
        self.df =  read_from_folder(dir)#

    def __len__(self):
        return len(self.df['image_2'])

    def __getitem__(self, index):
        print('num_iterations', index)
        img_path = self.df['image_2'][index]
        seg_path = self.df['semantic_rgb'][index]
        image = cv.imread(img_path)
        segment = cv.imread(seg_path)
        image, segment = self.transform(image), self.transform(segment)
        return torch.Tensor(image).permute(2,0,1), torch.Tensor(segment).permute(2,0,1)