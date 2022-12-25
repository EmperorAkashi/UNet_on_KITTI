import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import os
import cv2 as cv
from file_utils import read_from_folder


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
        print(self.df)

    def __len__(self):
        return len(self.df[0])

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, 'training', self.df['training'][index])
        seg_path = os.path.join(self.dir, 'semantic', self.df['semantic'][index])
        image = cv.imread(img_path)
        segment = cv.imread(seg_path)
        return image, segment