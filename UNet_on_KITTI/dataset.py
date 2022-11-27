import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import os
import cv2 as cv

class kitti_dataset(Dataset):
    "Dataset to load semantic data from Kitti"

    def __init__(self, image_dir, segment_dir, dataframe, transform = None):
        self.transform = transform
        self.img_dir = image_dir
        self.seg_dir = segment_dir
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, '00000'+str(index)+'_10.png')
        seg_path = os.path.join(self.seg_dir, '00000'+str(index)+'_10.png')
        image = cv.imread(img_path)
        segment = cv.imread(seg_path)
        return image, segment