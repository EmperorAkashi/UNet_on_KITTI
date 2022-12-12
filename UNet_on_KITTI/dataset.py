import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import os
import cv2 as cv

class kitti_dataset(Dataset):
    "Dataset to load semantic data from Kitti"

    def __init__(self, dir, transform = None):
        self.transform = transform
        self.dir = image_dir
        self.df = dataframe #need check the def of dataframe of kitti

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, '00000'+str(index)+'_10.png')
        seg_path = os.path.join(self.seg_dir, '00000'+str(index)+'_10.png')
        image = cv.imread(img_path)
        segment = cv.imread(seg_path)
        return image, segment