import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler


"running length encoder's decoder"
def rle_decode(rle_annota, shape, color = 1):
    s = rle_annota.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.float32)
    for i in range(len(starts)):
        lo = starts[i]
        hi = ends[i]
        img[lo:hi] = color
    return img.reshape(shape)

"decode mask for a certain image"
def create_mask(df_train, img_id, shape):
    dfimage = df_train[df_train['id'] == img_id]
    annota_all = dfimage['annotation'].tolist()
    img_mask = np.zeros(shape)
    for i in range(len(annota_all)):
        mask_i = rle_decode(annota_all[i], shape)
        img_mask += mask_i
    img_mask = img_mask.clip(0, 1)
    return np.array(img_mask)

class Data(Dataset):
    def __init__(self, df: pd.core.frame.DataFrame, train_pct, train = bool):
        self.Image = (224, 224)
        self.RESNET_MEAN = (0.485, 0.456, 0.406)
        self.RESNET_STD = (0.229, 0.224, 0.225)
        self.df = df
        self.dir = data_dir['TRAIN_PATH']
        self.gb = self.df.groupby('id')
 
        self.trans = Compose([Resize(self.Image[0], self.Image[1]), 
                                   Normalize(mean=self.RESNET_MEAN[0], std= self.RESNET_STD[0], p=1), 
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5)])
        all_img = np.array(df.id.unique())  #create an array for all image id
        np.random.seed(42)
        idx_shuf = np.random.permutation(len(all_img))  #shuffle the dataset
        num_train = int(len(all_img)*train_pct) #percent of training set
        if train:
            self.img_sf = all_img[idx_shuf[:num_train]]
        else:
            self.img_sf = all_img[idx_shuf[num_train:]]
            
    def __len__(self):
        return len(self.img_sf)
    
    #get image sample with idx from the dataset for each item
    def __getitem__(self, idx):
        img_samp = self.img_sf[idx]
        dfimage = self.gb.get_group(img_samp)#traindf[traindf['id']==img_samp] #each id has lots of img, group them
        image_path = os.path.join(self.dir, img_samp) + ".png"
        
        image = io.imread(image_path)
        mask = create_mask(self.df, img_samp, image.shape)
        mask = (mask >= 1).astype('float32')
        augmented = self.trans(image=image, mask=mask)
        img_aug = augmented['image']
        msk_aug = augmented['mask']
        img_aug = img_aug.astype('float32')
        return img_aug.reshape((1, self.Image[0], self.Image[1])), msk_aug.reshape((1, self.Image[0], self.Image[1]))