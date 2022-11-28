import numpy as np
import hydra
import torch
import pytorch_lightning as pl
import logging
import dataclasses
import config as cf
from model import UNet
from dice_loss import dice_loss
from dataset import kitti_dataset

class unet_trainer(pl.lightning_module):
    hparams: cf.unet_train_config #constant intitialized for each instance

    def __init__(self, config: cf.unet_train_config):
        super().__init__()
        self.save_hyperparameter(config)
        self.unet = UNet(config.unet_config.in_channel, config.unet_config.num_classes)

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx: int):
        image, mask = batch
        predict = self(image) #self call forward by default
        loss = dice_loss(predict, mask)
        
    #need validation step

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.unet.parameters(), lr=self.hparams.optim.learning_rate)
        return optim


class unet_data_module(pl.LightningDataModule):
    def __init__(self, config: cf.unet_data_config, batch_size) -> None:
        super().__init__()
        self.config = config
        self.ds = kitti_dataset(config.image_path, config.segment_path)
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        all_img = np.array(self.config.dataframe.id.unique())  #create an array for all image id
        np.random.seed(42)
        idx_shuf = np.random.permutation(len(all_img))  #shuffle the dataset
        num_train = int(len(all_img)*self.config.train_prop) #percent of training set
        if stage == train:
            self.img_sf = all_img[idx_shuf[:num_train]]
        else:
            self.img_sf = all_img[idx_shuf[num_train:]]
            
        return super().setup(stage)

    def train_dataloader(self):
        return torch.utils.data.Dataloader(self.ds, self.batch_size, shuffle=True)

@hydra.main(config_path=None, config_name='config')
def main(config: cf.unet_train_config, dm: pl.LightningDataModule):
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1)
    data_config = config.data
    dm = unet_data_module(data_config, config.batch_size)
    model = unet_trainer(config)

    trainer.fit(model,dm)
          

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train_base', node=cf.unet_train_config)
    main()