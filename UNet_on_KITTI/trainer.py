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
        return super().setup(stage)

    def train_dataloader(self):
        return torch.utils.data.Dataloader(self.ds, self.batch_size, shuffle=True)

@hydra.main(config_path=None, config_name='config')
    def train(config: cf.unet_train_config):
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1)

        #need data module for kitti: dm = 
        model = unet_trainer(config)

        trainer.fit(model, datamodule=dm)
        

        

        
