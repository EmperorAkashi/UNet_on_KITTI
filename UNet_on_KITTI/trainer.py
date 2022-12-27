import numpy as np
import hydra
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import pytorch_lightning as pl
import logging
import dataclasses
import config as cf
from model import UNet
from dice_loss import dice_loss
from dataset import kitti_dataset
from file_utils import read_from_folder


class unet_trainer(pl.LightningModule):
    hparams: cf.unet_train_config 
    #constant intitialized for each instance
    #pl module has save_haparams attr, 
    # enable Lightning to store all the provided arguments 
    # under the self.hparams attribute

    def __init__(self, config: cf.unet_train_config):
        super().__init__()
        self.save_hyperparameters(config)
        self.unet = UNet(config.model_config.in_channel, config.model_config.num_classes)

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx: int):
        image, mask = batch
        predict = self(image) #self call forward by default
        loss = dice_loss(predict, mask)
        return loss
        
    #need validation step

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.unet.parameters(), lr=self.hparams.optim.learning_rate)
        return optim


class unet_data_module(pl.LightningDataModule):
    def __init__(self, config: cf.unet_data_config, batch_size) -> None:
        super().__init__()
        self.config = config
        self.ds = kitti_dataset(config.file_path)
        self.batch_size = batch_size
        self.transform = transform.Compose([Resize(self.config.resize[0], self.config.resize[1]), 
                                   Normalize(mean=self.config.resnet_mean[0], std= self.config.resnet_std[0], p=1), 
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5)])
        self.ds_train = None
        self.ds_val = None

    def setup(self, stage: str) -> None:
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transforms etc.
        if self.config.limit is not None:
            limit = min(self.config.limit, len(self.ds))
            self.ds, _ = torch.utils.data.random_split(self.ds, [limit, len(self.ds) - limit])

        num_train_samples = int(len(self.ds) * self.config.train_prop)

        self.ds_train, self.ds_val = torch.utils.data.random_split(
            self.ds, [num_train_samples, len(self.ds) - num_train_samples],
            torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_train, self.batch_size, shuffle=True, num_workers=self.config.num_data_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_val, self.batch_size, shuffle=False, num_workers=self.config.num_data_workers)

#config_name should consistent with the one in cs.store()
#config store turns dataclass into dataframes
@hydra.main(config_path=None, config_name='train', version_base='1.1' ) 
def main(config: cf.unet_train_config):
    trainer = pl.Trainer(
        accelerator='cpu', 
        log_every_n_steps=config.log_every,
        max_epochs=config.num_epochs)
    data_config = config.data
    dm = unet_data_module(data_config, config.batch_size)
    model = unet_trainer(config)

    trainer.fit(model,dm)
          

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train', node=cf.unet_train_config)
    main()