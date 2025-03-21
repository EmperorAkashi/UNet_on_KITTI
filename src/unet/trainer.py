import numpy as np
import hydra
import logging
import omegaconf

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torchvision import transforms
import albumentations as alb
import pytorch_lightning as pl
import unet.config as cf
from unet.model import UNet
import unet.metric as M
from unet.dataset import KittiDataset
from unet.file_utils import read_from_folder
import unet.feed_forward as F


class UnetTrainer(pl.LightningModule):
    hparams: cf.UnetTrainConfig 
    # constant intitialized for each instance
    # pl module has save_hparams attr, 
    # enable Lightning to store all the provided arguments 
    # under the self.hparams attribute

    def __init__(self, config: cf.UnetTrainConfig):
        super().__init__()
        # The LightningModule automatically save all the hyperparameters 
        # passed to init simply by calling self.save_hyperparameters()
        # with config, we need to structured it before call save_hyperparameters()
        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)
            
        self.save_hyperparameters(config)
        self.unet = UNet(config.model_config.in_channel, config.model_config.num_classes)
        self.fcn = F.FeedForward(config.model_config.in_channel, config.model_config.num_classes, 
                                config.model_config.kernel_size)
        self.config = config
    def forward(self, x):
        if self.config.debug:
            return self.fcn(x)
        return self.unet(x)

    def training_log(self, batch, pred:torch.Tensor, mask:torch.Tensor, loss: float, batch_idx: int):
        # Lightning offers automatic log functionalities for logging scalars, 
        # or manual logging for anything else
        jaccard = M.metric_every_batch(mask, pred, M.m_jaccard)
        f1_score = M.metric_every_batch(mask, pred, M.m_dice_score)
        # if batch_idx % 20 == 0:
        #     self.logger.experiment.add_images(
        #         'predict/mask',
        #         torch.stack([
        #             pred.detach()[0],
        #             mask[0]
        #             ], dim=0),#.unsqueeze_(-1),
        #         self.global_step,
        #         dataformats='NCHW'
        #     )
        self.log('train/loss', loss)
        self.log('train/dice', f1_score)
        self.log('train/mIoU', jaccard)

    def training_step(self, batch, batch_idx: int):
        image, mask = batch # loader create an iterator
        predict = self(image) # self call forward by default
       
        loss = M.dice_loss(predict,mask)
        self.training_log(batch, predict, mask, loss, batch_idx)
        return loss

    def validation_log(self, batch, pred:torch.Tensor, mask:torch.Tensor, loss: float, batch_idx: int):
        jaccard = M.metric_every_batch(mask, pred, M.m_jaccard)
        f1_score = M.metric_every_batch(mask, pred, M.m_dice_score)

        self.log('val/loss', loss)
        self.log('val/dice', f1_score)
        self.log('val/mIoU', jaccard)
        tb = self.logger.experiment
        # tb.add_images(
        #         'predict/mask',
        #         torch.stack([
        #             pred.detach()[0],
        #             mask[0]
        #             ], dim=0),#.unsqueeze_(-1),
        #         self.global_step,
        #         dataformats='NCHW'
        #     )
        
    def validation_step(self, batch, batch_idx: int):
        image, mask = batch
        predict = self(image) 
        
        loss = M.dice_loss(predict,mask)
        self.validation_log(batch, predict, mask, loss, batch_idx)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.unet.parameters(), lr=self.hparams.optim.learning_rate)
        return optim


class UnetDataModule(pl.LightningDataModule):
    def __init__(self, config: cf.UnetDataConfig, batch_size: int, debug: bool) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.transform = alb.Compose([alb.RandomCrop(width=self.config.crop,height=self.config.crop),  
                                      alb.HorizontalFlip(p=0.5)])
        self.norm = alb.Normalize() # use default mean and std
        self.debug = debug
        self.ds = KittiDataset(hydra.utils.to_absolute_path(self.config.file_path), 
                                self.transform, self.norm, self.debug)
        self.ds_train = None
        self.ds_val = None

    def setup(self, stage: str = None) -> None:
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

# config_name should consistent with the one in cs.store()
# config store turns dataclass into dataframes
@hydra.main(config_path=None, config_name='train', version_base='1.1' ) 
def main(config: cf.UnetTrainConfig):
    logger = logging.getLogger(__name__)
    if config.num_gpus == 1:
        trainer = pl.Trainer(
        accelerator=config.device, 
        devices=config.num_gpus,
        log_every_n_steps=config.log_every,
        max_epochs=config.num_epochs)
    else:
        trainer = pl.Trainer(
            accelerator=config.device, 
            devices=config.num_gpus,
            strategy='ddp',
            log_every_n_steps=config.log_every,
            max_epochs=config.num_epochs)
    
    data_config = config.data
    dm = UnetDataModule(data_config, config.batch_size, config.debug)
    model = UnetTrainer(config)

    trainer.fit(model,dm)

    if trainer.is_global_zero:
        logger.info(f'Finished training. Final loss: {trainer.logged_metrics["train/loss"]}')
        logger.info(f'Finished training. Final F1-score: {trainer.logged_metrics["train/dice"]}')
        logger.info(f'Finished training. Final mIoU: {trainer.logged_metrics["train/mIoU"]}')
          

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train', node=cf.UnetTrainConfig)
    main()