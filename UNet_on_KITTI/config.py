import dataclasses
from typing import Optional 
import omegaconf
import pandas as pd

from UNet_on_KITTI.model import UNet

@dataclasses.dataclass
class unet_data_config:
    """configuration of data loading

    attr:
    image_path(str): path of input image
    segment_path(str): path of semantic image
    train_prop(float): percentage for training
    """
    dataframe: omegaconf.MISSING
    image_path: str = "./Kitti/training"
    segment_path: str = "./Kitti/semantic_rgb"
    train_prop: float = 0.9
    limit: Optional[int] = None

@dataclasses.dataclass
class optim_config:
    """hyperparams of optimization

    attr:
    learning_rate(float): lr of training
    """
    learning_rate: float = 1e-3

@dataclasses.dataclass
class unet_config:
    in_channel: int = 3
    num_classes: int = 3

@dataclasses.dataclass
class unet_train_config:
    model_config: omegaconf.MISSING
    data: unet_data_config = unet_data_config()
    optim: optim_config = optim_config()
    batch_size: int = 64
    num_epochs: int = 10


