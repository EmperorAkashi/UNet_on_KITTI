import dataclasses
from typing import Optional 
import omegaconf

from UNet_on_KITTI.model import UNet

@dataclasses.dataclass
class unet_data_config:
    """configuration of data loading

    attr:
    path(str): path of training data
    train_prop(float): percentage for training
    """
    path: str = "path/place/holder"
    train_prop: float = 0.9

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
    data: unet_data_config = unet_data_config()
    optim: optim_config = optim_config()
    model_config: omegaconf.MISSING
    batch_size: int = 64
    num_epochs: int = 10


