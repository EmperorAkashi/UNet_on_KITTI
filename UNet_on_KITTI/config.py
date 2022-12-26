import dataclasses
from typing import Optional 
import omegaconf
from labels import labels, get_numclasses

from model import UNet

@dataclasses.dataclass
class unet_data_config:
    """configuration of data loading

    attr:
    file_path(str): path of input image and semantic labels, 
                    files will be read separately by file utils
    train_prop(float): percentage for training
    """
    file_path: str = omegaconf.MISSING #
    #"/mnt/home/clin/ceph/dataset/kitti_semantic/training"
    train_prop: float = 0.9
    limit: Optional[int] = None
    num_data_workers: int = 16

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
    num_classes: int = get_numclasses(labels)

@dataclasses.dataclass
class unet_train_config:
    data: unet_data_config = unet_data_config()
    model_config: unet_config = unet_config()
    optim: optim_config = optim_config()
    batch_size: int = 64
    num_epochs: int = 10
    log_every: int = 1


