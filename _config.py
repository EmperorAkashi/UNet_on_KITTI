import dataclasses
from importlib.resources import path
from typing import Optional
from . import model


@dataclasses.dataclass
class unet_data_config:
    path: str = ""

@dataclasses.dataclass
class optim_config:
    learning_rate: float = 1e-3

@dataclasses.dataclass
class unet_trainer_config:
    data: unet_data_config = unet_data_config()
    model = model.UNet()