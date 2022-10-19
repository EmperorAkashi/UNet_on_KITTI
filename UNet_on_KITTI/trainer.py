import numpy as np
import hydra
import torch
import pytorch_ligtning
import logging
import dataclasses
from _config import *

class unet_trainer(object, pytorch_ligtning.lightning_module):
    hparams: unet_data_config #constant intitialized for each instance

    def __init__(self, config: unet_data_config):
        super().__init__()
        

        
