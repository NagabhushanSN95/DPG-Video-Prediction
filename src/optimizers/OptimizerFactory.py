# Shree KRISHNAya Namaha
# A Factory method that returns an Optimizer
# Author: Nagabhushan S N
# Last Modified: 30/10/2021

import time
import datetime
import traceback
from typing import Optional

import numpy
import skimage.io
import skvideo.io
import pandas
import torch

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot


def get_optimizer(configs: dict, parameters):
    name = configs['optimizer']['name']
    if name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=configs['optimizer']['lr'],
                                     betas=(configs['optimizer']['beta1'], configs['optimizer']['beta2']))
    else:
        raise RuntimeError(f'Unknown optimizer: {name}')
    return optimizer
