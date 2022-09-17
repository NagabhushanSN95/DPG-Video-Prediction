# Shree KRISHNAya Namaha
# A Factory method that returns a learning rate Scheduler
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

from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from matplotlib import pyplot


def get_scheduler(configs: dict, scheduler):
    if ('scheduler' not in configs.keys()) or (configs['scheduler'] is None):
        return None

    name = configs['scheduler']['name']
    if name == 'MultiStepLR':
        scheduler = MultiStepLR(scheduler, milestones=configs['scheduler']['milestones'],
                                gamma=configs['scheduler']['gamma'])
    else:
        raise RuntimeError(f'Unknown optimizer: {name}')
    return scheduler
