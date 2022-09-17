# Shree KRISHNAya Namaha
# Common Utility Functions
# Author: Nagabhushan S N
# Last Modified: 30/10/2021

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas
import simplejson

from pathlib import Path

import torch
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_device(device: str):
    """
    Returns torch device object
    :param device: cpu/gpu0/gpu1
    :return:
    """
    if device == 'cpu':
        device = torch.device('cpu')
    elif device.startswith('gpu') and torch.cuda.is_available():
        gpu_num = int(device[3:])
        device = torch.device(f'cuda:{gpu_num}')
    else:
        device = torch.device('cpu')
    return device


def move_to_device(tensors_dict: dict, device):
    if device.type == 'cpu':
        for key in tensors_dict.keys():
            if isinstance(tensors_dict[key], torch.Tensor):
                tensors_dict[key] = tensors_dict[key].to(device)
    else:
        for key in tensors_dict.keys():
            if isinstance(tensors_dict[key], torch.Tensor):
                tensors_dict[key] = tensors_dict[key].cuda(device, non_blocking=True)
    return
