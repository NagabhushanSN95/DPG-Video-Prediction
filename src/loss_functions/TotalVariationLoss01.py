# Shree KRISHNAya Namaha
# Total Variation Loss
# Author: Nagabhushan S N
# Last Modified: 30/10/2021

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot
import torch
import torch.nn.functional as F

from loss_functions.LossFunctionParent01 import LossFunctionParent
from utils.Warper import Warper


class TotalVariationLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.alpha = loss_configs['alpha']
        self.beta = loss_configs['beta']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict):
        predicted_frame = output_dict['predicted_frame']
        b = predicted_frame.shape[0]

        grad_x = predicted_frame[:, :, :-1, 1:] - predicted_frame[:, :, :-1, :-1]
        grad_y = predicted_frame[:, :, 1:, :-1] - predicted_frame[:, :, :-1, :-1]
        tv_loss_pixel = torch.sqrt(torch.sum(torch.square(grad_x), dim=1) + torch.sum(torch.square(grad_y), dim=1) + 1e-12)
        tv_loss = torch.mean(tv_loss_pixel)
        return tv_loss
