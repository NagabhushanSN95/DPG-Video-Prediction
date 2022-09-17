# Shree KRISHNAya Namaha
# Computes all specified losses
# Author: Nagabhushan S N
# Last Modified: 30/10/2021

import time
import datetime
import traceback
from typing import List, Tuple

import numpy
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot
import importlib.util


class LossComputer:
    def __init__(self, configs: dict):
        self.losses = {}
        for loss_configs in configs['losses']:
            loss_name = loss_configs['name']
            loss_weight = loss_configs['weight']
            self.losses[loss_name] = self.get_loss_object(loss_name, configs, loss_configs), loss_weight
        return

    @staticmethod
    def get_loss_object(loss_name, configs: dict, loss_configs: dict):
        if loss_name == 'PixelLoss01':
            from loss_functions.PixelLoss01 import PixelLoss
            loss_obj = PixelLoss(configs, loss_configs)
        elif loss_name == 'PixelLoss02':
            from loss_functions.PixelLoss02 import PixelLoss
            loss_obj = PixelLoss(configs, loss_configs)
        elif loss_name == 'SmoothnessLoss01':
            from loss_functions.SmoothnessLoss01 import SmoothnessLoss
            loss_obj = SmoothnessLoss(configs, loss_configs)
        elif loss_name == 'PerceptualLoss01':
            from loss_functions.PerceptualLoss01 import PerceptualLoss
            loss_obj = PerceptualLoss(configs, loss_configs)
        elif loss_name == 'StyleLoss01':
            from loss_functions.StyleLoss01 import StyleLoss
            loss_obj = StyleLoss(configs, loss_configs)
        elif loss_name == 'PerceptualAndStyleLoss01':
            from loss_functions.PerceptualAndStyleLoss01 import PerceptualAndStyleLoss
            loss_obj = PerceptualAndStyleLoss(configs, loss_configs)
        elif loss_name == 'TotalVariationLoss01':
            from loss_functions.TotalVariationLoss01 import TotalVariationLoss
            loss_obj = TotalVariationLoss(configs, loss_configs)
        else:
            raise RuntimeError(f'Unknown Loss Function: {loss_name}')
        return loss_obj

    def compute_losses(self, input_dict, output_dict):
        loss_values = {}
        total_loss = 0
        for loss_name in self.losses.keys():
            loss_obj, loss_weight = self.losses[loss_name]
            loss_value = loss_obj.compute_loss(input_dict, output_dict)
            loss_values[loss_name] = loss_value
            total_loss += loss_weight * loss_value
        loss_values['TotalLoss'] = total_loss
        return loss_values
