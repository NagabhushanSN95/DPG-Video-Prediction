# Shree KRISHNAya Namaha
# L1 regularization over flow gradient, weighted by frame gradients
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


class SmoothnessLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.flow_weight_coefficient = loss_configs['flow_weight_coefficient']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict):
        frame = input_dict['target_frame']
        flow = output_dict['predicted_flow']

        flow_dy, flow_dx = self.compute_gradients(flow)
        frame_dy, frame_dx = self.compute_gradients(frame)
        weights_x = torch.exp(-torch.mean(torch.abs(frame_dx), 1, keepdim=True) * self.flow_weight_coefficient)
        weights_y = torch.exp(-torch.mean(torch.abs(frame_dy), 1, keepdim=True) * self.flow_weight_coefficient)

        loss_x = weights_x * flow_dx.abs()
        loss_y = weights_y * flow_dy.abs()
        sm_loss = loss_x.mean() + loss_y.mean()
        return sm_loss

    @staticmethod
    def compute_gradients(image):
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        return grad_y, grad_x
