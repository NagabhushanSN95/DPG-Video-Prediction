# Shree KRISHNAya Namaha
# L1 Loss function on predicted frame
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


class PixelLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.alpha = loss_configs['alpha']
        self.beta = loss_configs['beta']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict):
        target_frame = input_dict['target_frame']
        predicted_frame = output_dict['predicted_frame']
        occlusion_map = output_dict['occlusion_map']

        known_mae = self.compute_mae(target_frame, predicted_frame, occlusion_map)
        known_ssim = self.compute_ssim(target_frame, predicted_frame, occlusion_map)
        known_pixel_loss = (self.alpha * (1 - known_ssim) / 2) + ((1 - self.alpha) * known_mae)

        unknown_mae = self.compute_mae(target_frame, predicted_frame, 1 - occlusion_map)
        unknown_ssim = self.compute_ssim(target_frame, predicted_frame, 1 - occlusion_map)
        unknown_pixel_loss = (self.alpha * (1 - unknown_ssim) / 2) + ((1 - self.alpha) * unknown_mae)

        pixel_loss = known_pixel_loss + self.beta * unknown_pixel_loss
        return pixel_loss

    @staticmethod
    def compute_mae(target_frame, predicted_frame, mask):
        masked_target = mask * target_frame
        masked_predicted = mask * predicted_frame
        mae = torch.mean(torch.abs(masked_target - masked_predicted))
        return mae

    @staticmethod
    def compute_ssim(target_frame, predicted_frame, mask, md=1):
        """
        Implementation taken from https://github.com/lliuz/ARFlow/blob/master/losses/loss_blocks.py
        """
        patch_size = 2 * md + 1
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        x = mask * target_frame
        y = mask * predicted_frame

        mu_x = torch.nn.AvgPool2d(patch_size, 1, 0)(x)
        mu_y = torch.nn.AvgPool2d(patch_size, 1, 0)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = torch.nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
        sigma_y = torch.nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
        sigma_xy = torch.nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

        nr = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        dr = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        ssim = torch.mean(torch.clamp(nr / dr, min=-1, max=1))
        return ssim
