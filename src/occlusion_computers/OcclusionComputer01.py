# Shree KRISHNAya Namaha
# Using predicted backward flow, computes occlusion + disocclusion mask
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

from utils.Warper import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class OcclusionComputer:

    def __init__(self, configs: dict) -> None:
        self.configs = configs
        self.warper = Warper(configs['device'])
        return

    def __call__(self, input_dict: dict):
        return self.compute_occlusion_map(input_dict)

    def compute_occlusion_map(self, input_dict: dict):
        backward_flow = input_dict['predicted_flow']
        density_last = self.compute_splatting_weights(backward_flow)
        density_next = self.warper.bilinear_interpolation(density_last, None, backward_flow, None, is_image=False)[0]
        valid_regions_last = ((density_next > 0) & (density_next < 2)).float()
        result_dict = {
            'occlusion_map': valid_regions_last
        }
        return result_dict

    def compute_splatting_weights(self, flow12: torch.Tensor) -> torch.Tensor:
        """
        Based on bilinear splatting here: https://github.com/NagabhushanSN95/Pose-Warping/blob/main/src/WarperPytorch.py
        dated 27/09/2021
        :param flow12: (b,c,h,w)
        :return: splatting_weights: (b,1,h,w)
        """
        b, _, h, w = flow12.shape
        grid = self.create_grid(b, h, w).to(flow12)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        weight_nw = torch.moveaxis(prox_weight_nw, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(flow12)
        batch_indices = torch.arange(b)[:, None, None].to(flow12.device)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                  weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                  weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                  weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                  weight_se, accumulate=True)

        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]
        return cropped_weights

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid
