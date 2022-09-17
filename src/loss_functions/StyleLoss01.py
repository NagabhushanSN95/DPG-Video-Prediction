# Shree KRISHNAya Namaha
# Style Loss with VGG features
# Author: Nagabhushan S N
# Last Modified: 31/10/2021

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas

from pathlib import Path

import torchvision
from tqdm import tqdm
from matplotlib import pyplot
import torch
import torch.nn.functional as F

from loss_functions.LossFunctionParent01 import LossFunctionParent
from utils import CommonUtils


class StyleLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.beta = loss_configs['beta']
        self.perceptual_loss_weight = loss_configs['perceptual_loss_weight']
        self.style_loss_weight = loss_configs['style_loss_weight']

        self.vgg_features = torchvision.models.vgg16(pretrained=True).features
        self.vgg_features.eval()
        self.vgg_features.requires_grad_(False)
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = CommonUtils.get_device(configs['device'])
        self.vgg_features.to(self.device)
        return

    def compute_loss(self, input_dict: dict, output_dict: dict):
        target_frame = input_dict['target_frame']
        predicted_frame = output_dict['predicted_frame']
        occlusion_map = output_dict['occlusion_map']

        target_frame_unit_range = (target_frame + 1) / 2
        pred_frame_unit_range = (predicted_frame + 1) / 2
        target_frame_norm = self.normalizer(target_frame_unit_range)
        pred_frame_norm = self.normalizer(pred_frame_unit_range)
        target_frame_features = self.vgg_features(target_frame_norm)
        predicted_frame_features = self.vgg_features(pred_frame_norm)

        known_style_loss = self.compute_style_loss(target_frame_features, predicted_frame_features, occlusion_map)
        unknown_style_loss = self.compute_style_loss(target_frame_features, predicted_frame_features, 1 - occlusion_map)
        style_loss = known_style_loss + self.beta * unknown_style_loss
        return style_loss

    def compute_style_loss(self, target_frame_features: torch.Tensor, predicted_frame_features: torch.Tensor,
                           mask: torch.Tensor):
        feature_error = target_frame_features - predicted_frame_features

        b, _, h, w = target_frame_features.shape
        resized_mask = F.interpolate(mask, size=(h, w), mode='nearest')
        masked_feature_error = resized_mask * feature_error

        error_gram_matrix = self.gram_matrix(masked_feature_error)
        style_loss = torch.mean(torch.abs(error_gram_matrix))
        return style_loss

    @staticmethod
    def gram_matrix(features):
        """
        Taken from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
        """
        b, c, h, w = features.size()  # b=batch size(=1)
        # c=number of feature maps
        # (h,w)=dimensions of b f. map (N=h*w)

        features = features.view(b, c, h * w)  # resise F_XL into \hat F_XL

        G = torch.bmm(features, features.transpose(0, 2, 1))  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(c * h * w)
