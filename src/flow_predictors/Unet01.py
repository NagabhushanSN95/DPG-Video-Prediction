# Shree KRISHNAya Namaha
# Unet.
# Author: Nagabhushan S N
# Date: 30/10/2021

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas

from pathlib import Path

import torch
from tqdm import tqdm
from matplotlib import pyplot
import torch.nn.functional as F


class FlowPredictor(torch.nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.configs = configs
        self.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv9 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv10 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv11 = torch.nn.Conv2d(in_channels=512+512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13 = torch.nn.Conv2d(in_channels=512+256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv14 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv15 = torch.nn.Conv2d(in_channels=512+128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv16 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv17 = torch.nn.Conv2d(in_channels=256+64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv18 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv19 = torch.nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv20 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        return

    def forward(self, input_batch):
        past_frames = torch.split(input_batch['past_frames'], 1, dim=1)
        network_input = torch.cat(past_frames, dim=2)[:, 0]
        
        x1 = self.conv1(network_input)  # 256x256, 64
        x1 = F.relu(x1)
        
        x2 = self.conv2(x1)  # 128x128, 64
        x2 = F.relu(x2)
        
        x3 = self.conv3(x2)  # 128x128, 128
        x3 = F.relu(x3)
        
        x4 = self.conv4(x3)  # 64x64, 128
        x4 = F.relu(x4)
        
        x5 = self.conv5(x4)  # 64x64, 256
        x5 = F.relu(x5)
        
        x6 = self.conv6(x5)  # 32x32, 256
        x6 = F.relu(x6)
        
        x7 = self.conv7(x6)  # 32x32, 512
        x7 = F.relu(x7)
        
        x8 = self.conv8(x7)  # 16x16, 512
        x8 = F.relu(x8)
        
        x9 = self.conv9(x8)  # 16x16, 512
        x9 = F.relu(x9)
        
        x10 = self.conv10(x9)  # 8x8, 512
        x10 = F.relu(x10)

        x11 = self.up(x10)  # 16X16, 512
        x11 = torch.cat([x11, x8], dim=1)  # 16X16, 512+512
        x11 = self.conv11(x11)  # 16X16, 512
        x11 = self.leaky_relu(x11)
        x12 = self.conv12(x11)  # 16X16, 512
        x12 = self.leaky_relu(x12)

        x13 = self.up(x12)  # 32x32, 512
        x13 = torch.cat([x13, x6], dim=1)  # 32x32, 512+256
        x13 = self.conv13(x13)  # 32x32, 512
        x13 = self.leaky_relu(x13)
        x14 = self.conv14(x13)  # 32x32, 512
        x14 = self.leaky_relu(x14)

        x15 = self.up(x14)  # 64X64, 512
        x15 = torch.cat([x15, x4], dim=1)  # 64X64, 512+128
        x15 = self.conv15(x15)  # 64X64, 256
        x15 = self.leaky_relu(x15)
        x16 = self.conv16(x15)  # 64X64, 256
        x16 = self.leaky_relu(x16)

        x17 = self.up(x16)  # 128X128, 256
        x17 = torch.cat([x17, x2], dim=1)  # 128X128, 256+64
        x17 = self.conv17(x17)  # 128X128, 128
        x17 = self.leaky_relu(x17)
        x18 = self.conv18(x17)  # 128X128, 128
        x18 = self.leaky_relu(x18)

        x19 = self.up(x18)  # 256X256, 128
        x19 = self.conv19(x19)  # 256X256, 2
        x19 = self.leaky_relu(x19)
        x20 = self.conv20(x19)  # 256X256, 2

        result_dict = {
            'predicted_flow': x20,
        }
        return result_dict
