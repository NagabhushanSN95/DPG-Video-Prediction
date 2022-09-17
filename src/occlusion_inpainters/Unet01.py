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

from libraries.partial_convolution.partialconv2d import PartialConv2d


class OcclusionInpainter(torch.nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.configs = configs
        self.conv1 = PartialConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, multi_channel=True, return_mask=True)
        self.conv2 = PartialConv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3, multi_channel=True, return_mask=True)
        self.conv3 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, multi_channel=True, return_mask=True)
        self.conv4 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.conv5 = PartialConv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2, multi_channel=True, return_mask=True)
        self.conv6 = PartialConv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.conv7 = PartialConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, multi_channel=True, return_mask=True)
        self.conv8 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.conv9 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, multi_channel=True, return_mask=True)
        self.conv10 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv11 = torch.nn.Conv2d(in_channels=512+512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13 = torch.nn.Conv2d(in_channels=512+256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv14 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv15 = torch.nn.Conv2d(in_channels=512+128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv16 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv17 = torch.nn.Conv2d(in_channels=256+64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv18 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv19 = torch.nn.Conv2d(in_channels=128+3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv20 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        return

    def forward(self, input_batch):
        x0 = input_batch['warped_frame']
        m0 = input_batch['occlusion_map'].repeat(1,3,1,1)

        x1, m1 = self.conv1(x0, m0)  # 256x256, 64
        x1 = F.relu(x1)
        
        x2, m2 = self.conv2(x1, m1)  # 128x128, 64
        x2 = F.relu(x2)
        
        x3, m3 = self.conv3(x2, m2)  # 128x128, 128
        x3 = F.relu(x3)
        
        x4, m4 = self.conv4(x3, m3)  # 64x64, 128
        x4 = F.relu(x4)
        
        x5, m5 = self.conv5(x4, m4)  # 64x64, 256
        x5 = F.relu(x5)
        
        x6, m6 = self.conv6(x5, m5)  # 32x32, 256
        x6 = F.relu(x6)
        
        x7, m7 = self.conv7(x6, m6)  # 32x32, 512
        x7 = F.relu(x7)
        
        x8, m8 = self.conv8(x7, m7)  # 16x16, 512
        x8 = F.relu(x8)
        
        x9, m9 = self.conv9(x8, m8)  # 16x16, 512
        x9 = F.relu(x9)
        
        x10, m10 = self.conv10(x9, m9)  # 8x8, 512
        x10 = F.relu(x10)

        x11 = self.up(x10)  # 16X16, 512
        x8_infilled = m8 * x8 + (1 - m8) * x11
        x11 = torch.cat([x11, x8_infilled], dim=1)  # 16X16, 512+512
        x11 = self.conv11(x11)  # 16X16, 512
        x11 = self.leaky_relu(x11)
        x12 = self.conv12(x11)  # 16X16, 512
        x12 = self.leaky_relu(x12)

        x13 = self.up(x12)  # 32x32, 512
        x6_infilled = m6 * x6 + (1 - m6) * x13[:, :256]
        x13 = torch.cat([x13, x6_infilled], dim=1)  # 32x32, 512+256
        x13 = self.conv13(x13)  # 32x32, 512
        x13 = self.leaky_relu(x13)
        x14 = self.conv14(x13)  # 32x32, 512
        x14 = self.leaky_relu(x14)

        x15 = self.up(x14)  # 64X64, 512
        x4_infilled = m4 * x4 + (1 - m4) * x15[:, :128]
        x15 = torch.cat([x15, x4_infilled], dim=1)  # 64X64, 512+128
        x15 = self.conv15(x15)  # 64X64, 256
        x15 = self.leaky_relu(x15)
        x16 = self.conv16(x15)  # 64X64, 256
        x16 = self.leaky_relu(x16)

        x17 = self.up(x16)  # 128X128, 256
        x2_infilled = m2 * x2 + (1 - m2) * x17[:, :64]
        x17 = torch.cat([x17, x2_infilled], dim=1)  # 128X128, 256+64
        x17 = self.conv17(x17)  # 128X128, 128
        x17 = self.leaky_relu(x17)
        x18 = self.conv18(x17)  # 128X128, 128
        x18 = self.leaky_relu(x18)

        x19 = self.up(x18)  # 256X256, 128
        x0_infilled = m0 * x0 + (1 - m0) * x19[:, :3]
        x19 = torch.cat([x19, x0_infilled], dim=1)
        x19 = self.conv19(x19)  # 256X256, 3
        x19 = self.leaky_relu(x19)
        x20 = self.conv20(x19)  # 256X256, 3
        x20 = torch.tanh(x20)

        result_dict = {
            'predicted_frame': x20,
        }
        return result_dict
