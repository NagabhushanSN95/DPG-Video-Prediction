# Shree KRISHNAya Namaha
# A Factory method that returns a Occlusion Inpainter
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

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot


def get_occlusion_inpainter(configs: dict):
    if ('occlusion_inpainter' not in configs.keys()) or (configs['occlusion_inpainter'] is None):
        return None

    name = configs['occlusion_inpainter']['name']
    if name == 'Unet01':
        from occlusion_inpainters.Unet01 import OcclusionInpainter
        occlusion_computer = OcclusionInpainter(configs)
    else:
        raise RuntimeError(f'Unknown occlusion inpainter: {name}')
    return occlusion_computer
