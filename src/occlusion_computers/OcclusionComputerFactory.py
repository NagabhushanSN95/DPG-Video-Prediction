# Shree KRISHNAya Namaha
# A Factory method that returns a Occlusion Computer
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


def get_occlusion_computer(configs: dict):
    name = configs['occlusion_computer']['name']
    if name == 'OcclusionComputer01':
        from occlusion_computers.OcclusionComputer01 import OcclusionComputer
        occlusion_computer = OcclusionComputer(configs)
    else:
        raise RuntimeError(f'Unknown occlusion computer: {name}')
    return occlusion_computer
