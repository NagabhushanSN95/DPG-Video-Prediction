# Shree KRISHNAya Namaha
# A Factory method that returns a Flow Predictor
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


def get_flow_predictor(configs: dict):
    name = configs['flow_predictor']['name']
    if name == 'Unet01':
        from flow_predictors.Unet01 import FlowPredictor
        flow_predictor = FlowPredictor(configs)
    else:
        raise RuntimeError(f'Unknown flow predictor: {name}')
    return flow_predictor
