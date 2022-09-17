# Shree KRISHNAya Namaha
# A Factory method that returns a Frame Predictor
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


def get_frame_predictor(configs: dict, flow_predictor=None, occlusion_computer=None, occlusion_inpainter=None):
    name = configs['frame_predictor']['name']
    if name == 'FramePredictor01':
        from frame_predictors.FramePredictor01 import FramePredictor
        model = FramePredictor(configs, flow_predictor, occlusion_computer, occlusion_inpainter)
    else:
        raise RuntimeError(f'Unknown Frame Predictor: {name}')
    return model
