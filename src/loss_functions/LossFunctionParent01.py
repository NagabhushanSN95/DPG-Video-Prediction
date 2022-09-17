# Shree KRISHNAya Namaha
# Abstract parent class
# Author: Nagabhushan S N
# Last Modified: 30/10/2021

import abc
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


class LossFunctionParent:
    @abc.abstractmethod
    def __init__(self, configs: dict, loss_configs: dict):
        pass

    @abc.abstractmethod
    def compute_loss(self, input_dict: dict, output_dict: dict):
        pass
