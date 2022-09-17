# Shree KRISHNAya Namaha
# 
# Author: Nagabhushan S N
# Last Modified: 30/10/2021
import abc
import time
import datetime
import traceback
from typing import Optional

import numpy
import skimage.io
import skvideo.io
import pandas
import simplejson

from pathlib import Path

import torch.utils.data
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DataLoaderParent(torch.utils.data.Dataset):

    def __init__(self):
        super(DataLoaderParent, self).__init__()
        pass

    @abc.abstractmethod
    def load_test_data(self, video_name: str, seq_num: int, pred_frame_num: int):
        pass
