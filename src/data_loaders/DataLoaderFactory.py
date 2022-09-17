# Shree KRISHNAya Namaha
# A Factory method that returns a Data Loader
# Author: Nagabhushan S N
# Last Modified: 02/11/2021

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


def get_data_loader(configs: dict, data_dirpath: Path, video_datapath: Optional[Path]):
    name = configs['data_loader']['name']
    if name == 'OursBlender01':
        from data_loaders.OursBlender01 import OursBlenderDataLoader
        data_loader = OursBlenderDataLoader(configs, data_dirpath, video_datapath)
    elif name == 'OursBlender02':
        from data_loaders.OursBlender02 import OursBlenderDataLoader
        data_loader = OursBlenderDataLoader(configs, data_dirpath, video_datapath)
    elif name == 'OursBlender03':
        from data_loaders.OursBlender03 import OursBlenderDataLoader
        data_loader = OursBlenderDataLoader(configs, data_dirpath, video_datapath)
    elif name == 'OursBlender04':
        from data_loaders.OursBlender04 import OursBlenderDataLoader
        data_loader = OursBlenderDataLoader(configs, data_dirpath, video_datapath)
    elif name == 'Sintel01':
        from data_loaders.Sintel01 import SintelDataLoader
        data_loader = SintelDataLoader(configs, data_dirpath, video_datapath)
    elif name == 'Sintel02':
        from data_loaders.Sintel02 import SintelDataLoader
        data_loader = SintelDataLoader(configs, data_dirpath, video_datapath)
    elif name == 'Sintel03':
        from data_loaders.Sintel03 import SintelDataLoader
        data_loader = SintelDataLoader(configs, data_dirpath, video_datapath)
    else:
        raise RuntimeError(f'Unknown data loader: {name}')
    return data_loader
