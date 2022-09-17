# Shree KRISHNAYa Namaha
# Modified from OursBlender01.py.
# Author: Nagabhushan S N
# Last Modified: 31/10/2021

import random
from pathlib import Path
from typing import Optional

import Imath
import OpenEXR
import numpy
import pandas
import skimage.io
import torch
from tqdm import tqdm

from data_loaders.DataLoaderParent import DataLoaderParent


class SintelDataLoader(DataLoaderParent):
    """
    Loads patches of resolution 256x256. Patches are selected such that they contain atleast 1 hole (either in
    warped_frame2 or warped_frame3 (without local motion prediction))
    """

    def __init__(self, configs: dict, data_dirpath: Path, frames_datapath: Optional[Path]):
        super(SintelDataLoader, self).__init__()
        self.dataroot = data_dirpath
        self.frames = {}
        self.num_past_frames = configs['data_loader']['num_past_frames']
        self.num_frames_data = self.read_num_frames_data()
        self.resolution = (436, 1024)
        self.patch_size = configs['data_loader']['patch_size']
        if frames_datapath is not None:
            self.samples_data = self.create_samples(frames_datapath)
            self.create_memmaps()
        else:
            self.samples_data = None
        return

    def read_num_frames_data(self):
        filepath = self.dataroot / 'all/NumberOfFramesPerVideo.csv'
        num_frames_data = dict(pandas.read_csv(filepath).to_numpy())
        return num_frames_data

    def create_samples(self, frames_datapath: Path):
        print('Creating Samples for Training')
        video_data = pandas.read_csv(frames_datapath)
        video_seqs = video_data[['video_name', 'seq_name']].drop_duplicates()
        samples_data = []
        for i, (video_name, seq_name) in video_seqs.iterrows():
            num_frames = self.num_frames_data[video_name]
            for pred_frame_num in range(self.num_past_frames, num_frames):
                samples_data.append([video_name, seq_name, pred_frame_num])
        return samples_data

    def create_memmaps(self):
        print('Memory mapping the database')
        for video_name, seq_name, pred_frame_num in tqdm(self.samples_data):
            if video_name not in self.frames.keys():
                self.frames[video_name] = {}
            if seq_name not in self.frames[video_name].keys():
                self.frames[video_name][seq_name] = {}

            for frame_num in range(pred_frame_num - 4, pred_frame_num + 1):
                if frame_num in self.frames[video_name][seq_name].keys():
                    continue
                frame_path = self.dataroot / f'all/RenderedData/{video_name}/rgb/{seq_name}/{frame_num:04}.npy'
                self.frames[video_name][seq_name][frame_num] = self.read_npy_file(frame_path, mmap_mode='r')
        return

    def __len__(self):
        return len(self.samples_data)

    def __getitem__(self, index):
        data_dict = self.load_training_data(index)
        return data_dict

    def load_training_data(self, index: int, random_crop: bool = True, random_flip: bool = True):
        video_name, seq_name, pred_frame_num = self.samples_data[index]

        if random_crop:
            h, w = self.resolution
            ph, pw = self.patch_size
            psp = [numpy.random.randint(0, h - ph), numpy.random.randint(0, w - pw)]
        else:
            psp = None

        if random_flip:
            flip = random.randint(0, 1) == 0
        else:
            flip = False

        padding = None

        past_frames = []
        for frame_num in range(pred_frame_num - 4, pred_frame_num):
            frame = self.frames[video_name][seq_name][frame_num]
            processed_frame = self.preprocess_frame(frame, psp, padding, flip)
            past_frames.append(processed_frame)
        past_frames = numpy.stack(past_frames)

        target_frame = self.frames[video_name][seq_name][pred_frame_num]
        processed_target = self.preprocess_frame(target_frame, psp, padding, flip)

        data_dict = {
            'past_frames': torch.from_numpy(past_frames),
            'target_frame': torch.from_numpy(processed_target),
        }
        return data_dict

    def load_testing_data(self, video_name: str, seq_name: str, pred_frame_num: int, padding: tuple = None):
        psp = None
        flip = False

        past_frames = []
        for frame_num in range(pred_frame_num - 4, pred_frame_num):
            frame_path = self.dataroot / f'all/RenderedData/{video_name}/rgb/{seq_name}/{frame_num:04}.png'
            frame = self.read_image(frame_path)
            processed_frame = self.preprocess_frame(frame, psp, padding, flip)
            past_frames.append(processed_frame)
        past_frames = numpy.stack(past_frames)

        data_dict = {
            'past_frames': torch.from_numpy(past_frames),
        }
        return data_dict

    @staticmethod
    def read_npy_file(path: Path, mmap_mode: str = None):
        if path.suffix == '.npy':
            array = numpy.load(path.as_posix(), mmap_mode=mmap_mode)
        else:
            raise RuntimeError(f'Unknown array format: {path.as_posix()}')
        return array

    @staticmethod
    def read_image(path: Path):
        if path.suffix == '.png':
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    def preprocess_frame(self, frame: numpy.ndarray, psp: Optional[list], padding: Optional[tuple], flip: bool):
        if psp is not None:
            y1, x1 = psp
            y2, x2 = y1 + self.patch_size[0], x1 + self.patch_size[1]
            frame = frame[y1:y2, x1:x2]
        if padding is not None:
            py, px = padding
            frame = numpy.pad(frame, pad_width=((py, py), (px, px), (0, 0)), mode='constant', constant_values=0)
        if flip:
            frame = numpy.flip(frame, axis=1)
        norm_frame = frame.astype('float32') / 255 * 2 - 1
        cf_frame = numpy.moveaxis(norm_frame, [0, 1, 2], [1, 2, 0])
        return cf_frame

    def load_test_data(self, video_name: str, seq_name: str, pred_frame_num: int):
        data_dict = self.load_testing_data(video_name, seq_name, pred_frame_num, padding=(6, 0))

        for key in data_dict.keys():
            data_dict[key] = data_dict[key][None]
        return data_dict
