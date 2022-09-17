# Shree KRISHNAya Namaha
# Command line access to OursBlenderTester01.py
# Author: Nagabhushan S N
# Last Modified: 01/11/2021

import argparse
import json
import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas
import simplejson

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

from OursBlenderTester01 import DpgVideoPredictor

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def run_dpg(test_configs_path: Path, past_frames_path: Path, next_frame_path: Path):
    with open(test_configs_path.as_posix(), 'r') as configs_file:
        test_configs = json.load(configs_file)
    past_frames = numpy.load(past_frames_path.as_posix())

    # Build and restore model
    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / f'Databases/{test_configs["database_name"]}/Data'

    train_dirpath = Path(f'../Runs/Training/Train{test_configs["train_num"]:04}')
    train_configs_path = train_dirpath / 'Configs.json'
    with open(train_configs_path.as_posix(), 'r') as configs_file:
        train_configs = simplejson.load(configs_file)
    train_configs['root_dirpath'] = root_dirpath
    # train_configs['device'] = 'gpu0'

    video_predictor = DpgVideoPredictor(root_dirpath, database_dirpath, train_configs, device='gpu0')
    video_predictor.load_model(test_configs['model_name'])

    # Run DPG
    frame2 = video_predictor.predict_next_frame1(past_frames)
    next_frame_path.parent.mkdir(parents=True, exist_ok=True)
    numpy.save(next_frame_path.as_posix(), frame2)
    return


def demo1():
    return


def demo2(args: dict):
    test_configs_path = args['test_configs_path']
    if test_configs_path is None:
        raise RuntimeError(f'Please provide test_configs_path')
    test_configs_path = Path(test_configs_path)

    past_frames_path = args['past_frames_path']
    if past_frames_path is None:
        raise RuntimeError(f'Please provide past_frames_path')
    past_frames_path = Path(past_frames_path)

    next_frame_path = args['next_frame_path']
    if next_frame_path is None:
        raise RuntimeError(f'Please provide next_frame_path')
    next_frame_path = Path(next_frame_path)

    run_dpg(test_configs_path, past_frames_path, next_frame_path)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo1')
    parser.add_argument('--test_configs_path')
    parser.add_argument('--past_frames_path')
    parser.add_argument('--next_frame_path')
    args = parser.parse_args()

    args_dict = {
        'demo_function_name': args.demo_function_name,
        'test_configs_path': args.test_configs_path,
        'past_frames_path': args.past_frames_path,
        'next_frame_path': args.next_frame_path,
    }
    return args_dict


def main():
    args = parse_args()
    if args['demo_function_name'] == 'demo1':
        demo1()
    elif args['demo_function_name'] == 'demo2':
        demo2(args)
    demo1()
    return


if __name__ == '__main__':
    main()
