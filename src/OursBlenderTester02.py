# Shree KRISHNAya Namaha
# Extended from OursBlenderTester01.py. Dataloaders are used to load the data
# Author: Nagabhushan S N
# Last Modified: 02/11/2021

import datetime
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import List

import Imath
import OpenEXR
import numpy
import pandas
import seaborn
import simplejson
import skimage.io
import skvideo.io

import torch
from tqdm import tqdm
from matplotlib import pyplot

from data_loaders.DataLoaderFactory import get_data_loader
from flow_predictors.FlowPredictorFactory import get_flow_predictor
from frame_predictors.FramePredictorFactory import get_frame_predictor
from occlusion_computers.OcclusionComputerFactory import get_occlusion_computer
from occlusion_inpainters.OcclusionInpainterFactory import get_occlusion_inpainter
from utils import CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DpgVideoPredictor:
    def __init__(self, root_dirpath: Path, database_dirpath: Path, train_configs: dict, device: str = 'gpu0'):
        self.root_dirpath = root_dirpath
        self.database_dirpath = database_dirpath
        self.train_configs = train_configs
        self.device = CommonUtils.get_device(device)

        self.frame_predictor = None
        self.data_loader = None
        self.build_model()
        self.num_frames_per_seq = 12
        self.get_data_loader()
        return

    def build_model(self):
        flow_predictor = None
        occlusion_computer = None
        occlusion_inpainter = None
        if 'flow_predictor' in self.train_configs.keys():
            flow_predictor = get_flow_predictor(self.train_configs)
        if 'occlusion_computer' in self.train_configs.keys():
            occlusion_computer = get_occlusion_computer(self.train_configs)
        if 'occlusion_inpainter' in self.train_configs.keys():
            occlusion_inpainter = get_occlusion_inpainter(self.train_configs)
        self.frame_predictor = get_frame_predictor(self.train_configs, flow_predictor, occlusion_computer, occlusion_inpainter).to(self.device)
        return

    def get_data_loader(self):
        self.data_loader = get_data_loader(self.train_configs, self.database_dirpath, None)
        return

    def load_model(self, model_name: str):
        train_num = self.train_configs["train_num"]
        full_model_name = f'Model_{model_name}.tar'
        train_dirpath = self.train_configs['root_dirpath'] / f'Runs/Training/Train{train_num:04}'
        saved_models_dirpath = train_dirpath / 'SavedModels'
        model_path = saved_models_dirpath / full_model_name
        checkpoint_state = torch.load(model_path, map_location=self.device)
        epoch_num = checkpoint_state['epoch']
        self.frame_predictor.load_state_dict(checkpoint_state['model_state_dict'])
        self.frame_predictor.eval()
        print(f'Loaded Model in Train{train_num:04}/{full_model_name} trained for {epoch_num} epochs')
        return

    def load_data(self, video_name, seq_num, pred_frame_num):
        input_data = self.data_loader.load_test_data(video_name, seq_num, pred_frame_num)
        return input_data

    def predict_next_frame1(self, input_batch: dict):
        CommonUtils.move_to_device(input_batch, self.device)
        with torch.no_grad():
            output_batch = self.frame_predictor(input_batch)
        processed_output = self.post_process_output(output_batch)

        pred_frame = self.post_process_pred_frame(processed_output['predicted_frame'])
        return pred_frame

    def predict_next_frame2(self, video_name: str, seq_num: int, pred_frame_nos: List[int]) -> List[numpy.ndarray]:
        """
        A wrapper around predict_next_frame1() that first loads data given video_name, seq_num and predicts frames
        recursively
        :return: All frames (ground truth and predicted) corresponding to the given video_name and sequence number
        """
        frames = {}
        for pred_frame_num in pred_frame_nos:
            input_dict = self.load_data(video_name, seq_num, pred_frame_num)
            next_frame = self.predict_next_frame1(input_dict)
            frames[pred_frame_num] = next_frame

        gt_frame_nos = sorted(set(range(self.num_frames_per_seq)) - set(pred_frame_nos))
        for gt_frame_num in gt_frame_nos:
            gt_frame_path = self.database_dirpath / f'all_short/RenderedData/{video_name}/seq{seq_num:02}/rgb/{gt_frame_num:04}.png'
            gt_frame = self.read_image(gt_frame_path)
            frames[gt_frame_num] = gt_frame

        all_frames = [frames[i] for i in sorted(frames.keys())]
        return all_frames

    @staticmethod
    def post_process_output(output_batch: dict):
        processed_batch = {}
        for key in output_batch.keys():
            if isinstance(output_batch[key], torch.Tensor):
                processed_batch[key] = numpy.moveaxis(output_batch[key].detach().cpu().numpy(), [0, 1, 2, 3], [0, 3, 1, 2])[0, 4:-4]
            elif isinstance(output_batch[key], list):
                processed_batch[key] = []
                for list_element in output_batch[key]:
                    if isinstance(list_element, torch.Tensor):
                        processed_batch[key].append(numpy.moveaxis(list_element.detach().cpu().numpy(), [0, 1, 2, 3], [0, 3, 1, 2])[0, 4:-4])
                    elif isinstance(list_element, tuple):
                        processed_tuple = []
                        for tuple_element in list_element:
                            if isinstance(tuple_element, torch.Tensor):
                                processed_tuple.append(numpy.moveaxis(tuple_element.detach().cpu().numpy(), [0, 1, 2, 3], [0, 3, 1, 2])[0, 4:-4])
                        processed_batch[key].append(processed_tuple)
        return processed_batch

    @staticmethod
    def post_process_pred_frame(pred_frame):
        uint8_frame = numpy.round((pred_frame + 1) * 255 / 2).astype('uint8')
        return uint8_frame

    @staticmethod
    def post_process_mask(mask: numpy.ndarray):
        bool_mask = mask[:, :, 0].astype('bool')
        return bool_mask

    @staticmethod
    def read_image(path: Path):
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def save_image(path: Path, image: numpy.ndarray):
        skimage.io.imsave(path.as_posix(), image)
        return

    @staticmethod
    def read_mask(path: Path):
        mask = skimage.io.imread(path.as_posix()) == 255
        return mask

    @staticmethod
    def read_depth(path: Path) -> numpy.ndarray:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['arr_0']
        elif path.suffix == '.exr':
            exr_file = OpenEXR.InputFile(path.as_posix())
            raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return depth


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = simplejson.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError('Configs mismatch while resuming testing')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_testing(test_configs: dict):
    test_num = test_configs['test_num']
    root_dirpath = Path('../')
    output_dirpath = root_dirpath / f'Runs/Testing/Test{test_num:04}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, test_configs)
    test_configs['root_dirpath'] = root_dirpath

    train_dirpath = Path(f'../Runs/Training/Train{test_configs["train_num"]:04}')
    train_configs_path = train_dirpath / 'Configs.json'
    with open(train_configs_path.as_posix(), 'r') as configs_file:
        train_configs = simplejson.load(configs_file)
    train_configs['root_dirpath'] = root_dirpath
    train_configs['device'] = test_configs['device']

    database_dirpath = root_dirpath / f"../../../../Databases/{test_configs['database_name']}/Data"

    test_set_num = test_configs['test_set_num']
    frames_datapath = database_dirpath / f'TrainTestSets/Set{test_set_num:02}/TestVideosData.csv'
    frames_data = pandas.read_csv(frames_datapath)

    video_predictor = DpgVideoPredictor(root_dirpath, database_dirpath, train_configs, device=test_configs['device'])
    video_predictor.load_model(test_configs['model_name'])

    videos_data = frames_data[['video_name', 'seq_num']].drop_duplicates().to_numpy()
    num_videos = videos_data.shape[0]
    pred_folder_name = test_configs['pred_folder_name']
    pred_frame_nos = [7, 9, 11]
    last_pred_frame_num = sorted(pred_frame_nos)[-1]
    for video_data in tqdm(videos_data, total=num_videos):
        video_name, seq_num = video_data

        video_output_dirpath = output_dirpath / f'{video_name}/seq{seq_num:02}'
        pred_frames_dirpath = video_output_dirpath / pred_folder_name
        pred_frames_dirpath.mkdir(parents=True, exist_ok=True)

        if not (pred_frames_dirpath / f'{last_pred_frame_num:04}.png').exists():
            all_frames = video_predictor.predict_next_frame2(video_name, seq_num, pred_frame_nos)
            for frame_num, frame in enumerate(all_frames):
                output_path = pred_frames_dirpath / f'{frame_num:04}.png'
                video_predictor.save_image(output_path, frame)

    # Run QA
    qa_filepath = Path('../../../../QA/00_Common/src/AllMetrics01_OursBlender.py')
    cmd = f'python {qa_filepath.absolute().as_posix()} ' \
          f'--demo_function_name demo2 ' \
          f'--pred_videos_dirpath {output_dirpath.absolute().as_posix()} ' \
          f'--database_dirpath {database_dirpath.absolute().as_posix()} ' \
          f'--frames_datapath {frames_datapath.absolute().as_posix()} ' \
          f'--pred_folder_name {pred_folder_name} ' \
          f'--pred_masks_dirname VSR001_01_1step_forward '
    os.system(cmd)
    return


def demo1():
    """
    Individual frames
    :return:
    """
    return


def demo2():
    test_num = 2
    train_num = 4
    configs = {
        'Tester': this_filename,
        'train_num': train_num,
        'test_num': test_num,
        'test_set_num': 7,
        'pred_folder_name': 'PredictedFrames',
        'model_name': 'Epoch100',
        'database_name': 'OursBlender',
        'database_dirpath': 'OursBlender/Data',
        'device': 'gpu0',
    }
    start_testing(configs)
    return


def demo3():
    configs = {
        'Tester': this_filename,
        'test_num': 2,
    }
    start_testing(configs)


def main():
    demo2()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

    from snb_utils import Mailer

    subject = f'VPL008/{this_filename}'
    mail_content = f'Program ended.\n' + run_result
    Mailer.send_mail(subject, mail_content)
