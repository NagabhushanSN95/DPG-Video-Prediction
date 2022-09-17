# Shree KRISHNAya Namaha
# Trainer for OursBlender Database. Modified from SintelTrainer01.py
# Author: Nagabhushan S N
# Last Modified: 31/10/2021

import datetime
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
import torch
from matplotlib import pyplot
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data_loaders.DataLoaderFactory import get_data_loader
from flow_predictors.FlowPredictorFactory import get_flow_predictor
from occlusion_computers.OcclusionComputerFactory import get_occlusion_computer
from occlusion_inpainters.OcclusionInpainterFactory import get_occlusion_inpainter
from frame_predictors.FramePredictorFactory import get_frame_predictor
from loss_functions.LossComputer01 import LossComputer
from optimizers.OptimizerFactory import get_optimizer
from optimizers.SchedulererFactory import get_scheduler
from utils import CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, train_num: int, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, frame_predictor,
                 loss_computer: LossComputer, optimizer, scheduler, output_dirpath: Path, device: str, configs: dict,
                 verbose_log: bool = True):
        self.train_num = train_num
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_data_loader = None
        self.val_data_loader = None
        self.frame_predictor = frame_predictor
        self.loss_computer = loss_computer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = CommonUtils.get_device(device)
        self.output_dirpath = output_dirpath
        self.configs = configs
        self.verbose_log = verbose_log

        self.frame_predictor.to(self.device)
        if 'pretrained_model_paths' in configs['frame_predictor'].keys():
            # TODO: Check if this works fine
            assert 1 <= len(configs['frame_predictor']['pretrained_model_paths']) <= 2
            for path in configs['frame_predictor']['pretrained_model_paths']:
                weights_path = configs['root_dirpath'] / path
                self.load_pretrained_weights(weights_path)
        return

    def train_one_epoch(self):
        def update_losses_dict_(epoch_losses_dict_: dict, iter_losses_dict_: dict, num_samples_: int):
            if epoch_losses_dict_ is None:
                epoch_losses_dict_ = {}
                for loss_name_ in iter_losses_dict_.keys():
                    epoch_losses_dict_[loss_name_] = iter_losses_dict_[loss_name_].item() * num_samples_
            else:
                for loss_name_ in epoch_losses_dict_.keys():
                    epoch_losses_dict_[loss_name_] += (iter_losses_dict_[loss_name_].item() * num_samples_)
            return epoch_losses_dict_

        epoch_losses_dict = None

        total_num_samples = 0
        for iter_num, input_batch in enumerate(tqdm(self.train_data_loader, leave=self.verbose_log)):
            CommonUtils.move_to_device(input_batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            output_batch = self.frame_predictor(input_batch)
            iter_losses_dict = self.loss_computer.compute_losses(input_batch, output_batch)
            iter_losses_dict['TotalLoss'].backward()
            self.optimizer.step()

            # Update losses and the number of samples
            input_batch_first_element = input_batch[list(input_batch.keys())[0]]
            if isinstance(input_batch_first_element, torch.Tensor):
                batch_num_samples = input_batch_first_element.shape[0]
            elif isinstance(input_batch_first_element, list):
                batch_num_samples = len(input_batch_first_element)
            else:
                raise RuntimeError('Please help me! I don\'t know how to compute num_samples')
            epoch_losses_dict = update_losses_dict_(epoch_losses_dict, iter_losses_dict, batch_num_samples)
            total_num_samples += batch_num_samples

        for loss_name in epoch_losses_dict.keys():
            epoch_losses_dict[loss_name] = epoch_losses_dict[loss_name] / total_num_samples

        if self.scheduler is not None:
            self.scheduler.step()
        return epoch_losses_dict

    def validation_one_epoch(self):
        def update_losses_dict_(epoch_losses_dict_: dict, iter_losses_dict_: dict, num_samples_: int):
            if epoch_losses_dict_ is None:
                epoch_losses_dict_ = {}
                for loss_name_ in iter_losses_dict_.keys():
                    epoch_losses_dict_[loss_name_] = iter_losses_dict_[loss_name_].item() * num_samples_
            else:
                for loss_name_ in epoch_losses_dict_.keys():
                    epoch_losses_dict_[loss_name_] += (iter_losses_dict_[loss_name_].item() * num_samples_)
            return epoch_losses_dict_

        epoch_losses_dict = None
        total_num_samples = 0
        self.frame_predictor.eval()
        with torch.no_grad():
            for iter_num, input_batch in enumerate(tqdm(self.val_data_loader, leave=False)):
                CommonUtils.move_to_device(input_batch, self.device)
                output_batch = self.frame_predictor(input_batch)
                iter_losses_dict = self.loss_computer.compute_losses(input_batch, output_batch)

                # Update losses and the number of samples
                input_batch_first_element = input_batch[list(input_batch.keys())[0]]
                if isinstance(input_batch_first_element, torch.Tensor):
                    batch_num_samples = input_batch_first_element.shape[0]
                elif isinstance(input_batch_first_element, list):
                    batch_num_samples = len(input_batch_first_element)
                else:
                    raise RuntimeError('Please help me! I don\'t know how to compute num_samples')
                epoch_losses_dict = update_losses_dict_(epoch_losses_dict, iter_losses_dict, batch_num_samples)
                total_num_samples += batch_num_samples

                if total_num_samples >= 500:
                    break
        for loss_name in epoch_losses_dict.keys():
            epoch_losses_dict[loss_name] = epoch_losses_dict[loss_name] / total_num_samples
        self.frame_predictor.train()
        return epoch_losses_dict

    def train(self, num_epochs: int):
        def update_losses_data_(epoch_num_: int, epoch_losses_: dict, cumulative_losses_: pandas.DataFrame,
                                save_path_: Path):
            curr_time = datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p')
            epoch_data = [epoch_num_, curr_time] + list(epoch_losses_.values())
            if cumulative_losses_ is None:
                loss_names = list(epoch_losses_.keys())
                column_names = ['Epoch Num', 'Time'] + loss_names
                cumulative_losses_ = pandas.DataFrame([epoch_data], columns=column_names)
            else:
                num_curr_rows = cumulative_losses_.shape[0]
                cumulative_losses_.loc[num_curr_rows] = epoch_data
            cumulative_losses_.to_csv(save_path_, index=False)
            return cumulative_losses_

        def print_losses(epoch_num_: int, epoch_losses_: dict, train_: bool):
            curr_time = datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p')
            if train_:
                log_string = 'Train '
            else:
                log_string = 'Validation '
            log_string += f'{epoch_num_:03}: {curr_time}; '
            for loss_name in epoch_losses_.keys():
                log_string += f'{loss_name}: {epoch_losses_[loss_name]:0.06f}; '
            print(log_string, flush=True)
            time.sleep(1)
            return

        print('Training begins...')
        loss_plots_dirpath = self.output_dirpath / 'LossPlots'
        sample_images_dirpath = self.output_dirpath / 'Samples'
        saved_models_dirpath = self.output_dirpath / 'SavedModels'
        loss_plots_dirpath.mkdir(exist_ok=True)
        sample_images_dirpath.mkdir(exist_ok=True)
        saved_models_dirpath.mkdir(exist_ok=True)

        train_losses_path = loss_plots_dirpath / 'TrainLosses.csv'
        val_losses_path = loss_plots_dirpath / 'ValidationLosses.csv'
        train_losses_data = pandas.read_csv(train_losses_path) if train_losses_path.exists() else None
        val_losses_data = pandas.read_csv(val_losses_path) if val_losses_path.exists() else None

        batch_size = self.configs['batch_size']
        sample_save_interval = self.configs['sample_save_interval']
        model_save_interval = self.configs['model_save_interval']
        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True,
                                            pin_memory=True, num_workers=4)
        self.val_data_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, shuffle=True,
                                          pin_memory=True, num_workers=4)
        start_epoch_num = self.load_model(saved_models_dirpath)
        for epoch_num in range(start_epoch_num, num_epochs):
            epoch_train_loss = self.train_one_epoch()
            train_losses_data = update_losses_data_(epoch_num + 1, epoch_train_loss, train_losses_data,
                                                    train_losses_path)
            print_losses(epoch_num + 1, epoch_train_loss, train_=True)
            epoch_val_loss = self.validation_one_epoch()
            val_losses_data = update_losses_data_(epoch_num + 1, epoch_val_loss, val_losses_data, val_losses_path)
            print_losses(epoch_num + 1, epoch_val_loss, train_=False)

            if (epoch_num + 1) % sample_save_interval == 0:
                self.save_sample_images(epoch_num + 1, sample_images_dirpath)

            if (epoch_num + 1) % model_save_interval == 0:
                self.save_model(epoch_num + 1, saved_models_dirpath)

            # Save model after every epoch
            self.save_model(epoch_num + 1, saved_models_dirpath, label='Latest')
        save_plots(loss_plots_dirpath, train_losses_path, prefix='Train')
        save_plots(loss_plots_dirpath, val_losses_path, prefix='Validation')
        return

    def save_sample_images(self, epoch_num, save_dirpath):
        def convert_tensor_to_image_(tensor_batch_):
            np_array = tensor_batch_.detach().cpu().numpy()
            image_batch = (numpy.moveaxis((np_array + 1) * 255 / 2, [0, 1, 2, 3], [0, 3, 1, 2])).astype('uint8')
            return image_batch

        def create_collage_(input_batch_, output_batch_):
            last_frame = input_batch_['past_frames'][:4, -1]
            warped_frame = output_batch_['warped_frame'][:4]
            occlusion_map = output_batch_['occlusion_map'][:4].repeat(1, 3, 1, 1) * 2 - 1
            sample_images = [last_frame, warped_frame, occlusion_map]
            if 'predicted_frame' in output_batch_.keys():
                predicted_frame = output_batch_['predicted_frame'][:4]
                sample_images.append(predicted_frame)
            target_frame = input_batch_['target_frame'][:4]
            sample_images.append(target_frame)

            for i in range(len(sample_images)):
                # noinspection PyTypeChecker
                numpy_batch = convert_tensor_to_image_(sample_images[i])
                padded = numpy.pad(numpy_batch, ((0, 0), (5, 5), (5, 5), (0, 0)), mode='constant', constant_values=255)
                sample_images[i] = numpy.concatenate(padded, axis=0)
            sample_collage_ = numpy.concatenate(sample_images, axis=1)
            return sample_collage_

        self.frame_predictor.eval()
        with torch.no_grad():
            # train set samples
            train_input_batch = next(self.train_data_loader.__iter__())
            CommonUtils.move_to_device(train_input_batch, self.device)
            train_output_batch = self.frame_predictor(train_input_batch)
            train_sample_collage = create_collage_(train_input_batch, train_output_batch)

            # validation set samples
            val_input_batch = next(self.val_data_loader.__iter__())
            CommonUtils.move_to_device(val_input_batch, self.device)
            val_output_batch = self.frame_predictor(val_input_batch)
            val_sample_collage = create_collage_(val_input_batch, val_output_batch)
        self.frame_predictor.train()

        save_path = save_dirpath / f'Epoch_{epoch_num:03}.png'
        sample_collage = numpy.concatenate([train_sample_collage, val_sample_collage], axis=0)
        skimage.io.imsave(save_path.as_posix(), sample_collage)
        return

    def save_model(self, epoch_num: int, save_dirpath: Path, label: str = None):
        if label is None:
            label = f'Epoch{epoch_num:03}'
        save_path = save_dirpath / f'Model_{label}.tar'
        checkpoint_state = {
            'epoch': epoch_num,
            'model_state_dict': self.frame_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint_state, save_path)
        return

    def load_model(self, saved_models_dirpath: Path):
        latest_model_path = saved_models_dirpath / 'Model_Latest.tar'
        if latest_model_path.exists():
            if self.device.type == 'cpu':
                checkpoint_state = torch.load(latest_model_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(latest_model_path)
            epoch_num = checkpoint_state['epoch']
            self.frame_predictor.load_state_dict(checkpoint_state['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
            print(f'Resuming Training from epoch {epoch_num + 1}')
        else:
            epoch_num = 0
        return epoch_num

    def load_pretrained_weights(self, weights_path: Path):
        if weights_path.exists():
            checkpoint_state = torch.load(weights_path, map_location=self.device)
            self.frame_predictor.load_state_dict(checkpoint_state['model_state_dict'], strict=False)
            print(f'Loaded pretrained weights of {weights_path.as_posix()}')
        return


def save_plots(save_dirpath: Path, loss_data_path: Path, prefix):
    loss_data = pandas.read_csv(loss_data_path)
    epoch_nums = loss_data['Epoch Num']
    for loss_name in loss_data.keys()[2:]:
        loss_values = loss_data[loss_name]
        save_path = save_dirpath / f'{prefix}_{loss_name}.png'
        pyplot.plot(epoch_nums, loss_values)
        pyplot.savefig(save_path)
        pyplot.close()
    return


def init_seeds(seed: int = 1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    return


def save_configs(output_dirpath: Path, configs: dict):
    # Save configs
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        # If resume_training is false, an error would've been raised when creating output directory. No need to handle
        # it here.
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        configs['seed'] = old_configs['seed']
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs['num_epochs'] > old_configs['num_epochs']:
            old_configs['num_epochs'] = configs['num_epochs']
        if configs != old_configs:
            raise RuntimeError('Configs mismatch while resuming training')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_training(configs: dict):
    root_dirpath = Path('../')
    project_dirpath = Path('../../../../../')

    # Setup output dirpath
    train_num = configs['train_num']
    output_dirpath = root_dirpath / f'Runs/Training/Train{train_num:04}'
    output_dirpath.mkdir(parents=True, exist_ok=configs['resume_training'])
    save_configs(output_dirpath, configs)
    init_seeds(configs['seed'])

    # Create data_loaders, models, optimizers etc
    configs['root_dirpath'] = root_dirpath
    database_dirpath = project_dirpath / configs['database_dirpath']
    set_num = configs['data_loader']['train_set_num']
    train_videos_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TrainVideosData.csv'
    val_videos_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/ValidationVideosData.csv'
    train_dataset = get_data_loader(configs, database_dirpath, train_videos_datapath)
    val_dataset = get_data_loader(configs, database_dirpath, val_videos_datapath)
    flow_predictor = get_flow_predictor(configs)
    occlusion_computer = get_occlusion_computer(configs)
    inpainter = get_occlusion_inpainter(configs)
    frame_predictor = get_frame_predictor(configs, flow_predictor=flow_predictor, occlusion_computer=occlusion_computer,
                                          occlusion_inpainter=inpainter)
    loss_computer = LossComputer(configs)
    optimizer = get_optimizer(configs, list(frame_predictor.parameters()))
    scheduler = get_scheduler(configs, optimizer)

    # Start training
    trainer = Trainer(train_num, train_dataset, val_dataset, frame_predictor, loss_computer, optimizer, scheduler, output_dirpath,
                      configs['device'], configs)
    trainer.train(num_epochs=configs['num_epochs'])

    del trainer, optimizer, scheduler, loss_computer, frame_predictor, flow_predictor, occlusion_computer, inpainter
    torch.cuda.empty_cache()
    return


def demo1():
    train_configs = {
        'trainer': this_filename,
        'train_num': 7,
        'database': 'OursBlender',
        'database_dirpath': 'Databases/OursBlender/Data',
        'data_loader': {
            'name': 'OursBlender04',
            'patch_size': [256, 256],
            'train_set_num': 7,
            'num_past_frames': 2,
            'upsampling_factor': 2,
            'video_duration': 'short',
            'numpy_mmap_mode': None,
            'cache_data': False,
        },
        'flow_predictor': {
            'name': 'Unet03',
            'update_weights': True,
        },
        'occlusion_computer': {
            'name': 'OcclusionComputer01',
        },
        'occlusion_inpainter': None,
        'frame_predictor': {
            'name': 'FramePredictor01',
        },
        'losses': [
            {
                'name': 'PixelLoss01',
                'weight': 1,
                'alpha': 0.9,
            },
            {
                'name': 'SmoothnessLoss01',
                'weight': 0.1,
                'flow_weight_coefficient': 1,
            },
        ],
        'optimizer': {
            'name': 'adam',
            'lr': 0.0001,
            'beta1': 0.9,
            'beta2': 0.999,
        },
        'scheduler': {
            'name': 'MultiStepLR',
            'milestones': [50, 75],
            'gamma': 0.1,
        },
        'resume_training': True,
        'num_epochs': 100,
        'batch_size': 8,
        'sample_save_interval': 1,
        'model_save_interval': 25,
        'seed': numpy.random.randint(1000),
        'device': 'gpu0',
    }
    start_training(train_configs)

    train_configs = {
        'trainer': this_filename,
        'train_num': 8,
        'database': 'OursBlender',
        'database_dirpath': 'Databases/OursBlender/Data',
        'data_loader': {
            'name': 'OursBlender04',
            'patch_size': [256, 256],
            'train_set_num': 7,
            'num_past_frames': 2,
            'upsampling_factor': 2,
            'video_duration': 'short',
            'numpy_mmap_mode': None,
            'cache_data': False,
        },
        'flow_predictor': {
            'name': 'Unet03',
            'update_weights': False,
        },
        'occlusion_computer': {
            'name': 'OcclusionComputer01',
        },
        'occlusion_inpainter': {
            'name': 'Unet01',
            'update_weights': True,
        },
        'frame_predictor': {
            'name': 'FramePredictor01',
            'pretrained_model_paths': ['Runs/Training/Train0007/SavedModels/Model_Epoch100.tar'],
        },
        'losses': [
            {
                'name': 'PixelLoss02',
                'weight': 1,
                'alpha': 0.9,
                'beta': 10,
            },
            {
                'name': 'PerceptualAndStyleLoss01',
                'weight': 1,
                'beta': 10,
                'perceptual_loss_weight': 0.05,
                'style_loss_weight': 120,
            },
            {
                'name': 'TotalVariationLoss01',
                'weight': 0.1,
                'alpha': 0.9,
                'beta': 10,
            },
        ],
        'optimizer': {
            'name': 'adam',
            'lr': 0.0001,
            'beta1': 0.9,
            'beta2': 0.999,
        },
        'scheduler': {
            'name': 'MultiStepLR',
            'milestones': [50, 75],
            'gamma': 0.1,
        },
        'resume_training': True,
        'num_epochs': 100,
        'batch_size': 4,
        'sample_save_interval': 1,
        'model_save_interval': 25,
        'seed': numpy.random.randint(1000),
        'device': 'gpu0',
    }
    start_training(train_configs)
    return


def demo2():
    configs = {
        'trainer': this_filename,
        'train_num': 4,
        'resume_training': True,
    }
    start_training(configs)
    return


def demo3():
    """
    Saves plots mid training
    :return:
    """
    train_num = 1
    loss_plots_dirpath = Path(f'../Runs/Training/Train{train_num:04}/LossPlots')
    train_losses_path = loss_plots_dirpath / 'TrainLosses.csv'
    val_losses_path = loss_plots_dirpath / 'ValidationLosses.csv'
    save_plots(loss_plots_dirpath, train_losses_path, prefix='Train')
    save_plots(loss_plots_dirpath, val_losses_path, prefix='Validation')
    import sys
    sys.exit(0)


def main():
    demo1()
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
