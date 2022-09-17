# Shree KRISHNAya Namaha
# Wrapper
# Author: Nagabhushan S N
# Last Modified: 30/10/2021

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
import torch
import torch.nn.functional as F

from utils.Warper import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class FramePredictor(torch.nn.Module):
    def __init__(self, configs: dict, flow_predictor, occlusion_computer, occlusion_inpainter):
        super().__init__()
        self.flow_predictor = flow_predictor
        self.occlusion_computer = occlusion_computer
        self.occlusion_inpainter = occlusion_inpainter
        self.warper = Warper(configs['device'])
        self.flow_grad_required = configs['flow_predictor']['update_weights']
        self.inpaint_grad_required = configs['occlusion_inpainter']['update_weights'] if occlusion_inpainter is not None else False
        return

    def forward(self, input_batch: dict):
        result_dict = {}

        # Predict flow
        if self.flow_grad_required:
            flow_predictor_output = self.flow_predictor(input_batch)
        else:
            with torch.no_grad():
                flow_predictor_output = self.flow_predictor(input_batch)
        predicted_flow = flow_predictor_output['predicted_flow']
        result_dict['predicted_flow'] = predicted_flow

        # Compute Occlusions
        occlusion_computer_output = self.occlusion_computer(flow_predictor_output)
        occlusion_map = occlusion_computer_output['occlusion_map']
        result_dict['occlusion_map'] = occlusion_map

        # Reconstruct next frame by warping
        last_past_frame = input_batch['past_frames'][:, -1]
        warped_frame = self.warper.bilinear_interpolation(last_past_frame, None, predicted_flow, None, is_image=True)[0]
        result_dict['warped_frame'] = warped_frame

        if self.occlusion_inpainter is not None:
            # Inpaint occlusions
            inpainter_input = {
                'warped_frame': warped_frame,
                'occlusion_map': occlusion_map,
            }
            if self.inpaint_grad_required:
                inpainter_output = self.occlusion_inpainter(inpainter_input)
            else:
                with torch.no_grad():
                    inpainter_output = self.occlusion_inpainter(inpainter_input)
            predicted_frame = inpainter_output['predicted_frame']
            result_dict['predicted_frame'] = predicted_frame

        return result_dict
