import torch
import datetime
import time
from tqdm import tqdm
import random

from modules.fuse_pipeline import Fuse_Pipeline
from modules.filter_pipeline import Filter_Pipeline
from modules.filter_pipeline_mlp import Filter_Pipeline_mlp

import math
import numpy as np
from scipy import ndimage

class Pipeline(torch.nn.Module):

    def __init__(self, config):

        super(Pipeline, self).__init__()

        self.config = config

        # setup pipeline
        self.fuse_pipeline = Fuse_Pipeline(config)
        if config.FILTERING_MODEL.model == 'mlp':
            self.filter_pipeline = Filter_Pipeline_mlp(config)
        elif config.FILTERING_MODEL.model == '3dconv': 
            self.filter_pipeline = Filter_Pipeline(config)


    def forward(self, batch, database, device): # train step
        scene_id = batch['frame_id'][0].split('/')[0]

        fused_output = self.fuse_pipeline.fuse_training(batch, database, device)
        filtered_output = self.filter_pipeline.filter_training(fused_output, database, scene_id, batch['sensor'], device)
        
        if filtered_output == 'save_and_exit':
            return 'save_and_exit'

        if filtered_output is not None:
            fused_output['filtered_output'] = filtered_output
        else:
            return None

        return fused_output

    def test(self, val_loader, val_dataset, val_database, sensors, device):
                
        for k, batch in tqdm(enumerate(val_loader), total=len(val_dataset)):
            # validation step - fusion
            # randomly integrate the selected sensors
            # random.shuffle(sensors)
            for sensor_ in sensors:
                # print(sensor_)
                batch['depth'] = batch[sensor_ + '_depth']
                # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_) 
                batch['routing_net'] = 'self._routing_network_' + sensor_
                batch['mask'] = batch[sensor_ + '_mask']
                batch['sensor'] = sensor_
                output = self.fuse_pipeline.fuse(batch, val_database, device)

            if k == 5:
                break # debug

   
        # run filtering network on all voxels which have a non-zero weight
        for scene in val_database.filtered.keys():   
            self.filter_pipeline.filter(scene, val_database, device)
    