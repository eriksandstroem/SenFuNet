import torch
import datetime
import time

from modules.filtering_net_avg import FilteringNet
from modules.translation_net import TranslationNet
import math
import numpy as np
from scipy import ndimage

class Filter_Pipeline_mlp(torch.nn.Module):

    def __init__(self, config):

        super(Filter_Pipeline_mlp, self).__init__()

        self.config = config

        if self.config.FILTERING_MODEL.setting == 'avg':
            self._filtering_network = FilteringNet(config)
        elif self.config.FILTERING_MODEL.setting == 'translate':
            self._filtering_network = TranslationNet(config)

    def _filtering(self, neighborhood): 
        if self.config.FILTERING_MODEL.fixed:
            with torch.no_grad():
                output = self._filtering_network.forward(neighborhood)
        else:
            output = self._filtering_network.forward(neighborhood)

        return output

    def _prepare_input_training(self, volumes, indices):

        n_side_length = self.config.FILTERING_MODEL.MLP_MODEL.neighborhood
        cube_vol = pow(n_side_length, 3)
        index_cube = np.transpose(np.ones((n_side_length, n_side_length, n_side_length)).nonzero())-math.floor(n_side_length/2)*np.ones((cube_vol, 3)) # shape (125, 3)

        indices = np.expand_dims(indices, 2).repeat(cube_vol, 2)

        index_cube = np.expand_dims(index_cube.transpose(), 0).repeat(indices.shape[0], 0)

        neighborhood_indices = indices + index_cube # shape (X, 3, 27) if using 3x3x3 neighborhood, otherwise replace 27 with 125 for 5x5x5

        # pad the grids so if the index is at the border, I can still filter the border voxel when extracting its neighbors
        # if volume.dim() == 3: # tsdf
        pad = math.floor(self.config.FILTERING_MODEL.MLP_MODEL.neighborhood/2)

        if self.config.DATA.truncation_strategy == 'standard':
            pad_val = -self.config.DATA.init_value
        elif self.config.DATA.truncation_strategy == 'artificial':
            pad_val = self.config.DATA.init_value

        neighborhood = None
        for k, volume in enumerate(volumes.keys()):
            if volumes[volume].dim() == 3: # tsdf of weight volume
                if volume == 'tsdf':
                    volume = torch.nn.functional.pad(volumes[volume], (pad,pad,pad,pad,pad,pad), 'constant', pad_val)
                    n_hood = volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])].unsqueeze(-1) # shape (X, 27)
                else:
                    volume = torch.nn.functional.pad(volumes[volume], (pad,pad,pad,pad,pad,pad), 'constant', 0)
                    n_hood = volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])].unsqueeze(-1) # shape (X, 27)
    
            else: # dim 4 for feature volume
                volume = torch.transpose(volumes[volume], 0, 3)
                volume = torch.nn.functional.pad(volume, (pad,pad,pad,pad,pad,pad), 'constant', 0.0)
                volume = torch.transpose(volume, 0, 3)
                n_hood = volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]), :] # shape (X, 27)

            if k == 0:
                neighborhood = n_hood 
            else:
                neighborhood = torch.cat((neighborhood, n_hood), dim=2)

        output = neighborhood.float()
        del neighborhood

        return output

    def _prepare_input_testing_old(self, scene, database, indices):
        # TODO: do reflection padding if the central voxel is at the grid boundary. For now I do constant padding
        neighborhood = dict()

        volume = database[scene] # this yields tensors for the database. I suppose I need to use this

        n_side_length = self.config.FILTERING_MODEL.MLP_MODEL.neighborhood
        cube_vol = pow(n_side_length, 3)
        index_cube = np.transpose(np.ones((n_side_length, n_side_length, n_side_length)).nonzero())-math.floor(n_side_length/2)*np.ones((cube_vol, 3)) # shape (125, 3)
   
        indices = np.expand_dims(indices, 2).repeat(cube_vol, 2)
  
        index_cube = np.expand_dims(index_cube.transpose(), 0).repeat(indices.shape[0], 0)
    
        neighborhood_indices = (indices + index_cube).astype(int) # shape (X, 3, 27) if using 3x3x3 neighborhood, otherwise replace 27 with 125 for 5x5x5

        # we pad so that we can also filter the voxels on the border of the voxelgrid. We put those voxels to weight 0 so that they will not influence the filter.
        pad = math.floor(self.config.FILTERING_MODEL.MLP_MODEL.neighborhood/2)
        if self.config.DATA.truncation_strategy == 'standard':
            pad_val = -self.config.DATA.init_value
        elif self.config.DATA.truncation_strategy == 'artificial':
            pad_val = self.config.DATA.init_value


        # for sensor_ in self.config.DATA.input:
        #     for key in ['tsdf', 'weights']:

        #     if self.config.FILTERING_MODEL.w_features:

        tsdf_volume = torch.nn.functional.pad(volume['tsdf_tof'], (pad,pad,pad,pad,pad,pad), 'constant', pad_val)
        neighborhood_tsdf = tsdf_volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
         neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
          neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])] # shape (X, 27)
        del tsdf_volume
        weights_volume = torch.nn.functional.pad(volume['weights_tof'], (pad,pad,pad,pad,pad,pad), 'constant', 0.0)
        neighborhood_weights = weights_volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
         neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
          neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])] # shape (X, 27)
        del weights_volume
        features_volume = torch.transpose(volume['features_tof'], 0, 3)
        features_volume = torch.nn.functional.pad(features_volume, (pad,pad,pad,pad,pad,pad), 'constant', 0.0)
        features_volume = torch.transpose(features_volume, 0, 3)
        neighborhood_features= features_volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
            neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
            neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]), :] # shape (X, 27)
        del features_volume
        # feature_weights_volume = torch.nn.functional.pad(volume['feature_weights_tof'], (pad,pad,pad,pad,pad,pad), 'constant', 0.0)
        # neighborhood_feature_weights = feature_weights_volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
        #  neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
        #   neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])] # shape (X, 27)
        # del feature_weights_volume


        neighborhood['tof'] = torch.cat([neighborhood_tsdf.unsqueeze(-1), 
                neighborhood_weights.unsqueeze(-1), neighborhood_features], dim=2).float().to(self.device) # shape (X, 27, 3)

        del neighborhood_tsdf, neighborhood_weights, neighborhood_features #, neighborhood_feature_weights


        tsdf_volume_stereo = torch.nn.functional.pad(volume['tsdf_stereo'], (pad,pad,pad,pad,pad,pad), 'constant', pad_val)
        neighborhood_tsdf = tsdf_volume_stereo[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
         neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
          neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])] # shape (X, 27)
        del tsdf_volume_stereo
        weights_volume_stereo = torch.nn.functional.pad(volume['weights_stereo'], (pad,pad,pad,pad,pad,pad), 'constant', 0.0)
        neighborhood_weights = weights_volume_stereo[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
         neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
          neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])] # shape (X, 27)
        del weights_volume_stereo
        features_volume = torch.transpose(volume['features_stereo'], 0, 3)
        features_volume = torch.nn.functional.pad(features_volume, (pad,pad,pad,pad,pad,pad), 'constant', 0.0)
        features_volume = torch.transpose(features_volume, 0, 3)
        neighborhood_features = features_volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
            neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
            neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]), :] # shape (X, 27)
        del features_volume
        # feature_weights_volume = torch.nn.functional.pad(volume['feature_weights_stereo'], (pad,pad,pad,pad,pad,pad), 'constant', 0.0)
        # neighborhood_feature_weights = feature_weights_volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
        #  neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
        #   neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])] # shape (X, 27)
        # del feature_weights_volume

  
        neighborhood['stereo'] = torch.cat([neighborhood_tsdf.unsqueeze(-1), 
                neighborhood_weights.unsqueeze(-1), neighborhood_features], dim=2).float().to(self.device) # shape (X, 27, 3)

        del neighborhood_tsdf, neighborhood_weights, neighborhood_features #, neighborhood_feature_weights

        return neighborhood

    def _prepare_input_testing(self, scene, database, indices):
        # TODO: do reflection padding if the central voxel is at the grid boundary. For now I do constant padding
        neighborhood = dict()

        volume = database[scene] # this yields tensors for the database. I suppose I need to use this

        n_side_length = self.config.FILTERING_MODEL.MLP_MODEL.neighborhood
        cube_vol = pow(n_side_length, 3)
        index_cube = np.transpose(np.ones((n_side_length, n_side_length, n_side_length)).nonzero())-math.floor(n_side_length/2)*np.ones((cube_vol, 3)) # shape (125, 3)
   
        indices = np.expand_dims(indices, 2).repeat(cube_vol, 2)
  
        index_cube = np.expand_dims(index_cube.transpose(), 0).repeat(indices.shape[0], 0)
    
        neighborhood_indices = (indices + index_cube).astype(int) # shape (X, 3, 27) if using 3x3x3 neighborhood, otherwise replace 27 with 125 for 5x5x5

        # we pad so that we can also filter the voxels on the border of the voxelgrid. We put those voxels to weight 0 so that they will not influence the filter.
        pad = math.floor(self.config.FILTERING_MODEL.MLP_MODEL.neighborhood/2)
        if self.config.DATA.truncation_strategy == 'standard':
            pad_val = -self.config.DATA.init_value
        elif self.config.DATA.truncation_strategy == 'artificial':
            pad_val = self.config.DATA.init_value

        for sensor_ in self.config.DATA.input:
            sensor_neighborhood = None
            for key in ['tsdf', 'weights']: # note that we feed the tsdf weights here and not features weights. The tsdf weights are 
            # positive at more locations than the feature weights due to the online outlier filter, but during test time 
            # we use the feature weights to get the mask for the filtering stage, so it does not matter that we use the 
            # tsdf weights
                if key == 'tsdf':
                    vol = torch.nn.functional.pad(volume[key + '_' + sensor_], (pad,pad,pad,pad,pad,pad), 'constant', pad_val)
                    n_hood = vol[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])] # shape (X, 27)
                    sensor_neighborhood = n_hood.unsqueeze(-1)
                else:
                    vol = torch.nn.functional.pad(volume[key + '_' + sensor_], (pad,pad,pad,pad,pad,pad), 'constant', 0)
                    n_hood = vol[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                        neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])] # shape (X, 27)
                    sensor_neighborhood = torch.cat((sensor_neighborhood, n_hood.unsqueeze(-1)), dim=2)
            if self.config.FILTERING_MODEL.features_to_sdf_enc:
                vol = torch.transpose(volume['features_' + sensor_], 0, 3)
                vol = torch.nn.functional.pad(vol, (pad,pad,pad,pad,pad,pad), 'constant', 0.0)
                vol = torch.transpose(vol, 0, 3)
                n_hood = vol[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                    neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
                    neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]), :] # shape (X, 27)
                sensor_neighborhood = torch.cat((sensor_neighborhood, n_hood), dim=2)
    
        # feature_weights_volume = torch.nn.functional.pad(volume['feature_weights_tof'], (pad,pad,pad,pad,pad,pad), 'constant', 0.0)
        # neighborhood_feature_weights = feature_weights_volume[neighborhood_indices[:, 0, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
        #  neighborhood_indices[:, 1, :] + pad*np.ones_like(neighborhood_indices[:, 0, :]),
        #   neighborhood_indices[:, 2, :] + pad*np.ones_like(neighborhood_indices[:, 0, :])] # shape (X, 27)
        # del feature_weights_volume

            neighborhood[sensor_] = sensor_neighborhood.float().to(self.device) # shape (X, 27, 3)

        return neighborhood

    def filter(self, scene, database, device):
        self.device = device
        # why do I have astype in16? Memory saveing? Well yeah I suppose I can save a bool as int16
        indices = np.zeros_like(database.scenes_gt[scene].volume).astype(np.int16)
        for sensor_ in self.config.DATA.input:
            # there should not really be a difference if we use the feature weights here or the weights
            # which include the online outlier filter since the bbox does not really change 
            indices = np.logical_or(indices, database.feature_weights[sensor_][scene] > 0).astype(np.int16) # - indices_and

        if self.config.FILTERING_MODEL.erosion:
            # if self.config.FILTERING_MODEL.neighborhood == 3:
            #     # erode indices mask once
            indices = ndimage.binary_erosion(indices, structure=np.ones((3,3,3)), iterations=1)
            # else:
                # erode indices mask twice
                # indices = ndimage.binary_erosion(indices, structure=np.ones((3,3,3)), iterations=2)

        indices = np.transpose(indices.nonzero()).astype(np.int16)

        idxList = list()
        for i in range(1, math.ceil(indices.shape[0]/300000)):
            idxList.append(i*300000)
        # print(valid_indices.shape)
        # print(math.ceil(valid_indices.shape[0]/5000000))
        indices = np.array_split(indices, idxList, 0)
        # print(len(indices))
        for i in range(len(indices)):
            # print(torch.cuda.max_memory_allocated(device))
            # print(indices[i].shape)
            # this function first calls _prepare_translation_input which prepares the vertices
            # where we have weight > 0. It also stores the indices for those vertices. 
            input_ = self._prepare_input_testing(scene, database, indices[i])

            with torch.no_grad():
                # sub_tsdf_filter = self._filtering(input_['neighborhood_filter'])[0]
                sub_output = self._filtering(input_)
            del input_

            sub_tsdf = sub_output['tsdf'].cpu().detach().numpy().squeeze() # + sub_tsdf_translate.cpu().detach().numpy().squeeze()

            if len(self.config.DATA.input) > 1 and self.config.SETTINGS.test_mode:
                database.sensor_weighting[scene].volume[indices[i][:, 0], 
                                    indices[i][:, 1], indices[i][:, 2]] = sub_output['sensor_weighting'].cpu().detach().numpy().squeeze()
            
            del sub_output
            sub_tsdf = np.clip(sub_tsdf,
                            -self.config.DATA.trunc_value,
                            self.config.DATA.trunc_value)

            database.filtered[scene].volume[indices[i][:, 0], indices[i][:, 1], indices[i][:, 2]] = sub_tsdf
            del sub_tsdf

            # break
        # neighborhood = torch.split(input_['neighborhood'], 5000000, 0)
        # center = torch.split(input_['center'], 5000000, 0)

        # print(valid_indices)


    def filter_training(self, input_dir, database, scene_id, sensor, device): 
 
        indices = input_dir['indices'].cpu()
        del input_dir['indices']

        idx = torch.randperm(indices.shape[0])
        indices = indices[idx]
        indices = indices[:10000, :]

        neighborhood = dict()
        for sensor_ in self.config.DATA.input:
            if sensor_ == sensor: 
                in_dir = {'tsdf': input_dir['tsdf'],
                            'weights': input_dir['weights']}
                if self.config.FILTERING_MODEL.features_to_sdf_enc:
                    in_dir['features'] = input_dir['features']
            else:
                in_dir = {'tsdf': database[scene_id]['tsdf_' + sensor_].to(device),
                            'weights': database[scene_id]['weights_' + sensor_].to(device)}
                if self.config.FILTERING_MODEL.features_to_sdf_enc:
                    in_dir['features'] = database[scene_id]['features_' + sensor_].to(device)
            neighborhood[sensor_] = self._prepare_input_training(in_dir, indices) 
 
        tsdf_filtered = self._filtering(neighborhood)
        del neighborhood

        if self.config.LOSS.gt_loss:
            raise NotImplementedError
            # Here I use feat_vol just for now, but ideally, I should use a different feat_vol for when feeding gt depth frames so
            # that the feature network also can be trained on gt images because it is not clear what feature the gt depth images correspond to
            # for now I make sure that I don't use gt loss when using features
            input_1 = self._prepare_input_training(gt_vol.to(device), self.config.FUSION_MODEL.max_weight*torch.ones_like(weights_vol), feat_vol, bbox) 

            input_2 = self._prepare_input_training(gt_vol.to(device), self.config.FUSION_MODEL.max_weight*torch.ones_like(weights_vol), feat_vol_op, bbox) 

            neighborhood = dict()
            neighborhood[sensor] = input_1['neighborhood']
            neighborhood[sensor_op] = input_2['neighborhood']
     
            tsdf_gt_filtered = self._filtering(neighborhood)
        else:
            tsdf_gt_filtered = None

        gt_vol = database[scene_id]['gt']
        tsdf_target = gt_vol[indices[:, 0], indices[:, 1], indices[:, 2]]
        # print(sub_tsdf_target.shape)
        tsdf_target = tsdf_target.float()

        del gt_vol

        tsdf_target = tsdf_target.to(device)
        output = dict()
        output['tsdf_filtered_grid'] = tsdf_filtered
        output['tsdf_gt_filtered_grid'] = tsdf_gt_filtered
        output['tsdf_target_grid'] = tsdf_target #.squeeze()

        return output