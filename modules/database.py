import os
import h5py

import numpy as np

from torch.utils.data import Dataset
from graphics import Voxelgrid
import trimesh
import skimage.measure
from scipy import ndimage


from utils.metrics import evaluation


class Database(Dataset):

    def __init__(self, dataset, config):

        super(Database, self).__init__()

        self.transform = config.transform
        self.initial_value = config.init_value
        self.trunc_value = config.trunc_value
        self.erosion = config.erosion
        self.n_features = config.n_features # this includes the append_depth option
        self.sensors = config.input
        self.w_features = config.features_to_sdf_enc or config.features_to_weight_head
        self.test_mode = config.test_mode

        self.scenes_gt = {}
        self.tsdf = {}
        self.fusion_weights = {}
        self.features = {}
        self.feature_weights = {}
        for sensor in config.input:
            self.tsdf[sensor] = {}
            self.fusion_weights[sensor] = {}
            # if config.w_features:# TODO: adapt to when not using features
            self.features[sensor] = {}
            self.feature_weights[sensor] = {}
    
        self.filtered = {} # grid to store the final sdf prediction
        if len(config.input) > 1 and config.test_mode:
            self.sensor_weighting = {}

        for s in dataset.scenes:
            grid = dataset.get_grid(s, truncation=self.trunc_value)
            self.scenes_gt[s] = grid

            init_volume = self.initial_value * np.ones_like(grid.volume, dtype=np.float16)

            for sensor in config.input:
                self.tsdf[sensor][s] = Voxelgrid(self.scenes_gt[s].resolution)
                self.tsdf[sensor][s].from_array(init_volume, self.scenes_gt[s].bbox)
                self.fusion_weights[sensor][s] = np.zeros(self.scenes_gt[s].volume.shape, dtype=np.float16)

                # if config.w_features:# TODO: adapt to when not using features
                fusion_feature_shape = (self.scenes_gt[s].volume.shape[0], self.scenes_gt[s].volume.shape[1], self.scenes_gt[s].volume.shape[2], self.n_features)
                self.features[sensor][s] = np.zeros(fusion_feature_shape, dtype=np.float16)
                self.feature_weights[sensor][s] = np.zeros(self.scenes_gt[s].volume.shape, dtype=np.float16)

            self.filtered[s] = Voxelgrid(self.scenes_gt[s].resolution)
            self.filtered[s].from_array(init_volume, self.scenes_gt[s].bbox)
            if len(config.input) > 1 and config.test_mode:
                self.sensor_weighting[s] = Voxelgrid(self.scenes_gt[s].resolution)
                # initialize to negative so that we know what values are initialized without needing the mask later in the visualization script
                self.sensor_weighting[s].from_array(-np.ones_like(grid.volume, dtype=np.float16), self.scenes_gt[s].bbox) 

        
        # self.reset()

    def __getitem__(self, item):

        sample = dict()

        sample['gt'] = self.scenes_gt[item].volume
        sample['origin'] = self.scenes_gt[item].origin
        sample['resolution'] = self.scenes_gt[item].resolution
        sample['filtered'] = self.filtered[item]
        if len(self.sensors) > 1 and self.test_mode:
            sample['sensor_weighting'] = self.sensor_weighting[item]
        for sensor in self.sensors:
            sample['tsdf_' + sensor] = self.tsdf[sensor][item].volume
            sample['weights_' + sensor] = self.fusion_weights[sensor][item]
            # if self.w_features:# TODO: adapt to when not using features
            sample['features_' + sensor] = self.features[sensor][item]
            sample['feature_weights_' + sensor] = self.feature_weights[sensor][item]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.scenes_gt)

    def save(self, path, scene_id=None):

        for sensor in self.sensors:
            filename = scene_id + '_' + sensor + '.tsdf.hf5'
            weightname = scene_id + '_' + sensor + '.weights.hf5'
            featurename = scene_id + '_' + sensor + '.features.hf5'

            with h5py.File(os.path.join(path, filename), 'w') as hf:
                hf.create_dataset("TSDF",
                                      shape=self.tsdf[sensor][scene_id].volume.shape,
                                      data=self.tsdf[sensor][scene_id].volume,
                                      compression='gzip',
                                      compression_opts=9)
            with h5py.File(os.path.join(path, weightname), 'w') as hf:
                hf.create_dataset("weights",
                                      shape=self.feature_weights[sensor][scene_id].shape, # NOTE MAYBE CHANGE LATER TO FUSION WEIGHTS?
                                      data=self.feature_weights[sensor][scene_id],
                                      compression='gzip',
                                      compression_opts=9)
            if self.w_features:
                with h5py.File(os.path.join(path, featurename), 'w') as hf:
                    hf.create_dataset("features",
                                          shape=self.features[sensor][scene_id].shape,
                                          data=self.features[sensor][scene_id],
                                          compression='gzip',
                                          compression_opts=9)

        sdfname = scene_id + '.tsdf_filtered.hf5'
        with h5py.File(os.path.join(path, sdfname), 'w') as hf:
            hf.create_dataset("TSDF_filtered",
                                  shape=self.filtered[scene_id].volume.shape,
                                  data=self.filtered[scene_id].volume)

        if len(self.sensors) > 1 and self.test_mode:
            sensor_weighting_name = scene_id + '.sensor_weighting.hf5'
            with h5py.File(os.path.join(path, sensor_weighting_name), 'w') as hf:
                hf.create_dataset("sensor_weighting",
                                      shape=self.sensor_weighting[scene_id].volume.shape,
                                      data=self.sensor_weighting[scene_id].volume)


    def evaluate(self, mode='train', workspace=None):

        eval_results = {}
        eval_results_scene_save = {}
        for sensor in self.sensors:
            eval_results[sensor] = {}
            eval_results_scene_save[sensor] = {}

        eval_results_filt = {}
        eval_results_scene_save_filt = {}
        if workspace is not None:
            workspace.log('-------------------------------------------------------', 
                mode)
        for scene_id in self.scenes_gt.keys():
            if workspace is None:
                print('Evaluating ', scene_id, '...')
            else:
                workspace.log('Evaluating {} ...'.format(scene_id),
                              mode)
            est = {}
            mask = {}
            for sensor in self.sensors:
                est[sensor] = self.tsdf[sensor][scene_id].volume
                mask[sensor] = self.feature_weights[sensor][scene_id] > 0

            est_filt = self.filtered[scene_id].volume
            gt = self.scenes_gt[scene_id].volume
            mask_filt = np.zeros_like(gt)
            for sensor in self.sensors:
                mask_filt = np.logical_or(mask_filt, mask[sensor])

            if self.erosion:
                # if self.translation_kernel == 3:
                # erode indices mask once
                
                mask_filt = ndimage.binary_erosion(mask_filt, structure=np.ones((3,3,3)), iterations=1)
                for sensor in self.sensors:
                    mask[sensor] = ndimage.binary_erosion(mask[sensor], structure=np.ones((3,3,3)), iterations=1)

                # else:
                #     # erode indices mask twice
                #     mask_filt = ndimage.binary_erosion(mask_filt, structure=np.ones((3,3,3)), iterations=2)
                #     mask_tof = ndimage.binary_erosion(mask_tof, structure=np.ones((3,3,3)), iterations=2)
                #     mask_stereo = ndimage.binary_erosion(mask_stereo, structure=np.ones((3,3,3)), iterations=2)
            eval_results_scene = dict()
            for sensor in self.sensors:
                eval_results_scene[sensor] = evaluation(est[sensor], gt, mask[sensor])
            eval_results_scene_filt = evaluation(est_filt, gt, mask_filt)

            del est, gt, mask, est_filt, mask_filt

            for sensor in self.sensors:
                eval_results_scene_save[sensor][scene_id] = eval_results_scene[sensor]
            eval_results_scene_save_filt[scene_id] = eval_results_scene_filt

            for key in eval_results_scene_filt.keys():
                if workspace is None:
                    for sensor in self.sensors:
                        print(sensor, ' ', key, eval_results_scene[sensor][key])
                    print('filtered ', key, eval_results_scene_filt[key])
                else:
                    for sensor in self.sensors:
                        workspace.log('{} {}'.format(key, eval_results_scene[sensor][key]),
                                    mode)
                    workspace.log('{} {}'.format(key, eval_results_scene_filt[key]),
                                  mode)

                if not eval_results_filt.get(key): # iou, mad, mse, acc as keys
                    for sensor in self.sensors:
                        eval_results[sensor][key] = eval_results_scene[sensor][key]
                    eval_results_filt[key] = eval_results_scene_filt[key]
                else:
                    for sensor in self.sensors:
                        eval_results[sensor][key] += eval_results_scene[sensor][key]
                    eval_results_filt[key] += eval_results_scene_filt[key]

        # normalizing metrics
        for key in eval_results_filt.keys():
            for sensor in self.sensors:
                eval_results[sensor][key] /= len(self.scenes_gt.keys())
            eval_results_filt[key] /= len(self.scenes_gt.keys())

        if mode == 'test':
            return eval_results, eval_results_filt, eval_results_scene_save, eval_results_scene_save_filt
        else:
            return eval_results, eval_results_filt

    def reset(self, scene_id=None):
        if scene_id:
            feature_shape = (self.scenes_gt[scene_id].volume.shape[0], self.scenes_gt[scene_id].volume.shape[1], self.scenes_gt[scene_id].volume.shape[2], self.n_features)
            for sensor in self.sensors:
                self.tsdf[sensor][scene_id].volume = self.initial_value * np.ones(self.scenes_gt[scene_id].volume.shape, dtype=np.float16)
                self.fusion_weights[sensor][scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape, dtype=np.float16)
                # if self.w_features:# TODO: adapt to when not using features
                self.features[sensor][scene_id] = np.zeros(feature_shape, dtype=np.float16)
                self.feature_weights[sensor][scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape, dtype=np.float16)
            self.filtered[scene_id].volume = self.initial_value * np.ones(self.scenes_gt[scene_id].volume.shape, dtype=np.float16)
        else:
            for scene_id in self.scenes_gt.keys():
                feature_shape = (self.scenes_gt[scene_id].volume.shape[0], self.scenes_gt[scene_id].volume.shape[1], self.scenes_gt[scene_id].volume.shape[2], self.n_features)
                for sensor in self.sensors:
                    self.tsdf[sensor][scene_id].volume = self.initial_value * np.ones(self.scenes_gt[scene_id].volume.shape, dtype=np.float16)
                    self.fusion_weights[sensor][scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape, dtype=np.float16)
                    # if self.w_features: # TODO: adapt to when not using features
                    self.features[sensor][scene_id] = np.zeros(feature_shape, dtype=np.float16)
                    self.feature_weights[sensor][scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape, dtype=np.float16)
                self.filtered[scene_id].volume = self.initial_value * np.ones(self.scenes_gt[scene_id].volume.shape, dtype=np.float16)
