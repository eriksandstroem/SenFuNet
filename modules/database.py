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
        self.n_features = config.n_features

        self.scenes_gt = {}
        self.scenes_est_tof = {}
        self.scenes_est_stereo = {}
        self.fusion_weights_tof = {}
        self.fusion_weights_stereo = {}
        self.features_tof = {}
        self.features_stereo = {}
        self.feature_weights_tof = {}
        self.feature_weights_stereo = {}
        self.filtered = {} # grid to store the final sdf prediction

        for s in dataset.scenes:
            grid = dataset.get_grid(s, truncation=self.trunc_value)
            self.scenes_gt[s] = grid

            init_volume1 = self.initial_value * np.ones_like(grid.volume, dtype=np.float16)
            init_volume2 = self.initial_value * np.ones_like(grid.volume, dtype=np.float16) # this was important for the integrator!
            init_volume3 = self.initial_value * np.ones_like(grid.volume, dtype=np.float16)

            self.scenes_est_tof[s] = Voxelgrid(self.scenes_gt[s].resolution)
            self.scenes_est_tof[s].from_array(init_volume1, self.scenes_gt[s].bbox)
            self.fusion_weights_tof[s] = np.zeros(self.scenes_gt[s].volume.shape, dtype=np.float16)
            self.scenes_est_stereo[s] = Voxelgrid(self.scenes_gt[s].resolution)
            self.scenes_est_stereo[s].from_array(init_volume2, self.scenes_gt[s].bbox)
            self.fusion_weights_stereo[s] = np.zeros(self.scenes_gt[s].volume.shape, dtype=np.float16)
            del init_volume1, init_volume2
            fusion_feature_shape = (self.scenes_gt[s].volume.shape[0], self.scenes_gt[s].volume.shape[1], self.scenes_gt[s].volume.shape[2], self.n_features)
            self.features_tof[s] = np.zeros(fusion_feature_shape, dtype=np.float16)
            self.features_stereo[s] = np.zeros(fusion_feature_shape, dtype=np.float16)
            self.feature_weights_tof[s] = np.zeros(self.scenes_gt[s].volume.shape, dtype=np.float16)
            self.feature_weights_stereo[s] = np.zeros(self.scenes_gt[s].volume.shape, dtype=np.float16)
            self.filtered[s] = Voxelgrid(self.scenes_gt[s].resolution)
            self.filtered[s].from_array(init_volume3, self.scenes_gt[s].bbox)
            del init_volume3
        
        # self.reset()

    def __getitem__(self, item):

        sample = dict()

        sample['gt'] = self.scenes_gt[item].volume
        sample['current_tof'] = self.scenes_est_tof[item].volume
        sample['current_stereo'] = self.scenes_est_stereo[item].volume
        sample['origin'] = self.scenes_gt[item].origin
        sample['resolution'] = self.scenes_gt[item].resolution
        sample['weights_tof'] = self.fusion_weights_tof[item]
        sample['weights_stereo'] = self.fusion_weights_stereo[item]
        sample['features_tof'] = self.features_tof[item]
        sample['features_stereo'] = self.features_stereo[item]
        sample['feature_weights_tof'] = self.feature_weights_tof[item]
        sample['feature_weights_stereo'] = self.feature_weights_stereo[item]
        sample['filtered'] = self.filtered[item]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.scenes_gt)

    def filter(self, value=2.):

        for key in self.scenes_est.keys():

            weights = self.fusion_weights[key]
            self.scenes_est_tof[key].volume[weights < value] = self.initial_value
            self.fusion_weights_tof[key][weights < value] = 0
            self.scenes_est_stereo[key].volume[weights < value] = self.initial_value
            self.fusion_weights_stereo[key][weights < value] = 0

    def save_to_workspace(self, workspace, is_best, is_best_tof, is_best_stereo, save_mode='ply'):

        for key in self.scenes_gt.keys():

            tsdf_volume_tof = self.scenes_est_tof[key].volume
            weight_volume_tof = self.feature_weights_tof[key]

            tsdf_volume_stereo = self.scenes_est_stereo[key].volume
            weight_volume_stereo = self.feature_weights_stereo[key]

            filtered_tsdf_volume = self.filtered[key].volume
            print('filtered_tsdf_volume min', filtered_tsdf_volume.min())
            print('filtered_tsdf_volume max', filtered_tsdf_volume.max())

            if is_best:
                mode_filt = 'best'
            else:
                mode_filt = 'latest'

            if is_best_tof:
                mode_tof = 'best'
            else:
                mode_tof = 'latest'

            if is_best_stereo:
                mode_stereo = 'best'
            else:
                mode_stereo = 'latest'

            if save_mode == 'tsdf':
                tsdf_file_tof = key.replace('/', '.') + '.tof_tsdf_' + mode_tof + '.hf5'
                weight_file_tof = key.replace('/', '.') + '.tof_weights_' + mode_tof + '.hf5'

                tsdf_file_stereo = key.replace('/', '.') + '.stereo_tsdf_' + mode_stereo + '.hf5'
                weight_file_stereo = key.replace('/', '.') + '.stereo_weights_' + mode_stereo + '.hf5'

                workspace.save_tsdf_data(tsdf_file_tof, tsdf_volume_tof)
                workspace.save_weights_data(weight_file_tof, weight_volume_tof)
                workspace.save_tsdf_data(tsdf_file_stereo, tsdf_volume_stereo)
                workspace.save_weights_data(weight_file_stereo, weight_volume_stereo)

                filtered_tsdf_file = key.replace('/', '.') + '.filtered_tsdf_' + mode_filt + '.hf5'
                workspace.save_tsdf_data(filtered_tsdf_file, filtered_tsdf_volume)
            elif save_mode == 'ply':
                ply_file_tof = key.replace('/', '.') + '_' + mode_tof + '_tof.ply'
                ply_file_stereo = key.replace('/', '.') + '_' + mode_stereo + '_stereo.ply'
                filtered_ply_file = key.replace('/', '.') + '_filtered_' + mode_filt + '.ply'
                if tsdf_volume_tof.min() < 0:
                    workspace.save_ply_data(ply_file_tof, tsdf_volume_tof)
                if tsdf_volume_stereo.min() < 0:
                    workspace.save_ply_data(ply_file_stereo, tsdf_volume_stereo)
                if filtered_tsdf_volume.min() < 0 and filtered_tsdf_volume.max() > 0:
                    workspace.save_ply_data(filtered_ply_file, filtered_tsdf_volume)

    def save(self, path, save_mode='ply', scene_id=None):


        if save_mode =='ply':
            # intermediate geometry
            ply_file = scene_id.replace('/', '.') + '_tof.ply'
            filename = os.path.join(path, ply_file)
            voxel_size = 0.01
            vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(self.scenes_est_tof[scene_id].volume, 
                    level=0, spacing=(voxel_size, voxel_size, voxel_size))
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
            mesh.export(filename)

            ply_file = scene_id.replace('/', '.') + '_stereo.ply'
            filename = os.path.join(path, ply_file)
            vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(self.scenes_est_stereo[scene_id].volume, 
                    level=0, spacing=(voxel_size, voxel_size, voxel_size))
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
            mesh.export(filename)

            # filtered geometry
            ply_file = scene_id.replace('/', '.') + '_filtered.ply'
            filename = os.path.join(path, ply_file)
            vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(self.filtered[scene_id].volume, 
                    level=0, spacing=(voxel_size, voxel_size, voxel_size))
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
            mesh.export(filename)

        elif save_mode == 'test':
            filename = '{}_tof.tsdf.hf5'.format(scene_id.replace('/', '.'))
            weightname = '{}_tof.weights.hf5'.format(scene_id.replace('/', '.'))

            with h5py.File(os.path.join(path, filename), 'w') as hf:
                hf.create_dataset("TSDF",
                                      shape=self.scenes_est_tof[scene_id].volume.shape,
                                      data=self.scenes_est_tof[scene_id].volume,
                                      compression='gzip',
                                      compression_opts=9)
            with h5py.File(os.path.join(path, weightname), 'w') as hf:
                hf.create_dataset("weights",
                                      shape=self.feature_weights_tof[scene_id].shape,
                                      data=self.feature_weights_tof[scene_id],
                                      compression='gzip',
                                      compression_opts=9)

            filename = '{}_stereo.tsdf.hf5'.format(scene_id.replace('/', '.'))
            weightname = '{}_stereo.weights.hf5'.format(scene_id.replace('/', '.'))

            with h5py.File(os.path.join(path, filename), 'w') as hf:
                hf.create_dataset("TSDF",
                                      shape=self.scenes_est_stereo[scene_id].volume.shape,
                                      data=self.scenes_est_stereo[scene_id].volume,
                                      compression='gzip',
                                      compression_opts=9)
            with h5py.File(os.path.join(path, weightname), 'w') as hf:
                hf.create_dataset("weights",
                                      shape=self.feature_weights_stereo[scene_id].shape,
                                      data=self.feature_weights_stereo[scene_id],
                                      compression='gzip',
                                      compression_opts=9)

            sdfname = '{}.tsdf_filtered.hf5'.format(scene_id.replace('/', '.'))
            with h5py.File(os.path.join(path, sdfname), 'w') as hf:
                hf.create_dataset("TSDF_filtered",
                                      shape=self.filtered[scene_id].volume.shape,
                                      data=self.filtered[scene_id].volume)


    def evaluate(self, mode='train', workspace=None):

        eval_results_tof = {}
        eval_results_scene_save_tof = {}
        eval_results_stereo = {}
        eval_results_scene_save_stereo = {}
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

            est_tof = self.scenes_est_tof[scene_id].volume
            est_stereo = self.scenes_est_stereo[scene_id].volume
            est_filt = self.filtered[scene_id].volume
            gt = self.scenes_gt[scene_id].volume

            mask_tof = (self.feature_weights_tof[scene_id] > 0) 
            mask_stereo = (self.feature_weights_stereo[scene_id] > 0) 
            mask_filt = np.logical_or(mask_tof, mask_stereo)

            if self.erosion:
                # if self.translation_kernel == 3:
                # erode indices mask once
                mask_filt = ndimage.binary_erosion(mask_filt, structure=np.ones((3,3,3)), iterations=1)
                mask_tof = ndimage.binary_erosion(mask_tof, structure=np.ones((3,3,3)), iterations=1)
                mask_stereo = ndimage.binary_erosion(mask_stereo, structure=np.ones((3,3,3)), iterations=1)
                # else:
                #     # erode indices mask twice
                #     mask_filt = ndimage.binary_erosion(mask_filt, structure=np.ones((3,3,3)), iterations=2)
                #     mask_tof = ndimage.binary_erosion(mask_tof, structure=np.ones((3,3,3)), iterations=2)
                #     mask_stereo = ndimage.binary_erosion(mask_stereo, structure=np.ones((3,3,3)), iterations=2)

            eval_results_scene_tof = evaluation(est_tof, gt, mask_tof)
            eval_results_scene_stereo = evaluation(est_stereo, gt, mask_stereo)
            eval_results_scene_filt = evaluation(est_filt, gt, mask_filt)

            del est_tof, est_stereo, gt, mask_tof, mask_stereo, est_filt, mask_filt

            eval_results_scene_save_tof[scene_id] = eval_results_scene_tof
            eval_results_scene_save_stereo[scene_id] = eval_results_scene_stereo
            eval_results_scene_save_filt[scene_id] = eval_results_scene_filt

            for key in eval_results_scene_tof.keys():

                if workspace is None:
                    print('tof ', key, eval_results_scene_tof[key])
                    print('stereo ', key, eval_results_scene_stereo[key])
                    print('filtered ', key, eval_results_scene_filt[key])
                else:
                    workspace.log('{} {}'.format(key, eval_results_scene_tof[key]),
                                  mode)
                    workspace.log('{} {}'.format(key, eval_results_scene_stereo[key]),
                                  mode)
                    workspace.log('{} {}'.format(key, eval_results_scene_filt[key]),
                                  mode)


                if not eval_results_tof.get(key): # iou, mad, mse, acc as keys
                    eval_results_tof[key] = eval_results_scene_tof[key]
                    eval_results_stereo[key] = eval_results_scene_stereo[key]
                    eval_results_filt[key] = eval_results_scene_filt[key]
                else:
                    eval_results_tof[key] += eval_results_scene_tof[key]
                    eval_results_stereo[key] += eval_results_scene_stereo[key]
                    eval_results_filt[key] += eval_results_scene_filt[key]

        # normalizing metrics
        for key in eval_results_tof.keys():
            eval_results_tof[key] /= len(self.scenes_gt.keys())
            eval_results_stereo[key] /= len(self.scenes_gt.keys())
            eval_results_filt[key] /= len(self.scenes_gt.keys())

        if mode == 'test':
            return eval_results_tof, eval_results_stereo, eval_results_filt, eval_results_scene_save_tof, eval_results_scene_save_stereo, eval_results_scene_save_filt
        else:
            return eval_results_tof, eval_results_stereo, eval_results_filt

    def reset(self, scene_id=None):
        if scene_id:
            feature_shape = (self.scenes_gt[scene_id].volume.shape[0], self.scenes_gt[scene_id].volume.shape[1], self.scenes_gt[scene_id].volume.shape[2], self.n_features)
            self.scenes_est_tof[scene_id].volume = self.initial_value * np.ones(self.scenes_est_tof[scene_id].volume.shape, dtype=np.float16)
            self.fusion_weights_tof[scene_id] = np.zeros(self.scenes_est_tof[scene_id].volume.shape, dtype=np.float16)
            self.scenes_est_stereo[scene_id].volume = self.initial_value * np.ones(self.scenes_est_stereo[scene_id].volume.shape, dtype=np.float16)
            self.fusion_weights_stereo[scene_id] = np.zeros(self.scenes_est_stereo[scene_id].volume.shape, dtype=np.float16)
            self.features_tof[scene_id] = np.zeros(feature_shape, dtype=np.float16)
            self.features_stereo[scene_id] = np.zeros(feature_shape, dtype=np.float16)
            self.feature_weights_tof[scene_id] = np.zeros(self.scenes_est_stereo[scene_id].volume.shape, dtype=np.float16)
            self.feature_weights_stereo[scene_id] = np.zeros(self.scenes_est_stereo[scene_id].volume.shape, dtype=np.float16)
            self.filtered[scene_id].volume = self.initial_value * np.ones(self.scenes_est_tof[scene_id].volume.shape, dtype=np.float16)
        else:
            for scene_id in self.scenes_gt.keys():
                feature_shape = (self.scenes_gt[scene_id].volume.shape[0], self.scenes_gt[scene_id].volume.shape[1], self.scenes_gt[scene_id].volume.shape[2], self.n_features)
                self.scenes_est_tof[scene_id].volume = self.initial_value * np.ones(self.scenes_est_tof[scene_id].volume.shape, dtype=np.float16)
                self.fusion_weights_tof[scene_id] = np.zeros(self.scenes_est_tof[scene_id].volume.shape, dtype=np.float16)
                self.scenes_est_stereo[scene_id].volume = self.initial_value * np.ones(self.scenes_est_stereo[scene_id].volume.shape, dtype=np.float16)
                self.fusion_weights_stereo[scene_id] = np.zeros(self.scenes_est_stereo[scene_id].volume.shape, dtype=np.float16)
                self.features_tof[scene_id] = np.zeros(feature_shape, dtype=np.float16)
                self.features_stereo[scene_id] = np.zeros(feature_shape, dtype=np.float16)
                self.feature_weights_tof[scene_id] = np.zeros(self.scenes_est_stereo[scene_id].volume.shape, dtype=np.float16)
                self.feature_weights_stereo[scene_id] = np.zeros(self.scenes_est_stereo[scene_id].volume.shape, dtype=np.float16)
                self.filtered[scene_id].volume = self.initial_value * np.ones(self.scenes_est_tof[scene_id].volume.shape, dtype=np.float16)
