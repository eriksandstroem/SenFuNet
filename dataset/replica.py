import os
import glob

import sys
import random
import numpy as np

from skimage import io, transform
from skimage.color import rgb2gray
from skimage import filters
from skimage.morphology import binary_erosion
from torch.utils.data import Dataset

from graphics import Voxelgrid
import h5py


class Replica(Dataset):

    def __init__(self, config_data):
        self.root_dir = os.getenv(config_data.root_dir) # when training on local scratch
        
        # os.getenv returns none when the input does not exist. When 
        # it returns none, we want to train on the work folder
        if not self.root_dir:
            self.root_dir  = config_data.root_dir

        self.sampling_density_stereo = config_data.sampling_density_stereo
        self.sampling_density_tof = config_data.sampling_density_tof

        self.resolution_stereo = (config_data.resy_stereo, config_data.resx_stereo)
        # self.resx_stereo = self.resolution_stereo[0]/512
        # self.resy_stereo = self.resolution_stereo[1]/512

        self.resolution_tof = (config_data.resy_tof, config_data.resx_tof)
        # self.resx_tof = self.resolution_tof[0]/512
        # self.resy_tof = self.resolution_tof[1]/512

        self.mask_stereo_width = config_data.mask_stereo_width
        self.mask_stereo_height = config_data.mask_stereo_height
        self.mask_tof_width = config_data.mask_tof_width
        self.mask_tof_height = config_data.mask_tof_height

        self.min_depth_stereo = config_data.min_depth_stereo
        self.max_depth_stereo = config_data.max_depth_stereo
        self.min_depth_tof = config_data.min_depth_tof
        self.max_depth_tof = config_data.max_depth_tof

        self.transform = config_data.transform
        self.pad = config_data.pad

        self.scene_list = config_data.scene_list
        self.input = config_data.input
        self.target = config_data.target
        self.mode = config_data.mode
        self.intensity_gradient = config_data.intensity_grad
        self.truncation_strategy = config_data.truncation_strategy
        self.fusion_strategy = config_data.fusion_strategy

        self._scenes = []

        if config_data.data_load_strategy == 'hybrid':
            self.nbr_load_scenes = config_data.load_scenes_at_once
            self._get_scene_load_order(self.nbr_load_scenes)
        else:
            self.scenedir = None

        self._load_color()
        self._load_depth_gt()
        self._load_cameras()
        if self.input == 'multidepth':
            self._load_depths()
        else:
            self._load_depth()

        

    def _get_scene_load_order(self, nbr_load_scenes):
        # This loading strategy 'hybrid' will always only load from at most 1 trajectory 
        # of a scene at a time. This is contrary to the 'max_depth_diversity' strategy 
        # which loads all trajectories from all scenes at the same time.

        # create list of training scenes
        scenes_list = list()
        with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:
            for line in file:
                if line.split(' ')[0].split('/')[0] not in scenes_list:
                    scenes_list.append(line.split(' ')[0].split('/')[0])

        self._scenes = scenes_list

        # make sure nbr_load_scenes <= len(trajectory_list)
        if nbr_load_scenes > len(scenes_list):
            raise ValueError('nbr_load_scenes variable is lower than the number of scenes')
        # create nbr_load_scenes empty lists
        listdir = dict()
        for i in range(nbr_load_scenes):
            listdir[i] = list()

        # sample from trajectory_list and fill listdir
        while scenes_list:
            if nbr_load_scenes > len(scenes_list):
                scene_indices = random.sample(range(0, len(scenes_list)), len(scenes_list))
            else:
                scene_indices = random.sample(range(0, len(scenes_list)), nbr_load_scenes)

            for key, scene_idx in enumerate(scene_indices):
                listdir[key].append(scenes_list[scene_idx])

            scenes_list = [val for idx, val in enumerate(scenes_list) if idx not in scene_indices]

        # add the trajectories to the listdir
        for key in listdir:
            # create new list to replace the old one
            new_list_element = list()
            for scene in listdir[key]:
                for i in range(3):
                    new_list_element.append(scene + '/' + str(i + 1))

            # shuffle the list before replacing the old one
            random.shuffle(new_list_element)
            listdir[key] = new_list_element

        self.scenedir = listdir


        print('Integration scene order: ', listdir)



    def _load_depth(self): # loads the paths of the noisy depth images to a list
        self.depth_images = []
        if self.scenedir is not None:
            modality = ['left_depth_gt', 'left_depth_noise_5.0', 'left_bts_depth', 'left_psmnet_depth']
            # create the full lists for each key in scenedir
            tmp_dict = dict()
            for key in self.scenedir:
                tmp_list = list()
                for trajectory in self.scenedir[key]:
                    if self.input == 'tof_depth':
                        files = glob.glob(os.path.join(self.root_dir, trajectory, modality[1], '*.png'))
                        # sort the list accoring to the indices. Here is where I could also invert the trajectory
                        files = sorted(files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
                        for file in files:
                            tmp_list.append(file)
                    elif self.input == 'mono_depth':
                        files = glob.glob(os.path.join(self.root_dir, trajectory, modality[2], '*.png'))
                        files = sorted(files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
                        for file in files:
                            tmp_list.append(file)
                    elif self.input == 'stereo_depth':
                        files = glob.glob(os.path.join(self.root_dir, trajectory, modality[3], '*.png'))
                        files = sorted(files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
                        for file in files:
                            tmp_list.append(file)
                    elif self.input == 'depth_gt':
                        files = glob.glob(os.path.join(self.root_dir, trajectory, modality[0], '*.png'))
                        files = sorted(files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
                        for file in files:
                            tmp_list.append(file)
                # replace the short list with the long list including all file paths
                tmp_dict[key] = tmp_list
            
            # fuse the lists into one
            # find key with longest list
            max_key = max(tmp_dict, key = lambda x: len(tmp_dict[x]))
            idx = 0
            while idx < len(tmp_dict[max_key]):
                for key in tmp_dict:
                    if idx < len(tmp_dict[key]):
                        self.depth_images.append(tmp_dict[key][idx])
                idx += 1

            del tmp_dict

        else:
            # reading files from list
            with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:
                for line in file:
                    line = line.split(' ')
                    if self.input == 'tof_depth':
                        files = glob.glob(os.path.join(self.root_dir, line[1], '*.png'))
                        for file in files:
                            self.depth_images.append(file)
                    elif self.input == 'mono_depth':
                        files = glob.glob(os.path.join(self.root_dir, line[2], '*.png'))
                        for file in files:
                            self.depth_images.append(file)
                    elif self.input == 'stereo_depth':
                        files = glob.glob(os.path.join(self.root_dir, line[3], '*.png'))
                        for file in files:
                            self.depth_images.append(file)
                    elif self.input == 'depth_gt':
                        files = glob.glob(os.path.join(self.root_dir, line[0], '*.png'))
                        for file in files:
                            self.depth_images.append(file)


            # perhaps it will be important to order the frames for testing and training the fusion network.
            self.depth_images = sorted(self.depth_images, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
        if self.mode == 'val':
            self.depth_images = self.depth_images[::4]


    def _load_depths(self): # loads the paths of the noisy depth images to a list

        if self.scenedir is not None:
            raise NotImplementedError

        else:
            # reading files from list
            self.depth_images_tof = []
            self.depth_images_mono = []
            self.depth_images_stereo = []
            with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:
                for line in file:
                    line = line.split(' ')
                    files = glob.glob(os.path.join(self.root_dir, line[1], '*.png'))
                    for file in files:
                        self.depth_images_tof.append(file)
                    files = glob.glob(os.path.join(self.root_dir, line[2], '*.png'))
                    for file in files:
                        self.depth_images_mono.append(file)
                    files = glob.glob(os.path.join(self.root_dir, line[3], '*.png'))
                    for file in files:
                        self.depth_images_stereo.append(file)

        # perhaps it will be important to order the frames for testing and training the fusion network.
        self.depth_images_tof = sorted(self.depth_images_tof, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
        self.depth_images_mono = sorted(self.depth_images_mono, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
        self.depth_images_stereo = sorted(self.depth_images_stereo, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
        if self.mode == 'val':
            self.depth_images_tof = self.depth_images_tof[::4]
            self.depth_images_mono = self.depth_images_mono[::4]
            self.depth_images_stereo = self.depth_images_stereo[::4]

    def _load_depth_gt(self): # loads the paths of the ground truth depth images to a list
        self.depth_images_gt = []
        if self.scenedir is not None:
            modality = 'left_depth_gt'
            # create the full lists for each key in scenedir
            tmp_dict = dict()
            for key in self.scenedir:
                tmp_list = list()
                for trajectory in self.scenedir[key]:
                    files = glob.glob(os.path.join(self.root_dir, trajectory, modality, '*.png'))
                    files = sorted(files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
                    for file in files:
                        tmp_list.append(file)
                # replace the short list with the long list including all file paths
                tmp_dict[key] = tmp_list
            
            # fuse the lists into one
            # find key with longest list
            max_key = max(tmp_dict, key = lambda x: len(tmp_dict[x]))
            idx = 0
            while idx < len(tmp_dict[max_key]):
                for key in tmp_dict:
                    if idx < len(tmp_dict[key]):
                        self.depth_images_gt.append(tmp_dict[key][idx])
                idx += 1

            del tmp_dict

        else:
            # reading files from list
            with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:
                for line in file:
                    line = line.split(' ')
                    if line[0].split('/')[0] not in self._scenes:
                        self._scenes.append(line[0].split('/')[0])
                    files = glob.glob(os.path.join(self.root_dir, line[0], '*.png'))
                    for file in files:
                        self.depth_images_gt.append(file)

            self.depth_images_gt = sorted(self.depth_images_gt, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))

        if self.mode == 'val':
            self.depth_images_gt = self.depth_images_gt[::4]


    def _load_color(self):
        self.color_images = []
        if self.scenedir is not None:
            modality = 'left_rgb'
            # create the full lists for each key in scenedir
            tmp_dict = dict()
            for key in self.scenedir:
                tmp_list = list()
                for trajectory in self.scenedir[key]:
                    files = glob.glob(os.path.join(self.root_dir, trajectory, modality, '*.png'))
                    files = sorted(files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
                    for file in files:
                        tmp_list.append(file)
                # replace the short list with the long list including all file paths
                tmp_dict[key] = tmp_list
            
            # fuse the lists into one
            # find key with longest list
            max_key = max(tmp_dict, key = lambda x: len(tmp_dict[x]))
            idx = 0
            while idx < len(tmp_dict[max_key]):
                for key in tmp_dict:
                    if idx < len(tmp_dict[key]):
                        self.color_images.append(tmp_dict[key][idx])
                idx += 1

            del tmp_dict

        else:
            # reading files from list
            with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:
                for line in file:
                    line = line.split(' ')
                    files = glob.glob(os.path.join(self.root_dir, line[-2], '*.png'))
                    for file in files:
                        self.color_images.append(file)

            self.color_images = sorted(self.color_images, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))

        if self.mode == 'val':
            self.color_images = self.color_images[::4]

    def _load_cameras(self):
        self.cameras = []
        if self.scenedir is not None:
            modality = 'left_camera_matrix'

            # create the full lists for each key in scenedir
            tmp_dict = dict()
            for key in self.scenedir:
                tmp_list = list()
                for trajectory in self.scenedir[key]:
                    files = glob.glob(os.path.join(self.root_dir, trajectory, modality, '*.txt'))
                    files = sorted(files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
                    for file in files:
                        tmp_list.append(file)
                # replace the short list with the long list including all file paths
                tmp_dict[key] = tmp_list
            
            # fuse the lists into one
            # find key with longest list
            max_key = max(tmp_dict, key = lambda x: len(tmp_dict[x]))
            idx = 0
            while idx < len(tmp_dict[max_key]):
                for key in tmp_dict:
                    if idx < len(tmp_dict[key]):
                        self.cameras.append(tmp_dict[key][idx])
                idx += 1

            del tmp_dict

        else:

            with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:

                for line in file:
                    line = line.split(' ')
                    files = glob.glob(os.path.join(self.root_dir, line[-1][:-1], '*.txt'))
                    for file in files:
                        self.cameras.append(file)

            self.cameras = sorted(self.cameras, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))

        if self.mode == 'val':
            self.cameras = self.cameras[::4]

    @property
    def scenes(self):
        return self._scenes

    def __len__(self):
        return len(self.depth_images_gt)

    def __getitem__(self, item):

        sample = dict()
        sample['item_id'] = item 

        # load rgb image
        file = self.color_images[item]

        pathsplit = file.split('/')
        scene = pathsplit[-4]
        trajectory = pathsplit[-3]
        frame = os.path.splitext(pathsplit[-1])[0]
        frame_id = '{}/{}/{}'.format(scene, trajectory, frame)

        image = io.imread(file)

        step_x = image.shape[0] / self.resolution_stereo[0]
        step_y = image.shape[1] / self.resolution_stereo[0]

        index_y = [int(step_y * i) for i in
                   range(0, int(image.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in
                   range(0, int(image.shape[0] / step_x))]

        image = image[:, index_y]
        image = image[index_x, :]
        sample['image'] = np.asarray(image).astype(np.float32)/255


        # if self.intensity_gradient: 
        intensity = rgb2gray(image) # seems to be in range 0 - 1 
        sample['intensity'] = np.asarray(intensity).astype(np.float32)
        grad_y = filters.sobel_h(intensity)
        grad_x = filters.sobel_v(intensity)
        grad = (grad_x**2 + grad_y**2)**(1/2)
        sample['gradient'] = np.asarray(grad).astype(np.float32)

        if self.input == 'multidepth':
            # load noisy depth maps
            file_tof = self.depth_images_tof[item]
            # file_mono = self.depth_images_mono[item]
            file_stereo = self.depth_images_stereo[item]

            depth_tof = io.imread(file_tof).astype(np.float32)
            # depth_mono = io.imread(file_mono).astype(np.float32)
            depth_stereo = io.imread(file_stereo).astype(np.float32)

            step_x = depth_tof.shape[0] / self.resolution_tof[0]
            step_y = depth_tof.shape[1] / self.resolution_tof[1]

            index_y = [int(step_y * i) for i in
                       range(0, int(depth_tof.shape[1] / step_y))]
            index_x = [int(step_x * i) for i in
                       range(0, int(depth_tof.shape[0] / step_x))]

            depth_tof = depth_tof[:, index_y]
            depth_tof = depth_tof[index_x, :]

            step_x = depth_stereo.shape[0] / self.resolution_stereo[0]
            step_y = depth_stereo.shape[1] / self.resolution_stereo[1]

            index_y = [int(step_y * i) for i in
                       range(0, int(depth_stereo.shape[1] / step_y))]
            index_x = [int(step_x * i) for i in
                       range(0, int(depth_stereo.shape[0] / step_x))]

            depth_stereo = depth_stereo[:, index_y]
            depth_stereo = depth_stereo[index_x, :]

            depth_tof /= 1000.
            # depth_mono /= 1000
            depth_stereo /= 1000.

            # define mask
            if self.fusion_strategy == 'routingNet':
                mask = np.logical_or((depth_tof > self.min_depth), (depth_stereo > self.min_depth))
                mask = np.logical_and(mask, np.logical_or(depth_tof < self.max_depth, depth_stereo < self.max_depth))
                # remove strong artifacts coming from pixels close to missing pixels
                # the routing network computes bad depths for these pixels that are
                # not missing in the original image due to the convolutions over
                # zero-valued pixels.
                # for i in range(10):
                #     mask = binary_erosion(mask)

                # do not integrate depth values close to the image boundary
                # this is relevant for the stereo modality. 
                mask[0:10, :] = 0
                mask[-10:-1, :] = 0
                mask[:, 0:10] = 0
                mask[:, -10:-1] = 0

                sample['mask'] = mask

            elif self.fusion_strategy == 'fusionNet' or self.fusion_strategy == 'two_fusionNet' or self.fusion_strategy == 'fusionNet_conditioned':
                mask = (depth_stereo > self.min_depth_stereo)
                mask = np.logical_and(mask, depth_stereo < self.max_depth_stereo)

                # do not integrate depth values close to the image boundary
                # this is relevant for the stereo modality. 
                mask[0:self.mask_stereo_height, :] = 0
                mask[-self.mask_stereo_height:-1, :] = 0
                mask[:, 0:self.mask_stereo_width] = 0
                mask[:, -self.mask_stereo_width:-1] = 0
                sample['stereo_mask'] = mask
                mask = (depth_tof > self.min_depth_tof)
                mask = np.logical_and(mask, depth_tof < self.max_depth_tof)

                # do not integrate depth values close to the image boundary
                # this is relevant for the stereo modality. 
                mask[0:self.mask_tof_height, :] = 0
                mask[-self.mask_tof_height:-1, :] = 0
                mask[:, 0:self.mask_tof_width] = 0
                mask[:, -self.mask_tof_width:-1] = 0
                sample['tof_mask'] = mask

            sample['tof_depth'] = np.asarray(depth_tof)
            # sample['mono_depth'] = np.asarray(depth_mono)
            sample['stereo_depth'] = np.asarray(depth_stereo)

        else:
            # load noisy depth map
            file = self.depth_images[item]
            # print(file)
            depth = io.imread(file).astype(np.float32)

            step_x = depth.shape[0] / self.resolution[0]
            step_y = depth.shape[1] / self.resolution[1]

            index_y = [int(step_y * i) for i in
                       range(0, int(depth.shape[1] / step_y))]
            index_x = [int(step_x * i) for i in
                       range(0, int(depth.shape[0] / step_x))]

            depth = depth[:, index_y]
            depth = depth[index_x, :]

            depth /= 1000.

            # define mask
            mask = (depth > self.min_depth)
            mask = np.logical_and(mask, depth < self.max_depth)

            # remove strong artifacts coming from pixels close to missing pixels
            # the routing network computes bad depths for these pixels that are
            # not missing in the original image due to the convolutions over
            # zero-valued pixels.
            # for i in range(10):
            #     mask = binary_erosion(mask)

            # do not integrate depth values close to the image boundary
            # this is relevant for the stereo modality. 
            mask[0:10, :] = 0
            mask[-10:-1, :] = 0
            mask[:, 0:10] = 0
            mask[:, -10:-1] = 0

            sample['mask'] = mask

            sample[self.input] = np.asarray(depth)

        # load ground truth depth map
        file = self.depth_images_gt[item]
        # print(file)
        depth = io.imread(file).astype(np.float32)

        step_x = depth.shape[0] / self.resolution_stereo[0]
        step_y = depth.shape[1] / self.resolution_stereo[0]

        index_y = [int(step_y * i) for i in
                   range(0, int(depth.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in
                   range(0, int(depth.shape[0] / step_x))]

        depth = depth[:, index_y]
        depth = depth[index_x, :]

        depth /= 1000.

        sample[self.target] = np.asarray(depth)

        # load extrinsics
        file = self.cameras[item]
        # print(file)
        extrinsics = np.loadtxt(file)
        extrinsics = np.linalg.inv(extrinsics).astype(np.float32)
        # the fusion code expects that the camera coordinate system is such that z is in the
        # camera viewing direction, y is down and x is to the right. This is achieved by a serie of rotations
        rot_180_around_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).astype(np.float32)
        rot_180_around_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).astype(np.float32)
        rot_90_around_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(np.float32)
        rotation = np.matmul(rot_180_around_z, rot_180_around_y)
        extrinsics =  np.matmul(rotation, extrinsics[0:3, 0:4])
        extrinsics = np.linalg.inv(np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0))
        sample['extrinsics'] = np.matmul(rot_90_around_x, extrinsics[0:3, 0:4])

        hfov = 90.
        f = self.resolution_tof[0]/2.*(1./np.tan(np.deg2rad(hfov)/2)) # I always assume square input images
        shift = self.resolution_tof[0]/2

        # load intrinsics
        intrinsics = np.asarray([[f, 0., shift],
                                 [0., f, shift],
                                 [0., 0., 1.]])

        sample['intrinsics_tof'] = intrinsics

        f = self.resolution_stereo[0]/2.*(1./np.tan(np.deg2rad(hfov)/2)) # I always assume square input images
        shift = self.resolution_stereo[0]/2

        # load intrinsics
        intrinsics = np.asarray([[f, 0., shift],
                                 [0., f, shift],
                                 [0., 0., 1.]])

        sample['intrinsics_stereo'] = intrinsics

        sample['frame_id'] = frame_id

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_grid(self, scene, truncation):
        file = os.path.join(self.root_dir, scene, 'sdf_' + scene + '.hdf')

        # read from hdf file!
        f = h5py.File(file, 'r')
        voxels = np.array(f['sdf']).astype(np.float16)
        if self.truncation_strategy == 'artificial':
            voxels[np.abs(voxels) >= truncation] = truncation
            # Add padding to grid to give more room to fusion net
            voxels = np.pad(voxels, self.pad, 'constant', constant_values=truncation)
        elif self.truncation_strategy == 'standard':
            voxels[voxels > truncation] = truncation
            voxels[voxels < -truncation] = -truncation
            # Add padding to grid to give more room to fusion net
            voxels = np.pad(voxels, self.pad, 'constant', constant_values=-truncation)

        print(scene, voxels.shape)
        bbox = np.zeros((3, 2))
        bbox[:, 0] = f.attrs['bbox'][:, 0] - self.pad*f.attrs['voxel_size']*np.ones((1,1,1))
        bbox[:, 1] = bbox[:, 0] + f.attrs['voxel_size'] * np.array(voxels.shape)

        voxelgrid = Voxelgrid(f.attrs['voxel_size'])
        voxelgrid.from_array(voxels, bbox)

        return voxelgrid



if __name__ == '__main__':


    from tqdm import tqdm

    path_to_utils_module =  '/home/esandstroem/scratch-second/euler_project/src/late_fusion/utils/' #'/cluster/project/cvl/esandstroem/src/late_fusion/utils/'
    sys.path.append(path_to_utils_module) # needed in order to load read_array and associate

    from loading import load_config_from_yaml
    import open3d as o3d

    from easydict import EasyDict
    from tsdf import TSDFHandle
    import skimage
    import trimesh




    # get config
    path_to_config = '/home/esandstroem/scratch-second/euler_project/src/late_fusion_w_featurefusion/configs/fusion/replica_euler.yaml'
    config = load_config_from_yaml(path_to_config)
    # dataset_config = get_data_config(config, mode='train')
    config.DATA.scene_list = config.DATA.test_scene_list
    # config.DATA.transform = dataset_config.transform
    config.DATA.mode = 'test'
    dataset = Replica(config.DATA) #get_data(config.DATA.dataset, config.DATA)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=config.TRAINING.train_batch_size, shuffle=False)

    truncation = 0.1
    pad = 2

    # load gt grid to get volume shape
    gt_path = '/home/esandstroem/scratch-second/euler_work/data/replica/manual/office_1/sdf_office_1.hdf'
    # read from hdf file!
    f = h5py.File(gt_path, 'r')

    gt = np.array(f['sdf']).astype(np.float16)
    gt[gt >= truncation] = truncation
    gt[gt <= -truncation] = -truncation
    if pad > 0:
        gt = np.pad(gt, pad, 'constant', constant_values=-truncation)

    bbox = np.zeros((3, 2))
    bbox[:, 0] = f.attrs['bbox'][:, 0] - pad*f.attrs['voxel_size']*np.ones((1,1,1))
    bbox[:, 1] = bbox[:, 0] + f.attrs['voxel_size'] * np.array(gt.shape)
    volume_shape = gt.shape
    voxel_size = f.attrs['voxel_size']


    # BELOW IS SILVANS TSDF FUSION

    box = np.array([bbox[0,:],bbox[1,:],bbox[2,:]])  # each cell depicts the interval where we will reconstruct the shape i.e.
    # [[-xmin,xmax],[-ymin,ymax],[-zmin,zmax]]
    # the resolution factor determines the truncation distance. The truncation distance is the resolution times
    # the resolution factor
    print(volume_shape)
    print(box)
    # box = np.array([[-8, 8], [-8, 8], [-8, 8]])
    # volume_shape = (200, 200, 200)
    tsdf = TSDFHandle.TSDF(bbox=box, resolution=0.01, resolution_factor=10, volume_shape=list(volume_shape))

    for i, frame in tqdm(enumerate(dataset), total=len(dataset)):
        if frame is None:
            print('None frame')
            continue
        
        extrinsics = np.linalg.inv(np.concatenate((frame['extrinsics'], np.array([[0, 0, 0, 1]])), axis=0)).astype(np.float32)
        # extrinsics = frame['extrinsics'].astype(np.float32)
        # print('translation: ', extrinsics[:, -1])
        # print(frame['intrinsics_stereo'])

        campose = np.matmul(frame['intrinsics_stereo'], extrinsics[0:3, 0:4]).astype(np.float32)

        depth = frame['stereo_depth'] #.astype(np.float)
        # depth = depth * frame['stereo_mask']
        # depth = depth.astype(np.uint16)
        weight_map = np.ones(depth.shape)
        tsdf.fuse(campose, depth.astype(np.float32), weight_map.astype(np.float32))
        # if i > 30:
        #     break

    # save visualization of sdf
    vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(tsdf.get_volume()[:, :, :, 0], level=0, spacing=(voxel_size, voxel_size, voxel_size))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    # mesh.show()
    # mesh.export('tof_tsdf_lewiner_office_1.ply')


    weights = np.asarray(tsdf.get_mask())
    mask = (weights > 0) # I think we need to think more about this. The masks are not the same between routedfusion and tsdf fusion. True but this
        # is the correct mask

    indices_x = mask.nonzero()[0]
    indices_y = mask.nonzero()[1]
    indices_z = mask.nonzero()[2]

    max_resolution = np.array(volume_shape).max()

    length = max_resolution*0.01

    volume = o3d.integration.UniformTSDFVolume(
                length=length,
                resolution=max_resolution,
                sdf_trunc=0.1,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
    tsdf_cube[:volume_shape[0], :volume_shape[1], :volume_shape[2]] = tsdf.get_volume()[:, :, :, 0]
                
    for i in range(indices_x.shape[0]):
        volume.set_tsdf_at(tsdf_cube[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
        volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])
                   

    print("Extract a filtered triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    del volume
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh('stereo_tsdf_office_1.ply', mesh)

    # from tqdm import tqdm
    # # from mayavi import mlab
    # import trimesh

    # import matplotlib.pyplot as plt
    # from utils.loading import load_config_from_yaml
    # from utils.setup import *


    # # get config
    # path_to_config = '/scratch-second/esandstroem/opportunistic_3d_capture/Erik_3D_Reconstruction_Project/src/RoutedFusion/configs/fusion/replica.yaml'
    # config = load_config_from_yaml(path_to_config)
    # ply_output = '/scratch-second/esandstroem/opportunistic_3d_capture/Erik_3D_Reconstruction_Project/data/replica/'
    # dataset_config = get_data_config(config, mode='train')
    # config.DATA.scene_list = dataset_config.scene_list
    # config.DATA.transform = dataset_config.transform
    # config.DATA.mode = dataset_config.mode
    # dataset = get_data(config.DATA.dataset, config.DATA)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=config.TRAINING.train_batch_size, shuffle=True)

    # # def pixel_to_camera_coord(point, intrinsics, z):

    # #     camera_coord = np.zeros(3, )

    # #     camera_coord[2] = z
    # #     camera_coord[1] = z * (point[1] - intrinsics[1, 2]) / intrinsics[1, 1]
    # #     camera_coord[0] = z * (point[0] - intrinsics[0, 1] * camera_coord[1] - intrinsics[0, 2]) / intrinsics[0, 0]

    # #     return camera_coord

    # # frame_counter = 0
    # # pointcloud = []

    # # # frames = np.random.choice(np.arange(0, len(dataset), 1), 20)
    # frames = np.arange(0, len(dataset), 1)


    # n_batches = int(len(dataset) / config.TESTING.test_batch_size)
    # for epoch in range(0, 2):
    #     for i, batch in enumerate(tqdm(loader, total=n_batches)):
    #         print(batch['frame_id'])
    #         if i == 10:
    #             break
    #     break
    # # for f in tqdm(frames, total=len(frames)):

    # #     frame = dataset[f]
    # #     print(frame['frame_id'])
    # #     if f == 5:
    # #         break
    # #         # print(batch)
    # #         # break
    # #     depth = frame['tof_depth']
    # #     mask = frame['mask']
    # #     # depth = np.flip(depth, axis=0)

    # #     plt.imshow(mask)
    # #     plt.show()
    # #     plt.imshow(depth)
    # #     plt.show()
   

    #     # for i in range(0, depth.shape[0]):
    #     #     for j in range(0, depth.shape[1]):

    #     #         z = depth[i, j]

    #     #         p = np.asarray([j, i, z])
    #     #         c = pixel_to_camera_coord([j, i], frame['intrinsics'], z)
    #     #         c = np.concatenate([c, np.asarray([1.])])
    #     #         w = np.dot(frame['extrinsics'], c)

    #     #         pointcloud.append(w)

    #     # frame_counter += 1
    #     # if frame_counter == 5:
    #     #     break



    #     # if (frame_counter + 1) % 5 == 0:
    #     #     print(frame_counter)
    #     #     array = np.asarray(pointcloud)
    #     #     print(np.max(array, axis=0))
    #     #     print(np.min(array, axis=0))
        
    #     #     mlab.points3d(array[:, 0],
    #     #                   array[:, 1],
    #     #                   array[:, 2],
    #     #                   scale_factor=0.05)
        
    #     #     mlab.show()
    #     #     mlab.close(all=True)

    # # array = np.asarray(pointcloud)
    # # print(np.max(array, axis=0))
    # # print(np.min(array, axis=0))
    # # print(array.shape)

    # # pointcloud = trimesh.points.PointCloud(array[:, :3]) # I should save this point cloud for all depth modalities and see the reconstruction F-score as a sanity method
    # # pointcloud.show()
    # # # pointcloud.color = np.ones((len(pointcloud.vertices), 4))
    # # pointcloud.export(ply_output + 'pc90.ply')

    # # array = np.asarray(pointcloud)
    # # # mlab.points3d(array[:, 0],
    # #               array[:, 1],
    # #               array[:, 2],
    # #               scale_factor=0.05)
    
    # # mlab.show()
