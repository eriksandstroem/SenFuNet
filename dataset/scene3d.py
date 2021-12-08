import os
import glob

import sys
import random
import numpy as np
import re

from skimage import io, transform
from skimage.color import rgb2gray
from skimage import filters
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from dataset.colmap import read_array
import itertools

from graphics import Voxelgrid
import h5py


class Scene3D(Dataset):

    def __init__(self, config_data):
        self.root_dir = os.getenv(config_data.root_dir)
        if self.root_dir:
            self.root_dir += '/cluster/work/cvl/esandstroem/data/scene3D' # when training on local scratch
        # os.getenv returns none when the input does not exist. When 
        # it returns none, we want to train on the work folder
        else:
            self.root_dir  = config_data.root_dir

        self.sampling_density_stereo = config_data.sampling_density_stereo
        self.sampling_density_tof = config_data.sampling_density_tof

        self.resolution_stereo = (config_data.resy_stereo, config_data.resx_stereo)

        self.resolution_tof = (config_data.resy_tof, config_data.resx_tof)

        self.resolution = (config_data.resy, config_data.resx)

        self.mask_stereo_width = config_data.mask_stereo_width
        self.mask_stereo_height = config_data.mask_stereo_height
        self.mask_tof_width = config_data.mask_tof_width
        self.mask_tof_height = config_data.mask_tof_height
        self.mask_height = config_data.mask_height
        self.mask_width = config_data.mask_width

        self.min_depth_stereo = config_data.min_depth_stereo
        self.max_depth_stereo = config_data.max_depth_stereo
        self.min_depth_tof = config_data.min_depth_tof
        self.max_depth_tof = config_data.max_depth_tof
        self.min_depth = config_data.min_depth
        self.max_depth = config_data.max_depth

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

        self.sensor_line_mapping = {'rgb': 0, 'camera_matrix': -1,
                                    'tof': 1, 'tof_2': 1, 'stereo': 2}

        self.scenedir = None

        self._load_color()
        self._load_cameras()
        self._load_depths()


    def _load_depths(self): # loads the paths of the noisy depth images to a list

        # reading files from list
        self.depth_images = dict()
        for sensor_ in self.input: # initialize empty lists
            self.depth_images[sensor_] = []

        with open(os.path.join(self.root_dir, self.scene_list), 'r') as scene_list:
            for line in scene_list:
                if len(line) > 1: # avoid parsing empty line only containing \n
                    line = line.split(' ')
                    for sensor_ in self.input:
                        if sensor_ == 'tof':
                            files = glob.glob(os.path.join(self.root_dir, line[self.sensor_line_mapping[sensor_]], '*.png'))
                        elif sensor_ == 'stereo':
                            files = glob.glob(os.path.join(self.root_dir, line[self.sensor_line_mapping[sensor_]], '*.geometric.bin'))
                        for file in files:
                            self.depth_images[sensor_].append(file)

        # perhaps it will be important to order the frames for testing and training the fusion network.
        for sensor_ in self.depth_images.keys():
            self.depth_images[sensor_]  = sorted(self.depth_images[sensor_] , key=lambda x: os.path.splitext(x.split('/')[-1])[0])

        # downsample dataset size to be comparable to neuralfusion
        # for sensor_ in self.depth_images.keys():
        #     self.depth_images[sensor_]  = self.depth_images[sensor_][::10]

    def _load_color(self):
        self.color_images = []

        # reading files from list
        with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:
            for line in file:
                if len(line) > 1: # avoid parsing empty line only containing \n
                    line = line.split(' ')
                    self._scenes.append(line[0].split('/')[0]) 
                    files = glob.glob(os.path.join(self.root_dir, line[self.sensor_line_mapping['rgb']], '*.png'))
                    for file in files:
                        self.color_images.append(file)

        self.color_images = sorted(self.color_images, key=lambda x: os.path.splitext(x.split('/')[-1])[0])

        # downsample dataset size to be comparable to neuralfusion
        # self.color_images = self.color_images[::10]

    def _load_cameras(self):
        def grouper_it(n, iterable):
            it = iter(iterable)
            while True:
                chunk_it = itertools.islice(it, n)
                try:
                    first_el = next(chunk_it)
                except StopIteration:
                    return
                yield itertools.chain((first_el,), chunk_it)
        self.cameras = dict()

        with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:
            for line in file:
                line = line.split(' ')
                if len(line) > 1: # avoid parsing empty line only containing \n
                    with open(os.path.join(self.root_dir, line[-1][:-1]), 'r') as traj_file:
                        chunk_iterable = grouper_it(5, traj_file)
                        for frame in chunk_iterable:
                            frame_id = next(frame)[:-1]
                            frame_id = re.split(r'\t+', frame_id.rstrip('\t'))[-1]
                            first = np.fromstring(next(frame), count=4, sep=' ', dtype=float)
                            second = np.fromstring(next(frame), count=4, sep=' ', dtype=float)
                            third = np.fromstring(next(frame), count=4, sep=' ', dtype=float)
                            fourth = np.fromstring(next(frame), count=4, sep=' ', dtype=float)
                           
                            extrinsics = np.zeros((4,4))
                            extrinsics[0, :] = first
                            extrinsics[1, :] = second
                            extrinsics[2, :] = third
                            extrinsics[3, :] = fourth
                            # I should not need to invert since the extrinsics are 
                            # from camera to world and that is what the extractor expects
                            # extrinsics = np.linalg.inv(extrinsics)
                            self.cameras[line[0].split('/')[0] + '/' + frame_id] = extrinsics


    @property
    def scenes(self):
        return self._scenes

    def __len__(self):
        return len(self.color_images)

    def __getitem__(self, item):

        # there is something strane if you print the item and frame here s.t. I don't print them in order
        # but when I print the frame id in the test function in the pipeline.py everything is in order.
        # I think the issue is with the printing and the need to flush.

        sample = dict()
        sample['item_id'] = item 

        # load rgb image
        file = self.color_images[item]
        pathsplit = file.split('/')
        scene = pathsplit[-3]
        frame = os.path.splitext(pathsplit[-1])[0]

        frame_id = '{}/{}'.format(scene, frame)

        image = io.imread(file)

        step_x = image.shape[0] / self.resolution[0]
        step_y = image.shape[1] / self.resolution[0]

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

        # load noisy depth maps
        for sensor_ in self.input:
            file = self.depth_images[sensor_][item]
            if sensor_ == 'tof':
                depth = io.imread(file).astype(np.float32)
                depth /= 1000.
            elif sensor_ == 'stereo':
                depth = read_array(file)

            try:
                step_x = depth.shape[0] / eval('self.resolution_' + sensor_ + '[0]')
                step_y = depth.shape[1] / eval('self.resolution_' + sensor_ + '[1]')
            except: # default values used in case sensor specific parameters do not exist
                step_x = depth.shape[0] / self.resolution[0]
                step_y = depth.shape[1] / self.resolution[1]

            index_y = [int(step_y * i) for i in
                       range(0, int(depth.shape[1] / step_y))]
            index_x = [int(step_x * i) for i in
                       range(0, int(depth.shape[0] / step_x))]

            depth = depth[:, index_y]
            depth = depth[index_x, :]


            sample[sensor_ + '_depth'] = np.asarray(depth)

            # plt.imsave('left' +frame +'.png', sample['image'])
            # plt.imsave(sensor_ + '_depth' +frame +'.png', sample[sensor_ + '_depth'])

       
            # define mask
            try:
                mask = (depth > eval('self.min_depth_' + sensor_))
                mask = np.logical_and(mask, depth < eval('self.max_depth_' + sensor_))

                # do not integrate depth values close to the image boundary
                mask[0:eval('self.mask_' + sensor_ + '_height'), :] = 0
                mask[-eval('self.mask_' + sensor_ + '_height'):-1, :] = 0
                mask[:, 0:eval('self.mask_' + sensor_ + '_width')] = 0
                mask[:, -eval('self.mask_' + sensor_ + '_width'):-1] = 0
                sample[sensor_ + '_mask'] = mask
            except:
                mask = (depth > self.min_depth)
                mask = np.logical_and(mask, depth < self.max_depth)

                # do not integrate depth values close to the image boundary
                mask[0:self.mask_height, :] = 0
                mask[-self.mask_height:-1, :] = 0
                mask[:, 0:self.mask_width] = 0
                mask[:, -self.mask_width:-1] = 0
                sample[sensor_ + '_mask'] = mask

        # load extrinsics
        extrinsics = self.cameras[scene + '/' + str(int(frame))]

        sample['extrinsics'] = extrinsics

        intrinsics_tof = np.asarray([[525.0*self.resolution_tof[1]/640, 0., 319.5*self.resolution_tof[1]/640],
                         [0., 525.0*self.resolution_tof[0]/480, 239.5*self.resolution_tof[0]/480],
                         [0., 0., 1.]])

        sample['intrinsics_tof'] = intrinsics_tof

        sample['intrinsics_tof_2'] = intrinsics_tof

        intrinsics_stereo = np.asarray([[525.0*self.resolution_stereo[1]/640, 0., 319.5*self.resolution_stereo[1]/640],
                                 [0., 525.0*self.resolution_stereo[0]/480, 239.5*self.resolution_stereo[0]/480],
                                 [0., 0., 1.]])

        sample['intrinsics_stereo'] = intrinsics_stereo

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
