import os
import glob

import sys
import random
import numpy as np

from skimage import io, transform
from skimage.color import rgb2gray
from skimage import filters
from torch.utils.data import Dataset

import h5py
import matplotlib.pyplot as plt
# uncomment to run python dataset/corbs.py main
# path_to_dataset_module =  '/home/esandstroem/scratch-second/euler_project/src/late_fusion/dataset/' #'/cluster/project/cvl/esandstroem/src/late_fusion/utils/'
# sys.path.append(path_to_dataset_module) # needed in order to load read_array and associate
# from associate import associate
# from colmap import read_array

# uncomment to run train_fusion and test_fusion
from dataset.associate import associate
from dataset.colmap import read_array


from pyquaternion import Quaternion



class CoRBS(Dataset):

    # NOTE: For now, the dataset class can only load one scene at a time
    def __init__(self, config_data):
        super(CoRBS, self).__init__()

        self.root_dir = os.getenv(config_data.root_dir) # when training on local scratch
        
        # os.getenv returns none when the input does not exist. When 
        # it returns none, we want to train on the work folder
        if not self.root_dir:
            self.root_dir  = config_data.root_dir
 
        self.sampling_density_stereo = config_data.sampling_density_stereo
        self.sampling_density_tof = config_data.sampling_density_tof

        self.resolution_stereo = (config_data.resy_stereo, config_data.resx_stereo)

        self.resolution_tof = (config_data.resy_tof, config_data.resx_tof)

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

        self.__init_dataset()

        # Not needed for corbs
        # if self.mode == 'val':
        #     keys = list(self.poses_matched.keys())
        #     keys = keys[::4]
        #     self.poses_matched_tmp = dict()
        #     for key in keys:
        #         self.poses_matched_tmp[key] = self.poses_matched[key]
        #     self.poses_matched = self.poses_matched_tmp


    def __init_dataset(self):

        # read paths to data from scene list file
        with open(os.path.join(self.root_dir, self.scene_list), 'r') as file:
            for line in file: # only contains one line now since we only load one scene at a time
                line = line.split(' ') 
                self._scenes.append(line[0].split('/')[0]) # change this into append when we use more scenes
                trajectory_file = os.path.join(self.root_dir, line[4][:-1]) # make this into a directory when we use more scenes
                rgb_file = os.path.join(self.root_dir, line[2])
                depth_file = os.path.join(self.root_dir, line[3])
                self.stereo_path = os.path.join(self.root_dir, line[0])
                self.tof_path = os.path.join(self.root_dir, line[1])
                self.rgb_path = os.path.join(self.root_dir, line[1])

        
        # read all files for pose, rgb, and depth
        self.poses = {}
        with open(trajectory_file, 'r') as file:
            for line in file:
                # skip comment lines
                if line[0] == '#':
                    continue
                elems = line.rstrip().split(' ')
                timestamp = float(elems[0])
                pose = [float(e) for e in elems[1:]]
                self.poses[timestamp] = pose
            
        self.rgb_frames = {}
        with open(rgb_file, 'r') as file:
            for line in file:
            # skip comment lines
                if line[0] == '#':
                    continue
                timestamp, file_path = line.rstrip().split(' ')
                timestamp = float(timestamp)
                self.rgb_frames[timestamp] = file_path

        self.depth_frames = {}
        with open(depth_file, 'r') as file:
            for line in file:
            # skip comment lines
                if line[0] == '#':
                    continue
                timestamp, file_path = line.rstrip().split(' ')
                timestamp = float(timestamp)
                self.depth_frames[timestamp] = file_path
            
        # match pose to rgb timestamp
        rgb_matches = associate(self.poses, self.rgb_frames, offset=0.0, max_difference=0.02)
        # build mapping databases to get matches from pose timestamp to frame timestamp
        self.pose_to_rgb = {t_p : t_r for (t_p, t_r) in rgb_matches}
            
        # match pose that are matched with rgb to depth timestamp
        depth_matches = associate(self.pose_to_rgb, self.depth_frames, offset=0.0, max_difference=0.02)
        # build mapping databases to get matches from pose timestamp to frame timestamp
        self.pose_to_depth = {t_p : t_d for (t_p, t_d) in depth_matches}
        self.poses_matched = {t_p: self.poses[t_p] for (t_p, t_r) in rgb_matches}


    @property
    def scenes(self):
        return self._scenes

    def __len__(self):
        return len(self.poses_matched)

    def __getitem__(self, item):

        sample = dict()
        sample['item_id'] = item

        timestamp_pose = list(self.poses_matched.keys())[item]
        timestamp_rgb = self.pose_to_rgb[timestamp_pose]
        timestamp_depth = self.pose_to_depth[timestamp_pose]

        # read RGB frame
        rgb_file = os.path.join(self.rgb_path, self.rgb_frames[timestamp_rgb].replace('\\', '/'))
        rgb_image = io.imread(rgb_file).astype(np.float32)

        # plt.imsave('imagef.png', np.asarray(rgb_image)/255)

        step_x = rgb_image.shape[0] / self.resolution_tof[0]
        step_y = rgb_image.shape[1] / self.resolution_tof[1]
            
        index_y = [int(step_y * i) for i in
                    range(0, int(rgb_image.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in
                    range(0, int(rgb_image.shape[0] / step_x))]

        rgb_image = rgb_image[:, index_y]
        rgb_image = rgb_image[index_x, :]
        sample['image'] = np.asarray(rgb_image).transpose(2, 0, 1)/255


        # plt.imsave('image.png', np.asarray(rgb_image)/255)
       

        frame_id = '{}/{}'.format(self._scenes[0], str(timestamp_pose))
        sample['frame_id'] = frame_id  

        # read kinect depth file
        depth_file = os.path.join(self.tof_path, self.depth_frames[timestamp_depth].replace('\\', '/'))
        depth_tof = io.imread(depth_file).astype(np.float32)
        depth_tof /= 5000.

        # plt.imsave('toff.png', np.asarray(depth_tof)/255)

        step_x = depth_tof.shape[0] / self.resolution_tof[0]
        step_y = depth_tof.shape[1] / self.resolution_tof[1]
            
        index_y = [int(step_y * i) for i in
                    range(0, int(depth_tof.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in
                    range(0, int(depth_tof.shape[0] / step_x))]

        depth_tof = depth_tof[:, index_y]
        depth_tof = depth_tof[index_x, :]
        sample['tof_depth'] = np.asarray(depth_tof) 
        # plt.imsave('tof.png', sample['tof_depth'])

        # read colmap stereo depth file
        try:
            stereo_file = os.path.join(self.stereo_path, self.rgb_frames[timestamp_rgb].replace('rgb\\', '') + '.geometric.bin')
            depth_stereo = read_array(stereo_file)
        except:
            print('stereo frame not found')
            return None

        # plt.imsave('stereof.png', np.asarray(depth_stereo)/255)
        step_x = depth_stereo.shape[0] / self.resolution_stereo[0]
        step_y = depth_stereo.shape[1] / self.resolution_stereo[1]

        index_y = [int(step_y * i) for i in
                   range(0, int(depth_stereo.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in
                   range(0, int(depth_stereo.shape[0] / step_x))]

        depth_stereo = depth_stereo[:, index_y]
        depth_stereo = depth_stereo[index_x, :]
        sample['stereo_depth'] = np.asarray(depth_stereo)
        # plt.imsave('stereo.png', sample['stereo_depth'])


        # define mask
        mask = (depth_stereo > self.min_depth_stereo)
        mask = np.logical_and(mask, depth_stereo < self.max_depth_stereo)

        # do not integrate depth values close to the image boundary
        mask[0:self.mask_stereo_height, :] = 0
        mask[-self.mask_stereo_height:-1, :] = 0
        mask[:, 0:self.mask_stereo_width] = 0
        mask[:, -self.mask_stereo_width:-1] = 0
        sample['stereo_mask'] = mask

        mask = (depth_tof > self.min_depth_tof)
        mask = np.logical_and(mask, depth_tof < self.max_depth_tof)

        # do not integrate depth values close to the image boundary
        mask[0:self.mask_tof_height, :] = 0
        mask[-self.mask_tof_height:-1, :] = 0
        mask[:, 0:self.mask_tof_width] = 0
        mask[:, -self.mask_tof_width:-1] = 0
        sample['tof_mask'] = mask

        # load extrinsics
        rotation = self.poses_matched[timestamp_pose][3:]
        rotation = Quaternion(rotation[-1], rotation[0], rotation[1], rotation[2])
        rotation = rotation.rotation_matrix
        translation = self.poses_matched[timestamp_pose][:3]
        
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = translation
        sample['extrinsics'] = extrinsics

        # load intrinsics 
        intrinsics_stereo = np.asarray([[468.60*self.resolution_stereo[1]/640, 0., 318.27*self.resolution_stereo[1]/640],
                                 [0., 468.61*self.resolution_stereo[0]/480, 243.99*self.resolution_stereo[0]/480],
                                 [0., 0., 1.]])

        sample['intrinsics_stereo'] = intrinsics_stereo

        intrinsics_tof = np.asarray([[468.60*self.resolution_tof[1]/640, 0., 318.27*self.resolution_tof[1]/640],
                                 [0., 468.61*self.resolution_tof[0]/480, 243.99*self.resolution_tof[0]/480],
                                 [0., 0., 1.]])

        sample['intrinsics_tof'] = intrinsics_tof


        # if self.transform:
        #     sample = self.transform(sample)

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

        return voxels, bbox, f.attrs['voxel_size']



if __name__ == '__main__':


    from tqdm import tqdm

    path_to_utils_module =  '/home/esandstroem/scratch-second/euler_project/src/late_fusion/utils/' #'/cluster/project/cvl/esandstroem/src/late_fusion/utils/'
    sys.path.append(path_to_utils_module) # needed in order to load read_array and associate

    from loading import load_config_from_yaml
    import open3d as o3d

    from easydict import EasyDict
    # to run the open3d integration scheem you need to use the open3d_env but this does not have the tsdf library installed so
    # the below line needs to be commented out.
    # from tsdf import TSDFHandle
    import skimage
    import trimesh




    # get config
    path_to_config = '/home/esandstroem/scratch-second/euler_project/src/late_fusion/configs/fusion/corbs_euler.yaml'
    config = load_config_from_yaml(path_to_config)
    # dataset_config = get_data_config(config, mode='train')
    config.DATA.scene_list = config.DATA.test_scene_list
    # config.DATA.transform = dataset_config.transform
    config.DATA.mode = 'test'
    dataset = CoRBS(config.DATA) #get_data(config.DATA.dataset, config.DATA)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=config.TRAINING.train_batch_size, shuffle=False)

    truncation = 0.1
    pad = 0

    # load gt grid to get volume shape
    gt_path = '/home/esandstroem/scratch-second/euler_work/data/corbs/human/sdf_human.hdf'
    # read from hdf file!
    f = h5py.File(gt_path, 'r')

    gt = np.array(f['sdf']).astype(np.float16)
    gt[gt >= truncation] = truncation
    gt[gt <= -truncation] = -truncation
    if pad > 0:
        gt = np.pad(gt, pad, 'constant', constant_values=truncation)

    bbox = np.zeros((3, 2))
    bbox[:, 0] = f.attrs['bbox'][:, 0] - pad*f.attrs['voxel_size']*np.ones((1,1,1))
    # bbox[1, 0] = bbox[1, 0] 
    bbox[:, 1] = bbox[:, 0] + f.attrs['voxel_size'] * np.array(gt.shape)

    # in order to avoid the blobs under the feet we crop the gt grid from below


    volume_shape = gt.shape
    voxel_size = f.attrs['voxel_size']


    # BELOW IS SILVANS TSDF FUSION

    # box = np.array([bbox[0,:],bbox[1,:],bbox[2,:]])  # each cell depicts the interval where we will reconstruct the shape i.e.
    # # [[-xmin,xmax],[-ymin,ymax],[-zmin,zmax]]
    # # the resolution factor determines the truncation distance. The truncation distance is the resolution times
    # # the resolution factor
    # print(volume_shape)
    # print(box)
    # # box = np.array([[-8, 8], [-8, 8], [-8, 8]])
    # # volume_shape = (200, 200, 200)
    # tsdf = TSDFHandle.TSDF(bbox=box, resolution=0.01, resolution_factor=10, volume_shape=list(volume_shape))

    # for i, frame in tqdm(enumerate(dataset), total=len(dataset)):
    #     # if i % 4 == 0:
    #     #     if frame is None:
    #     #         print('None frame')
    #     #         continue
            
    #     #     extrinsics = np.linalg.inv(frame['extrinsics'].astype(np.float32))
    #     #     # extrinsics = frame['extrinsics'].astype(np.float32)
    #     #     # print('translation: ', extrinsics[:, -1])
    #     #     # print(frame['intrinsics_stereo'])

    #     #     campose = np.matmul(frame['intrinsics_stereo'], extrinsics[0:3, 0:4]).astype(np.float32)

    #     #     depth = frame['stereo_depth']
    #     #     # depth = depth * frame['stereo_mask']
    #     #     # depth = depth * frame['stereo_mask']
    #     #     # depth = depth.astype(np.uint16)
    #     #     weight_map = np.ones(depth.shape)
    #     #     tsdf.fuse(campose, depth.astype(np.float32), weight_map.astype(np.float32))
    #     #     # if i > 300:
    #     #     #     break
    #     if frame is None:
    #         print('None frame')
    #         continue
            
    #     extrinsics = np.linalg.inv(frame['extrinsics'].astype(np.float32))
    #         # extrinsics = frame['extrinsics'].astype(np.float32)
    #         # print('translation: ', extrinsics[:, -1])
    #         # print(frame['intrinsics_stereo'])

    #     campose = np.matmul(frame['intrinsics_stereo'], extrinsics[0:3, 0:4]).astype(np.float32)

    #     depth = frame['stereo_depth']
    #     depth = depth * frame['stereo_mask']
    #         # depth = depth * frame['stereo_mask']
    #         # depth = depth * frame['stereo_mask']
    #         # depth = depth.astype(np.uint16)
    #     weight_map = np.ones(depth.shape)
    #     tsdf.fuse(campose, depth.astype(np.float32), weight_map.astype(np.float32))
    #     # if i > 300:
    #     #     break

    # # save visualization of sdf
    # # vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(tsdf.get_volume()[:, :, :, 0], level=0, spacing=(voxel_size, voxel_size, voxel_size))
    # # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    # # # mesh.show()
    # # mesh.export('tof_tsdf_lewiner_full_mask.ply')


    # weights = np.asarray(tsdf.get_mask())
    # mask = (weights > 0) # I think we need to think more about this. The masks are not the same between routedfusion and tsdf fusion. True but this
    #     # is the correct mask

    # indices_x = mask.nonzero()[0]
    # indices_y = mask.nonzero()[1]
    # indices_z = mask.nonzero()[2]

    # max_resolution = np.array(volume_shape).max()

    # length = max_resolution*0.01

    # volume = o3d.integration.UniformTSDFVolume(
    #             length=length,
    #             resolution=max_resolution,
    #             sdf_trunc=0.1,
    #             color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    # tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
    # tsdf_cube[:volume_shape[0], :volume_shape[1], :volume_shape[2]] = tsdf.get_volume()[:, :, :, 0]
                
    # for i in range(indices_x.shape[0]):
    #     volume.set_tsdf_at(tsdf_cube[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
    #     volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])
                   

    # print("Extract a filtered triangle mesh from the volume and visualize it.")
    # mesh = volume.extract_triangle_mesh()
    # del volume
    # mesh.compute_vertex_normals()
    # # o3d.visualization.draw_geometries([mesh])
    # o3d.io.write_triangle_mesh('stereo_human.ply', mesh)
    # BELOW IS OPEN3D TSDF FUSION
    color = o3d.integration.TSDFVolumeColorType.NoColor
    # stereo_volume = o3d.integration.ScalableTSDFVolume(voxel_length=1./64.,
    #                                                              sdf_trunc=0.06,
    #                                                              color_type=color)

    
    resolution = volume_shape
    max_resolution = np.array(resolution).max()
    length = max_resolution*voxel_size

    stereo_volume = o3d.integration.UniformTSDFVolume(length=length,
                                                        resolution=max_resolution,
                                                        sdf_trunc=truncation,
                                                        color_type=color)

    stereo_volume.origin = f.attrs['bbox'][:, 0] - [pad*voxel_size, pad*voxel_size, pad*voxel_size]



    # kinect_volume = o3d.integration.ScalableTSDFVolume(voxel_length=1./64.,
    #                                                              sdf_trunc=0.06,
    #                                                              color_type=color)

    kinect_volume = o3d.integration.UniformTSDFVolume(length=length,
                                                        resolution=max_resolution,
                                                        sdf_trunc=truncation,
                                                        color_type=color)

    kinect_volume.origin = f.attrs['bbox'][:, 0] - [pad*voxel_size, pad*voxel_size, pad*voxel_size]

    #grid = dataset.get_grid(config.scene)

    def draw_camera(pose):
        center = pose[:3, 3]
        z_direction = pose[:3, 2]
        y_direction = pose[:3, 1]
        x_direction = pose[:3, 0]

        top_left = center + 0.05 * z_direction - 0.05 * x_direction + 0.05 * y_direction
        top_right = center + 0.05 * z_direction + 0.05 * x_direction + 0.05 * y_direction
        bottom_left = center + 0.05 * z_direction - 0.05 * x_direction - 0.05 * y_direction
        bottom_right = center + 0.05 * z_direction + 0.05 * x_direction - 0.05 * y_direction

        points  = [center, top_left, top_right, bottom_left, bottom_right]
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [3, 4], [2, 4]] 

        colors = [[1, 0, 0] for i in range(len(lines))]
        camera = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                      lines=o3d.utility.Vector2iVector(lines))
        camera.colors = o3d.utility.Vector3dVector(colors)
        return camera

    cameras = []
    images = []

    for i, frame in tqdm(enumerate(dataset), total=len(dataset)):
        if frame is None:
            print('None frame')
            continue
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width=frame['stereo_depth'].shape[1],   
                                                       height=frame['stereo_depth'].shape[0],
                                                       fx=frame['intrinsics_stereo'][0, 0],
                                                       fy=frame['intrinsics_stereo'][1, 1],
                                                       cx=frame['intrinsics_stereo'][0, 2],
                                                       cy=frame['intrinsics_stereo'][1, 2])

        rgb = o3d.geometry.Image(np.ones_like(frame['stereo_depth']))

        depth = frame['stereo_depth'].astype(np.float) * 1000.
        depth = depth.astype(np.uint16)
        depth = o3d.geometry.Image(depth)

        image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,
                                                                   depth,
                                                                   depth_scale=1000,
                                                                   depth_trunc=4.,
                                                                   convert_rgb_to_intensity=False)

        extrinsics = np.linalg.inv(frame['extrinsics'].astype(np.float32))


        stereo_volume.integrate(image,
                                intrinsics,
                                extrinsics)

        depth = frame['tof_depth'].astype(np.float) * 1000.
        depth = depth.astype(np.uint16)
        depth = o3d.geometry.Image(depth)

        image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,
                                                                   depth,
                                                                   depth_scale=1000,
                                                                   depth_trunc=4.,
                                                                   convert_rgb_to_intensity=False)

        kinect_volume.integrate(image,
                                intrinsics,
                                extrinsics)

        cameras.append(draw_camera(np.linalg.inv(extrinsics)))

        # if i > 300:
        #     break

    stereo_mesh = stereo_volume.extract_triangle_mesh()
    stereo_mesh.compute_vertex_normals()

    stereo_colors = np.zeros_like(stereo_mesh.vertex_colors)
    stereo_colors[:, 0] = 1
    stereo_mesh.vertex_colors = o3d.utility.Vector3dVector(stereo_colors)

    kinect_mesh = kinect_volume.extract_triangle_mesh()
    kinect_mesh.compute_vertex_normals()

    kinect_colors = np.zeros_like(kinect_mesh.vertex_colors)
    kinect_colors[:, 1] = 1
    kinect_mesh.vertex_colors = o3d.utility.Vector3dVector(kinect_colors)

    # vertices, faces, normals, _ = marching_cubes(grid['sdf'], level=0)
    # gt_mesh = o3d.geometry.TriangleMesh()
    # gt_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # gt_mesh.triangles = o3d.utility.Vector3iVector(faces)
    # gt_mesh.compute_vertex_normals()

    geometries = [kinect_mesh] + [stereo_mesh] + cameras
    o3d.visualization.draw_geometries(geometries)
    geometries = [kinect_mesh] + cameras
    o3d.visualization.draw_geometries(geometries)
    geometries = [stereo_mesh] + cameras
    o3d.visualization.draw_geometries(geometries)
    
    o3d.io.write_triangle_mesh('open3d_tsdf_fusion_stereo_desk.ply', stereo_mesh)
    o3d.io.write_triangle_mesh('open3d_tsdf_fusion_tof_desk.ply', kinect_mesh)