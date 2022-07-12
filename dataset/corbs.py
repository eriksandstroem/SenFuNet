import os

import sys
import numpy as np

from skimage import io
from torch.utils.data import Dataset

import h5py
import matplotlib.pyplot as plt

# uncomment to run train_fusion and test_fusion
from dataset.associate import associate
from dataset.colmap import read_array


from pyquaternion import Quaternion


class CoRBS(Dataset):

    # NOTE: For now, the dataset class can only load one scene at a time
    def __init__(self, config_data):
        super(CoRBS, self).__init__()

        self.root_dir = os.getenv(
            config_data.root_dir
        )  # when training on local scratch

        # os.getenv returns none when the input does not exist. When
        # it returns none, we want to train on the work folder
        if not self.root_dir:
            self.root_dir = config_data.root_dir

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

        self._scenes = []

        self.__init_dataset()

    def __init_dataset(self):

        # read paths to data from scene list file
        with open(os.path.join(self.root_dir, self.scene_list), "r") as file:
            for (
                line
            ) in (
                file
            ):  # only contains one line now since we only load one scene at a time
                line = line.split(" ")
                self._scenes.append(
                    line[0].split("/")[0]
                )  # change this into append when we use more scenes
                trajectory_file = os.path.join(
                    self.root_dir, line[4][:-1]
                )  # make this into a directory when we use more scenes
                rgb_file = os.path.join(self.root_dir, line[2])
                depth_file = os.path.join(self.root_dir, line[3])
                self.stereo_path = os.path.join(self.root_dir, line[0])
                self.tof_path = os.path.join(self.root_dir, line[1])
                self.rgb_path = os.path.join(self.root_dir, line[1])

        # read all files for pose, rgb, and depth
        self.poses = {}
        with open(trajectory_file, "r") as file:
            for line in file:
                # skip comment lines
                if line[0] == "#":
                    continue
                elems = line.rstrip().split(" ")
                timestamp = float(elems[0])
                pose = [float(e) for e in elems[1:]]
                self.poses[timestamp] = pose

        self.rgb_frames = {}
        with open(rgb_file, "r") as file:
            for line in file:
                # skip comment lines
                if line[0] == "#":
                    continue
                timestamp, file_path = line.rstrip().split(" ")
                timestamp = float(timestamp)
                self.rgb_frames[timestamp] = file_path

        self.depth_frames = {}
        with open(depth_file, "r") as file:
            for line in file:
                # skip comment lines
                if line[0] == "#":
                    continue
                timestamp, file_path = line.rstrip().split(" ")
                timestamp = float(timestamp)
                self.depth_frames[timestamp] = file_path

        # match pose to rgb timestamp
        rgb_matches = associate(
            self.poses, self.rgb_frames, offset=0.0, max_difference=0.02
        )
        # build mapping databases to get matches from pose timestamp to frame timestamp
        self.pose_to_rgb = {t_p: t_r for (t_p, t_r) in rgb_matches}

        # match poses that are matched with rgb to a corresponding depth timestamp
        depth_matches = associate(
            self.pose_to_rgb, self.depth_frames, offset=0.0, max_difference=0.02
        )
        # build mapping databases to get matches from pose timestamp to frame timestamp
        self.pose_to_depth = {t_p: t_d for (t_p, t_d) in depth_matches}
        self.poses_matched = {t_p: self.poses[t_p] for (t_p, t_r) in rgb_matches}

    @property
    def scenes(self):
        return self._scenes

    def __len__(self):
        return len(self.poses_matched)

    def __getitem__(self, item):

        sample = dict()
        sample["item_id"] = item

        timestamp_pose = list(self.poses_matched.keys())[item]
        timestamp_rgb = self.pose_to_rgb[timestamp_pose]
        timestamp_depth = self.pose_to_depth[timestamp_pose]

        # read RGB frame
        rgb_file = os.path.join(
            self.rgb_path, self.rgb_frames[timestamp_rgb].replace("\\", "/")
        )
        rgb_image = io.imread(rgb_file).astype(np.float32)

        step_x = rgb_image.shape[0] / self.resolution_tof[0]
        step_y = rgb_image.shape[1] / self.resolution_tof[1]

        index_y = [int(step_y * i) for i in range(0, int(rgb_image.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in range(0, int(rgb_image.shape[0] / step_x))]

        rgb_image = rgb_image[:, index_y]
        rgb_image = rgb_image[index_x, :]
        sample["image"] = np.asarray(rgb_image) / 255

        frame_id = "{}/{}".format(self._scenes[0], str(timestamp_pose))
        sample["frame_id"] = frame_id

        # read kinect depth file
        depth_file = os.path.join(
            self.tof_path, self.depth_frames[timestamp_depth].replace("\\", "/")
        )
        depth_tof = io.imread(depth_file).astype(np.float32)
        depth_tof /= 5000.0

        step_x = depth_tof.shape[0] / self.resolution_tof[0]
        step_y = depth_tof.shape[1] / self.resolution_tof[1]

        index_y = [int(step_y * i) for i in range(0, int(depth_tof.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in range(0, int(depth_tof.shape[0] / step_x))]

        depth_tof = depth_tof[:, index_y]
        depth_tof = depth_tof[index_x, :]
        sample["tof_depth"] = np.asarray(depth_tof)

        # read colmap stereo depth file
        try:
            stereo_file = os.path.join(
                self.stereo_path,
                self.rgb_frames[timestamp_rgb].replace("rgb\\", "") + ".geometric.bin",
            )
            depth_stereo = read_array(stereo_file)
        except FileNotFoundError:
            print("stereo frame not found")
            return None

        step_x = depth_stereo.shape[0] / self.resolution_stereo[0]
        step_y = depth_stereo.shape[1] / self.resolution_stereo[1]

        index_y = [
            int(step_y * i) for i in range(0, int(depth_stereo.shape[1] / step_y))
        ]
        index_x = [
            int(step_x * i) for i in range(0, int(depth_stereo.shape[0] / step_x))
        ]

        depth_stereo = depth_stereo[:, index_y]
        depth_stereo = depth_stereo[index_x, :]
        sample["stereo_depth"] = np.asarray(depth_stereo)

        # define mask
        mask = depth_stereo > self.min_depth_stereo
        mask = np.logical_and(mask, depth_stereo < self.max_depth_stereo)

        # do not integrate depth values close to the image boundary
        mask[0 : self.mask_stereo_height, :] = 0
        mask[-self.mask_stereo_height : -1, :] = 0
        mask[:, 0 : self.mask_stereo_width] = 0
        mask[:, -self.mask_stereo_width : -1] = 0
        sample["stereo_mask"] = mask

        mask = depth_tof > self.min_depth_tof
        mask = np.logical_and(mask, depth_tof < self.max_depth_tof)

        # do not integrate depth values close to the image boundary
        mask[0 : self.mask_tof_height, :] = 0
        mask[-self.mask_tof_height : -1, :] = 0
        mask[:, 0 : self.mask_tof_width] = 0
        mask[:, -self.mask_tof_width : -1] = 0
        sample["tof_mask"] = mask

        # load extrinsics
        rotation = self.poses_matched[timestamp_pose][3:]
        rotation = Quaternion(rotation[-1], rotation[0], rotation[1], rotation[2])
        rotation = rotation.rotation_matrix
        translation = self.poses_matched[timestamp_pose][:3]

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = translation
        sample["extrinsics"] = extrinsics

        # load intrinsics
        intrinsics_stereo = np.asarray(
            [
                [
                    468.60 * self.resolution_stereo[1] / 640,
                    0.0,
                    318.27 * self.resolution_stereo[1] / 640,
                ],
                [
                    0.0,
                    468.61 * self.resolution_stereo[0] / 480,
                    243.99 * self.resolution_stereo[0] / 480,
                ],
                [0.0, 0.0, 1.0],
            ]
        )

        sample["intrinsics_stereo"] = intrinsics_stereo

        intrinsics_tof = np.asarray(
            [
                [
                    468.60 * self.resolution_tof[1] / 640,
                    0.0,
                    318.27 * self.resolution_tof[1] / 640,
                ],
                [
                    0.0,
                    468.61 * self.resolution_tof[0] / 480,
                    243.99 * self.resolution_tof[0] / 480,
                ],
                [0.0, 0.0, 1.0],
            ]
        )

        sample["intrinsics_tof"] = intrinsics_tof

        # convert key image ndarray to compatible pytorch tensor shape. The function also converts the ndarrays to tensors, but this is not necessary as the pytorch dataloader does this anyway in a step later.
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_grid(self, scene, truncation):
        file = os.path.join(self.root_dir, scene, "sdf_" + scene + ".hdf")

        # read from hdf file!
        f = h5py.File(file, "r")
        voxels = np.array(f["sdf"]).astype(np.float16)

        voxels[voxels > truncation] = truncation
        voxels[voxels < -truncation] = -truncation
        # Add padding to grid to give more room to fusion net
        voxels = np.pad(voxels, self.pad, "constant", constant_values=-truncation)

        print(scene, voxels.shape)
        bbox = np.zeros((3, 2))
        bbox[:, 0] = f.attrs["bbox"][:, 0] - self.pad * f.attrs["voxel_size"] * np.ones(
            (1, 1, 1)
        )
        bbox[:, 1] = bbox[:, 0] + f.attrs["voxel_size"] * np.array(voxels.shape)

        return voxels, bbox, f.attrs["voxel_size"]
