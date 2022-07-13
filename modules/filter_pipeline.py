import torch

from modules.filtering_net import *
import math
import numpy as np


class Filter_Pipeline(torch.nn.Module):
    def __init__(self, config):

        super(Filter_Pipeline, self).__init__()

        self.config = config

        self._filtering_network = FilteringNet(config)

    def _filtering(self, neighborhood):
        if self.config.FILTERING_MODEL.CONV3D_MODEL.fixed:
            with torch.no_grad():
                output = self._filtering_network.forward(neighborhood)
        else:
            output = self._filtering_network.forward(neighborhood)

        return output

    def _prepare_input_training(self, volumes, bbox, device):
        # Now we have a bounding box which we know have valid indices. Now it is time to extract this volume from the input grid,
        # which is already on the gpu.

        neighborhood = None
        for k, volume in enumerate(volumes.keys()):
            if volumes[volume].dim() == 3:  # tsdf of weight volume
                n_hood = volumes[volume][
                    bbox[0, 0] : bbox[0, 1],
                    bbox[1, 0] : bbox[1, 1],
                    bbox[2, 0] : bbox[2, 1],
                ].unsqueeze_(0)

            else:  # dim 4 for feature volume
                n_hood = volumes[volume][
                    bbox[0, 0] : bbox[0, 1],
                    bbox[1, 0] : bbox[1, 1],
                    bbox[2, 0] : bbox[2, 1],
                    :,
                ]
                n_hood = n_hood.permute(3, 0, 1, 2)
            if k == 0:
                neighborhood = n_hood
            else:
                neighborhood = torch.cat(
                    (neighborhood, n_hood), dim=0
                )  # shape (C, D, H ,W)

        output = neighborhood.unsqueeze_(0).float().to(device)  # add batch dimension
        del neighborhood

        return output

    def _prepare_local_grids(self, bbox, database, scene):
        output = dict()
        # pad the local grids so that the dimension is divisible by half the chunk size and then ad
        # the chunk size divided by 4 so that we can update only the central region
        divisible_by = self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size / 2
        if (
            bbox[0, 1] - bbox[0, 0]
        ) % divisible_by != 0:  # I use 8 here because we need at least 4 due to the box_shift variable regardless
            # of network_depth
            pad_x = divisible_by - (bbox[0, 1] - bbox[0, 0]) % divisible_by
        else:
            pad_x = 0

        if (bbox[1, 1] - bbox[1, 0]) % divisible_by != 0:
            pad_y = divisible_by - (bbox[1, 1] - bbox[1, 0]) % divisible_by
        else:
            pad_y = 0

        if (bbox[2, 1] - bbox[2, 0]) % divisible_by != 0:
            pad_z = divisible_by - (bbox[2, 1] - bbox[2, 0]) % divisible_by
        else:
            pad_z = 0

        # pad the local grid
        pad = torch.nn.ReplicationPad3d((0, int(pad_z), 0, int(pad_y), 0, int(pad_x)))
        # pad the grid with the chunk size divided by 4 along each dimension
        extra_pad = int(self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size / 4)
        extra_pad = torch.nn.ReplicationPad3d(extra_pad)

        for sensor_ in self.config.DATA.input:
            # extract bbox from global grid
            tsdf = database[scene]["tsdf_" + sensor_][
                bbox[0, 0] : bbox[0, 1],
                bbox[1, 0] : bbox[1, 1],
                bbox[2, 0] : bbox[2, 1],
            ]

            weights = database[scene]["weights_" + sensor_][
                bbox[0, 0] : bbox[0, 1],
                bbox[1, 0] : bbox[1, 1],
                bbox[2, 0] : bbox[2, 1],
            ]

            feat = database[scene]["features_" + sensor_][
                bbox[0, 0] : bbox[0, 1],
                bbox[1, 0] : bbox[1, 1],
                bbox[2, 0] : bbox[2, 1],
                :,
            ]

            feat = feat.permute(3, 0, 1, 2)
            # concatenate the two grids and make them on the form N, C, D, H, W
            local_grid = torch.cat(
                (tsdf.unsqueeze(0), weights.unsqueeze(0), feat), dim=0
            ).unsqueeze(
                0
            )  # add batch dimension
            local_grid = pad(local_grid.float())
            local_grid = extra_pad(local_grid.float())

            output[sensor_] = local_grid

        return output, int(pad_x), int(pad_y), int(pad_z)

    def filter(
        self, scene, database, device
    ):  # here we use a stride which is half the chunk size
        self.device = device

        indices = np.zeros_like(database.scenes_gt[scene].volume)
        for sensor_ in self.config.DATA.input:
            indices = np.logical_or(
                indices, database.fusion_weights[sensor_][scene] > 0
            )

        uninit_indices = np.invert(indices)
        indices = np.transpose(indices.nonzero()).astype(np.int16)
        uninit_indices = np.transpose(uninit_indices.nonzero()).astype(np.int16)
        chunk_size = self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size

        # get minimum box size
        x_size = indices[:, 0].max() - indices[:, 0].min()
        y_size = indices[:, 1].max() - indices[:, 1].min()
        z_size = indices[:, 2].max() - indices[:, 2].min()

        bbox = np.array(
            [
                [
                    indices[:, 0].min(),
                    indices[:, 0].min() + x_size + 1,
                ],  # + 1 here because we want to include the max index in the bbox because we later do min:max during extraction
                [
                    indices[:, 1].min(),
                    indices[:, 1].min() + y_size + 1,
                ],  # + 1 here because we want to include the max index in the bbox because we later do min:max during extraction
                [indices[:, 2].min(), indices[:, 2].min() + z_size + 1],
            ]
        )  # + 1 here because we want to include the max index in the bbox because we later do min:max during extraction

        # prepare local grids
        local_grids, pad_x, pad_y, pad_z = self._prepare_local_grids(
            bbox, database, scene
        )

        filtered_local_grid = torch.zeros(
            tuple(local_grids[self.config.DATA.input[0]][0, 0, :, :, :].shape)
        )

        sensor_weighting_local_grid = torch.zeros_like(filtered_local_grid)
        if self.config.FILTERING_MODEL.CONV3D_MODEL.outlier_channel:
            sensor_weighting_local_grid = sensor_weighting_local_grid.unsqueeze(0)
            sensor_weighting_local_grid = torch.cat(
                (sensor_weighting_local_grid, sensor_weighting_local_grid), dim=0
            )

        # traverse the local grid, extracting chunks from the local grid to feed to the
        # filtering network one at a time.
        moving_bbox = np.zeros_like(bbox)

        x_size, y_size, z_size = local_grids[self.config.DATA.input[0]][
            0, 0, :, :, :
        ].shape
        for i in range(
            math.ceil(2 * x_size / chunk_size) - 1
        ):  # multiply by 2 since we have a stride that is half the chunk size
            moving_bbox[0, 0] = (
                i * 0.5 * chunk_size
            )  # multiply by 0.5 because the stride is half the chunk size
            moving_bbox[0, 1] = moving_bbox[0, 0] + chunk_size
            for j in range(math.ceil(2 * y_size / chunk_size) - 1):
                moving_bbox[1, 0] = j * 0.5 * chunk_size
                moving_bbox[1, 1] = moving_bbox[1, 0] + chunk_size
                for k in range(math.ceil(2 * z_size / chunk_size) - 1):
                    moving_bbox[2, 0] = k * 0.5 * chunk_size
                    moving_bbox[2, 1] = moving_bbox[2, 0] + chunk_size

                    input_ = dict()
                    for sensor_ in self.config.DATA.input:
                        input_[sensor_] = local_grids[sensor_][
                            :,
                            :,
                            moving_bbox[0, 0] : moving_bbox[0, 1],
                            moving_bbox[1, 0] : moving_bbox[1, 1],
                            moving_bbox[2, 0] : moving_bbox[2, 1],
                        ].to(self.device)

                    with torch.no_grad():
                        input_["test_mode"] = True
                        sub_filter_dict = self._filtering(input_)
                        if sub_filter_dict is None:
                            print("encountered nan in filtering net. Exit")
                            return

                    del input_

                    sensor_weighting = (
                        sub_filter_dict["sensor_weighting"].cpu().detach()
                    )
                    if self.config.FILTERING_MODEL.CONV3D_MODEL.outlier_channel:
                        sensor_weighting_local_grid[
                            :,
                            moving_bbox[0, 0]
                            + int(chunk_size / 4) : moving_bbox[0, 1]
                            - int(chunk_size / 4),
                            moving_bbox[1, 0]
                            + int(chunk_size / 4) : moving_bbox[1, 1]
                            - int(chunk_size / 4),
                            moving_bbox[2, 0]
                            + int(chunk_size / 4) : moving_bbox[2, 1]
                            - int(chunk_size / 4),
                        ] = sensor_weighting[
                            :,
                            int(chunk_size / 4) : -int(chunk_size / 4),
                            int(chunk_size / 4) : -int(chunk_size / 4),
                            int(chunk_size / 4) : -int(chunk_size / 4),
                        ]
                    else:
                        sensor_weighting_local_grid[
                            moving_bbox[0, 0]
                            + int(chunk_size / 4) : moving_bbox[0, 1]
                            - int(chunk_size / 4),
                            moving_bbox[1, 0]
                            + int(chunk_size / 4) : moving_bbox[1, 1]
                            - int(chunk_size / 4),
                            moving_bbox[2, 0]
                            + int(chunk_size / 4) : moving_bbox[2, 1]
                            - int(chunk_size / 4),
                        ] = sensor_weighting[
                            int(chunk_size / 4) : -int(chunk_size / 4),
                            int(chunk_size / 4) : -int(chunk_size / 4),
                            int(chunk_size / 4) : -int(chunk_size / 4),
                        ]

                    sub_tsdf = sub_filter_dict["tsdf"].cpu().detach()

                    del sub_filter_dict

                    # insert sub_tsdf into the local filtered grid
                    filtered_local_grid[
                        moving_bbox[0, 0]
                        + int(chunk_size / 4) : moving_bbox[0, 1]
                        - int(chunk_size / 4),
                        moving_bbox[1, 0]
                        + int(chunk_size / 4) : moving_bbox[1, 1]
                        - int(chunk_size / 4),
                        moving_bbox[2, 0]
                        + int(chunk_size / 4) : moving_bbox[2, 1]
                        - int(chunk_size / 4),
                    ] = sub_tsdf[
                        int(chunk_size / 4) : -int(chunk_size / 4),
                        int(chunk_size / 4) : -int(chunk_size / 4),
                        int(chunk_size / 4) : -int(chunk_size / 4),
                    ]

                    del sub_tsdf

        # transfer the local_filtered_grid to the global grid
        # first remove the padding
        if self.config.FILTERING_MODEL.CONV3D_MODEL.outlier_channel:
            sensor_weighting_local_grid = sensor_weighting_local_grid[
                :,
                int(chunk_size / 4) : -int(chunk_size / 4) - pad_x,
                int(chunk_size / 4) : -int(chunk_size / 4) - pad_y,
                int(chunk_size / 4) : -int(chunk_size / 4) - pad_z,
            ]

            database.sensor_weighting[scene][
                :,
                bbox[0, 0] : bbox[0, 1],
                bbox[1, 0] : bbox[1, 1],
                bbox[2, 0] : bbox[2, 1],
            ] = sensor_weighting_local_grid.numpy().squeeze()
            # I write to all voxels in the local grid, even the uninitialized, but here I replace the uninitialized
            # voxel values with their default value
            database.sensor_weighting[scene][
                :, uninit_indices[:, 0], uninit_indices[:, 1], uninit_indices[:, 2]
            ] = -1
        else:
            sensor_weighting_local_grid = sensor_weighting_local_grid[
                int(chunk_size / 4) : -int(chunk_size / 4) - pad_x,
                int(chunk_size / 4) : -int(chunk_size / 4) - pad_y,
                int(chunk_size / 4) : -int(chunk_size / 4) - pad_z,
            ]

            database.sensor_weighting[scene][
                bbox[0, 0] : bbox[0, 1],
                bbox[1, 0] : bbox[1, 1],
                bbox[2, 0] : bbox[2, 1],
            ] = sensor_weighting_local_grid.numpy().squeeze()
            # I write to all voxels in the local grid, even the uninitialized, but here I replace the uninitialized
            # voxel values with their default value
            database.sensor_weighting[scene][
                uninit_indices[:, 0], uninit_indices[:, 1], uninit_indices[:, 2]
            ] = -1
        del sensor_weighting_local_grid

        filtered_local_grid = filtered_local_grid[
            int(chunk_size / 4) : -int(chunk_size / 4) - pad_x,
            int(chunk_size / 4) : -int(chunk_size / 4) - pad_y,
            int(chunk_size / 4) : -int(chunk_size / 4) - pad_z,
        ]

        database.filtered[scene].volume[
            bbox[0, 0] : bbox[0, 1], bbox[1, 0] : bbox[1, 1], bbox[2, 0] : bbox[2, 1]
        ] = filtered_local_grid.numpy().squeeze()

        # I write to all voxels in the local grid, even the uninitialized, but here I replace the uninitialized
        # voxel values with their default value
        database.filtered[scene].volume[
            uninit_indices[:, 0], uninit_indices[:, 1], uninit_indices[:, 2]
        ] = self.config.DATA.init_value

        del filtered_local_grid

    def filter_training(
        self, input_dir, database, epoch, frame, scene_id, sensor, device
    ):
        self.device = device

        indices = input_dir["indices"].cpu()
        del input_dir["indices"]

        output = self.request_random_bbox(indices, epoch, sensor, frame)

        if output is None:
            return None
        else:
            bbox = output[0]
            valid_indices = output[1]

        neighborhood = dict()
        neighborhood["test_mode"] = False
        for sensor_ in self.config.DATA.input:
            if sensor_ == sensor:
                in_dir = {"tsdf": input_dir["tsdf"], "weights": input_dir["weights"]}
                if (
                    self.config.FILTERING_MODEL.CONV3D_MODEL.features_to_weight_head
                ):
                    in_dir["features"] = input_dir["features"]
            else:
                in_dir = {
                    "tsdf": database[scene_id]["tsdf_" + sensor_],
                    "weights": database[scene_id]["weights_" + sensor_],
                }
                if (
                    self.config.FILTERING_MODEL.CONV3D_MODEL.features_to_weight_head
                ):
                    in_dir["features"] = database[scene_id]["features_" + sensor_]

            neighborhood[sensor_] = self._prepare_input_training(in_dir, bbox, device)

        tsdf_filtered = self._filtering(neighborhood)

        if tsdf_filtered is None:
            return "save_and_exit"

        gt_vol = database[scene_id]["gt"]

        del neighborhood

        # mask target for loss
        tsdf_target = gt_vol[
            bbox[0, 0] : bbox[0, 1], bbox[1, 0] : bbox[1, 1], bbox[2, 0] : bbox[2, 1]
        ]

        tsdf_target = tsdf_target.float()

        if (
            self.config.LOSS.alpha_supervision
            or self.config.LOSS.alpha_single_sensor_supervision
        ):
            if self.config.LOSS.alpha_supervision:
                proxy_alpha = database[scene_id]["proxy_alpha"]
                # mask target for loss
                alpha_target = proxy_alpha[
                    bbox[0, 0] : bbox[0, 1],
                    bbox[1, 0] : bbox[1, 1],
                    bbox[2, 0] : bbox[2, 1],
                ]

                alpha_target = alpha_target.float()
                alpha_target = alpha_target.to(device)

            alpha = tsdf_filtered["sensor_weighting"]

        del gt_vol

        # compute a mask determining the valid indices in the tsdf_filtered variable
        # There is only a translation shift between the origin of the gt_vol and the
        # tsdf_filtered variable. Thus, we are able to take the valid_indices minus
        # bbox[:, 0] in order to get the correct valid indices for tsdf_filtered
        valid_indices = valid_indices - bbox[:, 0]

        tsdf_target = tsdf_target.to(device)

        output = dict()
        output["tsdf_filtered_grid"] = tsdf_filtered
        output["tsdf_target_grid"] = tsdf_target
        if (
            self.config.LOSS.alpha_supervision
            or self.config.LOSS.alpha_single_sensor_supervision
        ):
            if self.config.LOSS.alpha_supervision:
                output["proxy_alpha_grid"] = alpha_target
            output["alpha_grid"] = alpha

        return output

    def request_random_bbox(self, indices, epoch, sensor, frame):

        # get minimum box size
        x_size = indices[:, 0].max() - indices[:, 0].min()
        y_size = indices[:, 1].max() - indices[:, 1].min()
        z_size = indices[:, 2].max() - indices[:, 2].min()

        bbox_input = np.array(
            [
                [indices[:, 0].min(), indices[:, 0].min() + x_size],
                [indices[:, 1].min(), indices[:, 1].min() + y_size],
                [indices[:, 2].min(), indices[:, 2].min() + z_size],
            ]
        )
        bbox = np.zeros_like(bbox_input)
        # extract a random location chunk inside the minimum bound. If the minimum bound is smaller
        # than the chunk size, extract the minimum bound
        idx_threshold = 2000

        # we sample at most 600 times a random box and if any box is higher than some threshold, we select this box
        for i in range(600):
            if x_size / self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size > 1:
                bbox[0, 0] = np.random.random_integers(
                    bbox_input[0, 0],
                    bbox_input[0, 1]
                    - self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size,
                )
                bbox[0, 1] = (
                    bbox[0, 0] + self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size
                )
            else:
                bbox[0, 0] = bbox_input[0, 0]
                bbox[0, 1] = bbox_input[0, 1]
            if y_size / self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size > 1:
                bbox[1, 0] = np.random.random_integers(
                    bbox_input[1, 0],
                    bbox_input[1, 1]
                    - self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size,
                )
                bbox[1, 1] = (
                    bbox[1, 0] + self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size
                )
            else:
                bbox[1, 0] = bbox_input[1, 0]
                bbox[1, 1] = bbox_input[1, 1]
            if z_size / self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size > 1:
                bbox[2, 0] = np.random.random_integers(
                    bbox_input[2, 0],
                    bbox_input[2, 1]
                    - self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size,
                )
                bbox[2, 1] = (
                    bbox[2, 0] + self.config.FILTERING_MODEL.CONV3D_MODEL.chunk_size
                )
            else:
                bbox[2, 0] = bbox_input[2, 0]
                bbox[2, 1] = bbox_input[2, 1]

            # make sure that each dimension of the bounding box is divisible by 2
            # this was only necessary when I used the Unet 3D nets as weighting networks.
            if (
                bbox[0, 1] - bbox[0, 0]
            ) % 2 ** self.config.FILTERING_MODEL.CONV3D_MODEL.network_depth != 0:
                bbox[0, 1] -= (
                    bbox[0, 1] - bbox[0, 0]
                ) % 2 ** self.config.FILTERING_MODEL.CONV3D_MODEL.network_depth
            if (
                bbox[1, 1] - bbox[1, 0]
            ) % 2 ** self.config.FILTERING_MODEL.CONV3D_MODEL.network_depth != 0:
                bbox[1, 1] -= (
                    bbox[1, 1] - bbox[1, 0]
                ) % 2 ** self.config.FILTERING_MODEL.CONV3D_MODEL.network_depth
            if (
                bbox[2, 1] - bbox[2, 0]
            ) % 2 ** self.config.FILTERING_MODEL.CONV3D_MODEL.network_depth != 0:
                bbox[2, 1] -= (
                    bbox[2, 1] - bbox[2, 0]
                ) % 2 ** self.config.FILTERING_MODEL.CONV3D_MODEL.network_depth

            # compute a mask determining what indices should be used in the loss out of all indices in the bbox. Note
            # that the bbox is not the full min bounding volume of the indices, but only a random extraction
            # according to the chunk size. Thus, we need to select the valid indices within the chunk volume. We don't necessarily need to do this, since we can also used indices that were not updated by the feature net to train the weighting network, but for training consistency, I train both networks on the exact same indices.
            valid_indices = (
                (indices[:, 0] >= bbox[0, 0])
                & (indices[:, 0] < bbox[0, 1])
                &  # I think I can use equals and less than here since the bbox
                # is defined by the max index value
                (indices[:, 1] >= bbox[1, 0])
                & (indices[:, 1] < bbox[1, 1])
                & (indices[:, 2] >= bbox[2, 0])
                & (indices[:, 2] < bbox[2, 1])
            )

            valid_indices = torch.nonzero(valid_indices)[
                :, 0
            ]  # gives valid indices in indices variable but not in global grid
            valid_indices = indices[
                valid_indices, :
            ]  # extract the indices in the global grid for the valid indices

            if valid_indices.shape[0] > idx_threshold:
                return bbox, valid_indices

        print(
            "The desired amount of valid indices were not met or no valid indices were found"
        )
        return None
