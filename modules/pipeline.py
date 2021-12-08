import torch
import datetime
import time
from tqdm import tqdm
import random
import math

from modules.fuse_pipeline import Fuse_Pipeline
from modules.filter_pipeline import Filter_Pipeline

import math
import numpy as np
from scipy import ndimage


class Pipeline(torch.nn.Module):
    def __init__(self, config):

        super(Pipeline, self).__init__()

        self.config = config

        # setup pipeline
        self.fuse_pipeline = Fuse_Pipeline(config)
        if config.FILTERING_MODEL.model == "3dconv":
            self.filter_pipeline = Filter_Pipeline(config)
        else:
            self.filter_pipeline = (
                None  # used when we run the tsdf fusion or routedfusion
            )

    def forward(self, batch, database, epoch, device):  # train step
        scene_id = batch["frame_id"][0].split("/")[0]

        frame = batch["frame_id"][0].split("/")[-1]

        fused_output = self.fuse_pipeline.fuse_training(batch, database, device)

        if self.filter_pipeline is not None:
            filtered_output = self.filter_pipeline.filter_training(
                fused_output, database, epoch, frame, scene_id, batch["sensor"], device
            )  # CHANGE
        else:
            filtered_output = None

        if filtered_output == "save_and_exit":
            return "save_and_exit"

        if filtered_output is not None:
            fused_output["filtered_output"] = filtered_output
        else:
            if not self.config.FILTERING_MODEL.model == "routedfusion":
                return None

        if self.config.LOSS.alpha_2d_supervision:
            fused_output["sensor"] = batch["sensor"]
            for sensor_ in self.config.DATA.input:
                fused_output[sensor_ + "_depth"] = batch[sensor_ + "_depth"].to(device)
                fused_output[sensor_ + "_mask"] = batch[sensor_ + "_mask"].to(device)
                fused_output["gt"] = batch["gt"].to(device)

        return fused_output

    def test(self, loader, dataset, database, sensors, device):

        for k, batch in tqdm(enumerate(loader), total=len(dataset)):
            if self.config.DATA.collaborative_reconstruction:
                if (
                    math.ceil(
                        int(batch["frame_id"][0].split("/")[-1])
                        / self.config.DATA.frames_per_chunk
                    )
                    % 2
                    == 0
                ):
                    sensor_ = sensors[0]
                else:
                    sensor_ = sensors[1]

                batch["depth"] = batch[sensor_ + "_depth"]
                # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_)
                batch["routing_net"] = "self._routing_network_" + sensor_
                batch["mask"] = batch[sensor_ + "_mask"]
                if self.config.FILTERING_MODEL.model == "routedfusion":
                    batch["sensor"] = self.config.DATA.input[0]
                else:
                    batch["sensor"] = sensor_

                batch["routingNet"] = sensor_  # used to be able to train routedfusion
                batch["fusionNet"] = sensor_  # used to be able to train routedfusion
                output = self.fuse_pipeline.fuse(batch, database, device)
            else:
                # print(batch['frame_id'])
                for sensor_ in sensors:

                    batch["depth"] = batch[sensor_ + "_depth"]
                    # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_)
                    batch["routing_net"] = "self._routing_network_" + sensor_
                    batch["mask"] = batch[sensor_ + "_mask"]
                    if self.config.FILTERING_MODEL.model == "routedfusion":
                        batch["sensor"] = self.config.DATA.input[0]
                    else:
                        batch["sensor"] = sensor_

                    batch[
                        "routingNet"
                    ] = sensor_  # used to be able to train routedfusion
                    batch[
                        "fusionNet"
                    ] = sensor_  # used to be able to train routedfusion
                    output = self.fuse_pipeline.fuse(batch, database, device)

                    # return

            # if k == 5:
            #     break # debug

        if self.filter_pipeline is not None:
            # run filtering network on all voxels which have a non-zero weight
            for scene in database.filtered.keys():
                self.filter_pipeline.filter(scene, database, device)

            # apply outlier filter i.e. make the weights of the outlier voxels zero so
            # that they are not used in the evaluation of the IoU
            # This step might not be needed anymore - check!
            for scene in database.filtered.keys():
                mask = np.zeros_like(database[scene]["gt"])
                and_mask = np.ones_like(database[scene]["gt"])
                sensor_mask = dict()

                for sensor_ in self.config.DATA.input:
                    # print(sensor_)
                    weights = database.fusion_weights[sensor_][scene]
                    mask = np.logical_or(mask, weights > 0)
                    and_mask = np.logical_and(and_mask, weights > 0)
                    sensor_mask[sensor_] = weights > 0
                    # break

                # load weighting sensor grid
                if self.config.FILTERING_MODEL.outlier_channel:
                    sensor_weighting = database[scene]["sensor_weighting"][1, :, :, :]
                else:
                    sensor_weighting = database[scene]["sensor_weighting"]

                only_one_sensor_mask = np.logical_xor(mask, and_mask)
                for sensor_ in self.config.DATA.input:
                    only_sensor_mask = np.logical_and(
                        only_one_sensor_mask, sensor_mask[sensor_]
                    )
                    if sensor_ == self.config.DATA.input[0]:
                        rem_indices = np.logical_and(
                            only_sensor_mask, sensor_weighting < 0.5
                        )
                    else:
                        # before I fixed the bug always ended up here when I had tof and stereo as sensors
                        # but this would mean that for the tof sensor I removed those indices
                        # if alpha was larger than 0.5 which it almost always is. This means that
                        # essentially all (cannot be 100 % sure) voxels where we only integrated
                        # tof, was removed. Since the histogram is essentially does not have
                        # any voxels with trust less than 0.5, we also removed all alone stereo voxels
                        # so at the end we end up with a mask very similar to the and_mask
                        rem_indices = np.logical_and(
                            only_sensor_mask, sensor_weighting > 0.5
                        )

                    # rem_indices = rem_indices.astype(dtype=bool)
                    database[scene]["weights_" + sensor_][rem_indices] = 0

    def test_tsdf(self, val_loader, val_dataset, val_database, sensors, device):

        for k, batch in tqdm(enumerate(val_loader), total=len(val_dataset)):

            if (
                self.config.ROUTING.do
                and self.config.FILTERING_MODEL.model == "tsdf_early_fusion"
            ):
                # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_)
                batch["routing_net"] = "self._routing_network"
                batch["sensor"] = self.config.DATA.input[0]
                output = self.fuse_pipeline.fuse(batch, val_database, device)
            else:
                for sensor_ in sensors:
                    # print(sensor_)
                    batch["depth"] = batch[sensor_ + "_depth"]
                    # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_)
                    batch["routing_net"] = "self._routing_network_" + sensor_
                    batch["mask"] = batch[sensor_ + "_mask"]
                    batch["sensor"] = sensor_
                    batch[
                        "routingNet"
                    ] = sensor_  # used to be able to train routedfusion
                    batch[
                        "fusionNet"
                    ] = sensor_  # used to be able to train routedfusion
                    output = self.fuse_pipeline.fuse(batch, val_database, device)

            # if k == 10:
            #     break # debug

        # perform the fusion of the grids
        if self.config.FILTERING_MODEL.model == "tsdf_early_fusion":
            for scene in val_database.filtered.keys():
                val_database.filtered[scene].volume = val_database.tsdf[
                    self.config.DATA.input[0]
                ][scene].volume

        elif (
            self.config.FILTERING_MODEL.model == "tsdf_middle_fusion"
        ):  # this is weighted average fusion
            for scene in val_database.filtered.keys():
                weight_sum = np.zeros_like(val_database.filtered[scene].volume)
                for sensor_ in sensors:
                    weight_sum += val_database.fusion_weights[sensor_][scene]
                    val_database.filtered[scene].volume += (
                        val_database.tsdf[sensor_][scene].volume
                        * val_database.fusion_weights[sensor_][scene]
                    )
                val_database.filtered[scene].volume = np.divide(
                    val_database.filtered[scene].volume,
                    weight_sum,
                    out=np.zeros_like(weight_sum),
                    where=weight_sum != 0.0,
                )

                val_database.sensor_weighting[scene] = np.divide(
                    val_database.fusion_weights[sensors[0]][scene],
                    weight_sum,
                    out=np.zeros_like(weight_sum),
                    where=weight_sum != 0.0,
                )

        elif (
            self.config.FILTERING_MODEL.model == "tsdf_plain_average_fusion"
        ):  # simple average fusion
            for scene in val_database.filtered.keys():
                for sensor_ in sensors:
                    val_database.filtered[scene].volume += val_database.tsdf[sensor_][
                        scene
                    ].volume
                val_database.filtered[scene].volume /= len(
                    sensors
                )  # this can shift the outliers more i.e. we take the average
                # even if only one sensor integrates compared to the middle fusion case where we don't consider
                # uninitialized voxels in the average. But we cannot get rid of the outliers though....

                val_database.sensor_weighting[scene][:, :, :] = 0.5

        elif (
            self.config.FILTERING_MODEL.model == "tsdf_average_fusion"
        ):  # simple average fusion
            for scene in val_database.filtered.keys():
                weight_mask = np.zeros_like(val_database.filtered[scene].volume)
                for sensor_ in sensors:
                    val_database.filtered[scene].volume += val_database.tsdf[sensor_][
                        scene
                    ].volume
                    weight_mask += val_database.fusion_weights[sensor_][scene] > 0

                val_database.filtered[scene].volume = np.divide(
                    val_database.filtered[scene].volume,
                    weight_mask,
                    out=np.zeros_like(weight_mask),
                    where=weight_mask != 0.0,
                )

                # here I don't divide by 2 everywhere, but 1 where we only have 1 sensor integration
                val_database.sensor_weighting[scene][
                    :, :, :
                ] = 0.5  # this is not correct. It should be 0.5 and 0 and 1, but I don't care about this now

        elif (
            self.config.FILTERING_MODEL.model == "tsdf_plain_average_andmask_fusion"
        ):  # simple average fusion
            for scene in val_database.filtered.keys():
                and_mask = np.ones_like(val_database.filtered[scene].volume)
                for sensor_ in sensors:
                    and_mask = np.logical_and(
                        and_mask, val_database.fusion_weights[sensor_][scene] > 0
                    )
                    val_database.filtered[scene].volume += val_database.tsdf[sensor_][
                        scene
                    ].volume
                val_database.filtered[scene].volume /= len(
                    sensors
                )  # this can shift the outliers more i.e. we take the average
                # even if only one sensor integrates compared to the middle fusion case where we don't consider
                # uninitialized voxels in the average. But we cannot get rid of the outliers though....
                for sensor_ in sensors:
                    val_database.feature_weights[sensor_][scene] = and_mask.astype(
                        np.float32
                    )
                val_database.sensor_weighting[scene][:, :, :] = 0.5

    def test_step(
        self, batch, database, sensors, device
    ):  # used for trajectory performance plot

        for sensor_ in sensors:

            batch["depth"] = batch[sensor_ + "_depth"]
            # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_)
            batch["routing_net"] = "self._routing_network_" + sensor_
            batch["mask"] = batch[sensor_ + "_mask"]
            batch["sensor"] = sensor_
            batch["routingNet"] = sensor_  # used to be able to train routedfusion
            batch["fusionNet"] = sensor_  # used to be able to train routedfusion
            output = self.fuse_pipeline.fuse(batch, database, device)

        # run filtering network on all voxels which have a non-zero weight
        scene = batch["frame_id"][0].split("/")[0]
        self.filter_pipeline.filter(scene, database, device)

        # apply outlier filter i.e. make the weights of the outlier voxels zero so
        # that they are not used in the evaluation of the IoU
        mask = np.zeros_like(database[scene]["gt"])
        and_mask = np.ones_like(database[scene]["gt"])
        sensor_mask = dict()

        for sensor_ in sensors:
            # print(sensor_)
            weights = database.fusion_weights[sensor_][scene]
            mask = np.logical_or(mask, weights > 0)
            and_mask = np.logical_and(and_mask, weights > 0)
            sensor_mask[sensor_] = weights > 0
            # break

        # load weighting sensor grid
        if self.config.FILTERING_MODEL.outlier_channel:
            sensor_weighting = database[scene]["sensor_weighting"][1, :, :, :]
        else:
            sensor_weighting = database[scene]["sensor_weighting"]

        only_one_sensor_mask = np.logical_xor(mask, and_mask)
        for sensor_ in sensors:
            only_sensor_mask = np.logical_and(
                only_one_sensor_mask, sensor_mask[sensor_]
            )
            if sensor_ == sensors[0]:
                rem_indices = np.logical_and(only_sensor_mask, sensor_weighting < 0.5)
            else:
                # before I fixed the bug always ended up here when I had tof and stereo as sensors
                # but this would mean that for the tof sensor I removed those indices
                # if alpha was larger than 0.5 which it almost always is. This means that
                # essentially all (cannot be 100 % sure) voxels where we only integrated
                # tof, was removed. Since the histogram is essentially does not have
                # any voxels with trust less than 0.5, we also removed all alone stereo voxels
                # so at the end we end up with a mask very similar to the and_mask
                rem_indices = np.logical_and(only_sensor_mask, sensor_weighting > 0.5)

            # rem_indices = rem_indices.astype(dtype=bool)
            database[scene]["weights_" + sensor_][rem_indices] = 0

    def test_speed(
        self, val_loader, val_dataset, val_database, sensors, device
    ):  # used to evaluate speed of algorithm

        for k, batch in tqdm(enumerate(val_loader), total=len(val_dataset)):

            if self.config.DATA.collaborative_reconstruction:
                if (
                    math.ceil(
                        int(batch["frame_id"][0].split("/")[-1])
                        / self.config.DATA.frames_per_chunk
                    )
                    % 2
                    == 0
                ):
                    sensor_ = sensors[0]
                else:
                    sensor_ = sensors[1]

                batch["depth"] = batch[sensor_ + "_depth"]
                # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_)
                batch["routing_net"] = "self._routing_network_" + sensor_
                batch["mask"] = batch[sensor_ + "_mask"]
                if self.config.FILTERING_MODEL.model == "routedfusion":
                    batch["sensor"] = self.config.DATA.input[0]
                else:
                    batch["sensor"] = sensor_

                batch["routingNet"] = sensor_  # used to be able to train routedfusion
                batch["fusionNet"] = sensor_  # used to be able to train routedfusion
                output = self.fuse_pipeline.fuse(batch, val_database, device)
            else:
                # print(batch['frame_id'])
                for sensor_ in sensors:

                    batch["depth"] = batch[sensor_ + "_depth"]
                    # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_)
                    batch["routing_net"] = "self._routing_network_" + sensor_
                    batch["mask"] = batch[sensor_ + "_mask"]
                    if self.config.FILTERING_MODEL.model == "routedfusion":
                        batch["sensor"] = self.config.DATA.input[0]
                    else:
                        batch["sensor"] = sensor_

                    batch[
                        "routingNet"
                    ] = sensor_  # used to be able to train routedfusion
                    batch[
                        "fusionNet"
                    ] = sensor_  # used to be able to train routedfusion
                    output = self.fuse_pipeline.fuse(batch, val_database, device)

                    # return

            # if k == 5:
            #     break # debug

            if self.filter_pipeline is not None:
                # run filtering network on all voxels which have a non-zero weight
                for scene in val_database.filtered.keys():
                    self.filter_pipeline.filter(
                        scene, val_database, device
                    )  # it would be good
                    # to implement a function filter_speed which takes an additional bbox as input
                    # from the min bbox of the updated indices and runs only over those indices.
                    # this requires outputting this bbox from the for loop of fusion steps

                # apply outlier filter i.e. make the weights of the outlier voxels zero so
                # that they are not used in the evaluation of the IoU
                for scene in val_database.filtered.keys():
                    mask = np.zeros_like(val_database[scene]["gt"])
                    and_mask = np.ones_like(val_database[scene]["gt"])
                    sensor_mask = dict()

                    for sensor_ in self.config.DATA.input:
                        # print(sensor_)
                        weights = val_database.fusion_weights[sensor_][scene]
                        mask = np.logical_or(mask, weights > 0)
                        and_mask = np.logical_and(and_mask, weights > 0)
                        sensor_mask[sensor_] = weights > 0
                        # break

                    # load weighting sensor grid
                    if self.config.FILTERING_MODEL.outlier_channel:
                        sensor_weighting = val_database[scene]["sensor_weighting"][
                            1, :, :, :
                        ]
                    else:
                        sensor_weighting = val_database[scene]["sensor_weighting"]

                    only_one_sensor_mask = np.logical_xor(mask, and_mask)
                    for sensor_ in self.config.DATA.input:
                        only_sensor_mask = np.logical_and(
                            only_one_sensor_mask, sensor_mask[sensor_]
                        )
                        if sensor_ == self.config.DATA.input[0]:
                            rem_indices = np.logical_and(
                                only_sensor_mask, sensor_weighting < 0.5
                            )
                        else:
                            # before I fixed the bug always ended up here when I had tof and stereo as sensors
                            # but this would mean that for the tof sensor I removed those indices
                            # if alpha was larger than 0.5 which it almost always is. This means that
                            # essentially all (cannot be 100 % sure) voxels where we only integrated
                            # tof, was removed. Since the histogram is essentially does not have
                            # any voxels with trust less than 0.5, we also removed all alone stereo voxels
                            # so at the end we end up with a mask very similar to the and_mask
                            rem_indices = np.logical_and(
                                only_sensor_mask, sensor_weighting > 0.5
                            )

                        # rem_indices = rem_indices.astype(dtype=bool)
                        val_database[scene]["weights_" + sensor_][rem_indices] = 0

    def test_step_video(
        self, batch, database, sensor, sensors, device
    ):  # used for video creation

        if sensor == "fused" or sensor == "weighting":
            for sensor_ in sensors:
                batch["depth"] = batch[sensor_ + "_depth"]
                # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_)
                batch["routing_net"] = "self._routing_network_" + sensor_
                batch["mask"] = batch[sensor_ + "_mask"]
                batch["sensor"] = sensor_
                batch["routingNet"] = sensor_  # used to be able to train routedfusion
                batch["fusionNet"] = sensor_  # used to be able to train routedfusion
                output = self.fuse_pipeline.fuse(batch, database, device)

            # run filtering network on all voxels which have a non-zero weight
            scene = batch["frame_id"][0].split("/")[0]
            self.filter_pipeline.filter(scene, database, device)

        else:
            batch["depth"] = batch[sensor + "_depth"]
            # batch['confidence_threshold'] = eval('self.config.ROUTING.threshold_' + sensor_)
            batch["routing_net"] = "self._routing_network_" + sensor
            batch["mask"] = batch[sensor + "_mask"]
            batch["sensor"] = sensor
            batch["routingNet"] = sensor  # used to be able to train routedfusion
            batch["fusionNet"] = sensor  # used to be able to train routedfusion
            output = self.fuse_pipeline.fuse(batch, database, device)
