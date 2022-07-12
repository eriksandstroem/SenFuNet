import torch
from tqdm import tqdm
import math

from modules.fuse_pipeline import Fuse_Pipeline
from modules.filter_pipeline import Filter_Pipeline

import numpy as np


class Pipeline(torch.nn.Module):
    def __init__(self, config):

        super(Pipeline, self).__init__()

        self.config = config

        # setup pipeline
        self.fuse_pipeline = Fuse_Pipeline(config)
        if config.FILTERING_MODEL.do:
            if config.FILTERING_MODEL.model == "3dconv":
                self.filter_pipeline = Filter_Pipeline(config)
            else:
                self.filter_pipeline = (
                    None  # used when we run the tsdf fusion or routedfusion
                )
        else:
            self.filter_pipeline = None

    def forward(self, batch, database, epoch, device):  # train step
        scene_id = batch["frame_id"][0].split("/")[0]

        frame = batch["frame_id"][0].split("/")[-1]

        fused_output = self.fuse_pipeline.fuse_training(batch, database, device)

        if self.config.FILTERING_MODEL.do:
            if self.filter_pipeline is not None:
                filtered_output = self.filter_pipeline.filter_training(
                    fused_output,
                    database,
                    epoch,
                    frame,
                    scene_id,
                    batch["sensor"],
                    device,
                )
            else:
                filtered_output = None

            if filtered_output == "save_and_exit":
                return "save_and_exit"

            if filtered_output is not None:
                fused_output["filtered_output"] = filtered_output
            else:
                if not self.config.FILTERING_MODEL.model == "routedfusion":
                    return None

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
                batch["routing_net"] = "self._routing_network_" + sensor_
                batch["mask"] = batch[sensor_ + "_mask"]
                if self.config.FILTERING_MODEL.model == "routedfusion":
                    batch["sensor"] = self.config.DATA.input[0]
                else:
                    batch["sensor"] = sensor_

                batch["routingNet"] = sensor_  # used to be able to train routedfusion
                batch["fusionNet"] = sensor_  # used to be able to train routedfusion
                self.fuse_pipeline.fuse(batch, database, device)
            else:
                for sensor_ in sensors:
                    if (
                        sensor_ + "_depth"
                    ) in batch:  # None on the Replica dataset when simulating sensors of different frame rates
                        batch["depth"] = batch[sensor_ + "_depth"]
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
                        self.fuse_pipeline.fuse(batch, database, device)

        if self.filter_pipeline is not None:
            # run filtering network on all voxels which have a non-zero weight
            for scene in database.filtered.keys():
                self.filter_pipeline.filter(scene, database, device)

    def test_tsdf(self, val_loader, val_dataset, val_database, sensors, device):

        for k, batch in tqdm(enumerate(val_loader), total=len(val_dataset)):

            if (
                self.config.ROUTING.do
                and self.config.FILTERING_MODEL.model == "tsdf_early_fusion"
            ):
                batch["routing_net"] = "self._routing_network"
                batch["sensor"] = self.config.DATA.input[0]
                batch[
                    "fusionNet"
                ] = None  # We don't use a fusion net during early fusion
                self.fuse_pipeline.fuse(batch, val_database, device)
            else:
                for sensor_ in sensors:
                    batch["depth"] = batch[sensor_ + "_depth"]
                    batch["routing_net"] = "self._routing_network_" + sensor_
                    batch["mask"] = batch[sensor_ + "_mask"]
                    batch["sensor"] = sensor_
                    batch[
                        "routingNet"
                    ] = sensor_  # used to be able to train routedfusion
                    batch[
                        "fusionNet"
                    ] = sensor_  # used to be able to train routedfusion
                    self.fuse_pipeline.fuse(batch, val_database, device)

        if self.config.FILTERING_MODEL.do:
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
