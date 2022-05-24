import torch

# needed when plotting interactively
# import matplotlib.pyplot as plt

from modules.routing import ConfidenceRouting
from modules.extractor import Extractor
from modules.model import FusionNet
from modules.model_features import FeatureNet
from modules.model_features import FeatureResNet
from modules.integrator import Integrator


class Fuse_Pipeline(torch.nn.Module):
    def __init__(self, config):

        super(Fuse_Pipeline, self).__init__()

        self.config = config

        if config.ROUTING.do:
            # define model
            if config.ROUTING.intensity_grad:
                Cin = 2
            else:
                Cin = 0
            if config.FILTERING_MODEL.model == "tsdf_early_fusion":
                Cin += len(config.DATA.input)
                self._routing_network = ConfidenceRouting(
                    Cin=Cin,
                    F=config.ROUTING_MODEL.contraction,
                    batchnorms=config.ROUTING_MODEL.normalization,
                )
            else:
                self._routing_network = torch.nn.ModuleDict()
                Cin += 1
                for sensor_ in config.DATA.input:
                    self._routing_network[sensor_] = ConfidenceRouting(
                        Cin=Cin,
                        F=config.ROUTING_MODEL.contraction,
                        batchnorms=config.ROUTING_MODEL.normalization,
                    )

        else:
            self._routing_network = None

        config.FUSION_MODEL.trunc_value = config.DATA.trunc_value
        config.FUSION_MODEL.init_value = -config.DATA.init_value

        config.FEATURE_MODEL.n_points = config.FUSION_MODEL.n_points
        config.FEATURE_MODEL.n_points_tof = config.FUSION_MODEL.n_points_tof
        config.FEATURE_MODEL.n_points_stereo = config.FUSION_MODEL.n_points_stereo
        config.FEATURE_MODEL.n_tail_points_tof = config.FUSION_MODEL.n_tail_points_tof
        config.FEATURE_MODEL.n_tail_points_stereo = (
            config.FUSION_MODEL.n_tail_points_stereo
        )
        config.FEATURE_MODEL.resx = config.DATA.resx
        config.FEATURE_MODEL.resy = config.DATA.resy

        self.n_features = self.config.FEATURE_MODEL.n_features

        self._extractor = dict()
        self._fusion_network = torch.nn.ModuleDict()
        self._feature_network = torch.nn.ModuleDict()
        for sensor in config.DATA.input:
            self._extractor[sensor] = Extractor(config.FUSION_MODEL, sensor)
            if config.FUSION_MODEL.use_fusion_net:
                self._fusion_network[sensor] = FusionNet(config.FUSION_MODEL, sensor)
            if config.FEATURE_MODEL.use_feature_net:
                if config.FEATURE_MODEL.network == "resnet":
                    self._feature_network[sensor] = FeatureResNet(
                        config.FEATURE_MODEL, sensor
                    )
                else:
                    self._feature_network[sensor] = FeatureNet(
                        config.FEATURE_MODEL, sensor
                    )
            else:
                self._feature_network[sensor] = None

        self._integrator = Integrator(config.FUSION_MODEL)

    def _routing(self, data):
        if self.config.FILTERING_MODEL.model == "tsdf_early_fusion":
            for k, sensor_ in enumerate(self.config.DATA.input):
                if k == 0:
                    inputs = data[sensor_ + "_depth"].unsqueeze_(1)
                else:
                    inputs = torch.cat(
                        (data[sensor_ + "_depth"].unsqueeze_(1), inputs), 1
                    )

            inputs = inputs.to(self.device)

            est = self._routing_network.forward(inputs)

            frame = est[:, 0, :, :]
            confidence = torch.exp(-1.0 * est[:, 1, :, :])
        else:
            inputs = data["depth"].unsqueeze_(1)

            if self.config.ROUTING.intensity_grad:
                intensity = data["intensity"].unsqueeze_(1)
                grad = data["gradient"].unsqueeze_(1)
                inputs = torch.cat((intensity, grad, inputs), 1)

            inputs = inputs.to(self.device)

            est = self._routing_network[data["routingNet"]].forward(inputs)

            if self.config.ROUTING.dont_smooth_where_uncertain:
                frame = est[:, 0, :, :]
                confidence = torch.exp(-1.0 * est[:, 1, :, :])
                input_depth = data["depth"][:, 0, :, :][
                    confidence < self.config.ROUTING.threshold
                ]
                frame[confidence < self.config.ROUTING.threshold] = input_depth.to(
                    self.device
                )
            else:
                frame = est[:, 0, :, :]
                confidence = torch.exp(-1.0 * est[:, 1, :, :])

        return frame, confidence

    def _fusion(
        self,
        input_,
        input_features,
        values,
        sensor,
        fusionNet,
        gt_depth=None,
        extraction_band=11,
    ):  # TODO: adapt to when not using features
        output = dict()
        b, c, h, w = input_.shape
        if self.config.FUSION_MODEL.use_fusion_net:
            if self.config.FUSION_MODEL.fixed:
                with torch.no_grad():
                    # print('in: ', input_.sum())
                    tsdf_pred = self._fusion_network[fusionNet].forward(input_)
                    # print('out: ', tsdf_pred.sum())
            else:
                tsdf_pred = self._fusion_network[fusionNet].forward(input_)
        else:  # TSDF Fusion
            tsdf_pred = torch.zeros(
                (1, extraction_band, input_.shape[2], input_.shape[3])
            )
            temp = torch.linspace(
                (tsdf_pred.shape[1] - 1) / 200,
                -(tsdf_pred.shape[1] - 1) / 200,
                steps=tsdf_pred.shape[1],
            )
            temp = temp.unsqueeze_(0)
            temp = temp.unsqueeze_(-1)
            temp = temp.unsqueeze_(-1)
            try:
                resolution = eval("self.config.DATA.resx_" + sensor)
            except AttributeError:
                resolution = self.config.DATA.resx

            tsdf_pred[:, :, :, :] = temp.expand(-1, -1, resolution, resolution)

        if self.config.FEATURE_MODEL.use_feature_net:
            feat_pred = self._feature_network[sensor].forward(input_features[sensor])
        else:
            feat_pred = dict()
            feat_pred["feature"] = input_features[sensor]

        tsdf_pred = tsdf_pred.permute(0, 2, 3, 1)

        feat_pred = feat_pred["feature"].permute(0, 2, 3, 1)

        # # save feature maps
        # for i in range(feat_pred.shape[-1]):
        #     plt.imsave('/cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/' + sensor + '/' + str(i)+ '_'+ sensor+   '.jpeg', feat_pred[0, :, :, i].cpu().detach().numpy())

        # save confidence prediction
        # using exp(-relu(x)) activation
        # plt.imsave(sensor + '/' + sensor+   '_confidence.jpeg', torch.exp(-output['confidence_2d'][0, 0, :, :]).cpu().detach().numpy())
        # using sigmoid activation
        # plt.imsave(sensor + '/' + sensor+   '_confidence.jpeg', output['confidence_2d'][0, 0, :, :].cpu().detach().numpy())

        # for i in range(input_features[sensor].shape[1]):
        #     m = input_features[sensor][:, i, :, :].squeeze()
        #     plt.imsave('/cluster/project/cvl/esandstroem/src/late_fusion_3dconvnet/' + sensor + '/input_' + str(i)+ '_'+ sensor+   '.jpeg', m.cpu().detach().numpy())

        try:
            n_points = eval("self.config.FUSION_MODEL.n_points_" + sensor)
        except AttributeError:
            n_points = self.config.FUSION_MODEL.n_points

        tsdf_est = tsdf_pred.view(b, h * w, n_points)
        tsdf_new = torch.clamp(
            tsdf_est, -self.config.DATA.trunc_value, self.config.DATA.trunc_value
        )

        feature_est = feat_pred.view(b, h * w, 1, self.n_features)
        feature_est = feature_est.repeat(1, 1, n_points, 1)

        # computing weighted updates for loss calculation
        tsdf_old = values[
            "fusion_values"
        ]  # values that were used as input to the fusion net
        weights_old = values[
            "fusion_weights"
        ]  # weights that were used as input to the fusion net

        if self.config.FUSION_MODEL.use_fusion_net:
            tsdf_fused = (weights_old * tsdf_old + tsdf_new) / (
                weights_old + torch.ones_like(weights_old)
            )
            output[
                "tsdf_fused"
            ] = tsdf_fused  # fused version at the floating point location (we supervise on this) i.e.
            # we supervise in floating coordinate voxel space on the fused values

        output[
            "tsdf_est"
        ] = tsdf_new  # output from fusion net (used for integration into voxel grid)
        output[
            "feature_est"
        ] = feature_est  # output from fusion net (used for integration into voxel grid)

        return output

    def _prepare_fusion_input(
        self,
        frame,
        values_sensor,
        sensor,
        confidence=None,
        n_points=None,
        rgb=None,
        rgb_warp=None,
    ):  # TODO: adapt to when not using features

        # get frame shape
        b, h, w = frame.shape
        # extracting data
        # reshaping data
        tsdf_input = {}
        tsdf_weights = {}

        tsdf_input[sensor] = values_sensor[sensor]["fusion_values"].view(
            b, h, w, n_points
        )
        tsdf_weights[sensor] = values_sensor[sensor]["fusion_weights"].view(
            b, h, w, n_points
        )

        tsdf_frame = torch.unsqueeze(frame, -1)

        if rgb is not None:
            rgb = rgb.unsqueeze(-1)
            rgb = rgb.permute(3, 1, 2, 0)  # never use view here

        feature_input = dict()
        feature_input[sensor] = torch.unsqueeze(frame, -1)
        if rgb is not None:
            if sensor == "tof" and self.config.FEATURE_MODEL.w_rgb_tof:
                feature_input[sensor] = torch.cat((feature_input[sensor], rgb), dim=3)
            elif sensor != "tof":
                feature_input[sensor] = torch.cat((feature_input[sensor], rgb), dim=3)

        if confidence is not None and self.config.FEATURE_MODEL.confidence:
            confidence = torch.unsqueeze(confidence, -1)

            feature_input[sensor] = torch.cat(
                (feature_input[sensor], confidence), dim=3
            )

        if sensor == "stereo" and rgb_warp is not None:
            feature_input[sensor] = torch.cat((feature_input[sensor], rgb_warp), dim=3)

        # permuting input
        feature_input[sensor] = feature_input[sensor].permute(0, -1, 1, 2)

        del rgb
        # stacking input data

        if self.config.FUSION_MODEL.confidence:
            assert confidence is not None
            tsdf_confidence = torch.unsqueeze(confidence, -1)
            tsdf_input = torch.cat(
                [tsdf_frame, tsdf_confidence, tsdf_weights[sensor], tsdf_input[sensor]],
                dim=3,
            )
            del tsdf_confidence
        else:
            tsdf_input = torch.cat(
                [tsdf_frame, tsdf_weights[sensor], tsdf_input[sensor]], dim=3
            )

        # permuting input
        tsdf_input = tsdf_input.permute(0, -1, 1, 2)

        del tsdf_frame

        return tsdf_input, feature_input

    def _prepare_volume_update(
        self, values, est, features, inputs, sensor
    ) -> dict:  # TODO: adapt to when not using features

        output = dict()

        try:
            tail_points = eval("self.config.FUSION_MODEL.n_tail_points_" + sensor)
        except AttributeError:
            tail_points = self.config.FUSION_MODEL.n_tail_points

        b, h, w = inputs.shape
        depth = inputs.view(b, h * w, 1)

        valid = depth != 0.0
        valid = valid.nonzero()[:, 1]

        valid_filter = inputs[0, :, :].cpu().detach().numpy()
        valid_filter = valid_filter != 0.0

        valid_filter = torch.tensor(valid_filter).unsqueeze(0)

        valid_filter = valid_filter.view(b, h * w, 1)

        valid_filter = valid_filter.nonzero()[:, 1]

        update_indices = values["indices"][:, valid, :tail_points, :, :]

        update_weights = values["weights"][:, valid, :tail_points, :]

        if self.config.FUSION_MODEL.n_empty_space_voting > 0:
            update_indices_empty = values["indices_empty"][:, valid, :, :, :]
            update_weights_empty = values["weights_empty"][:, valid, :, :]
            output["update_indices_empty"] = update_indices_empty
            output["update_weights_empty"] = update_weights_empty

        update_values = est[:, valid, :tail_points]

        update_values = torch.clamp(
            update_values, -self.config.DATA.trunc_value, self.config.DATA.trunc_value
        )

        update_features = features[:, valid, :tail_points, :]

        del valid

        # packing
        output["update_values"] = update_values
        output["update_features"] = update_features
        output["update_weights"] = update_weights
        output["update_indices"] = update_indices

        return output

    def fuse(self, batch, database, device):  # TODO: adapt to when not using features

        self.device = device
        # routing
        if self.config.ROUTING.do:
            if self.config.FILTERING_MODEL.model == "tsdf_early_fusion":
                depth, conf = self._routing(batch)

                frame = depth.squeeze_(1)
                confidence = None  # Need to implement this if I want to use it again
            else:
                # if batch['sensor'] == 'tof': # ONLY FOR A FUN TEST
                #     frame = batch[batch['sensor'] + '_depth'].squeeze_(1).to(device)
                #     confidence = None
                # else:
                depth, conf = self._routing(batch)
                frame = depth.squeeze_(1)
                confidence = conf.squeeze_(1)

        else:
            frame = batch["depth"].squeeze_(1)
            frame = frame.to(device)
            confidence = None

        if self.config.FEATURE_MODEL.w_rgb:
            rgb = batch["image"].squeeze().to(device)
        elif self.config.FEATURE_MODEL.w_intensity_gradient:
            i = batch["intensity"].squeeze()
            g = batch["gradient"].squeeze()
            rgb = torch.cat((i, g), dim=0).to(device)
        else:
            rgb = None

        if self.config.FEATURE_MODEL.stereo_warp_right:
            rgb_warp = batch["right_warped_rgb_stereo"].to(device)
        else:
            rgb_warp = None

        mask = batch["mask"].to(device)

        filtered_frame = torch.where(mask == 0, torch.zeros_like(frame), frame)

        # get current tsdf values
        scene_id = batch["frame_id"][0].split("/")[0]

        extracted_values = dict()

        try:
            intrinsics = batch["intrinsics" + "_" + batch["sensor"]]
        except KeyError:
            intrinsics = batch["intrinsics"]

        extracted_values[batch["sensor"]] = self._extractor[batch["sensor"]].forward(
            frame,
            batch["extrinsics"],
            intrinsics,
            database[scene_id]["tsdf" + "_" + batch["sensor"]],
            database[scene_id]["features_" + batch["sensor"]],
            database[scene_id]["origin"],
            database[scene_id]["resolution"],
            self.config.SETTINGS.gpu,
            database[scene_id]["weights" + "_" + batch["sensor"]],
        )

        try:
            n_points = eval("self.config.FUSION_MODEL.n_points_" + batch["sensor"])
        except AttributeError:
            n_points = self.config.FUSION_MODEL.n_points
        tsdf_input, feature_input = self._prepare_fusion_input(
            frame,
            extracted_values,
            batch["sensor"],
            confidence,
            n_points,
            rgb,
            rgb_warp,
        )
        del rgb, frame

        fusion_output = self._fusion(
            tsdf_input,
            feature_input,
            extracted_values[batch["sensor"]],
            batch["sensor"],
            batch["fusionNet"],
            extraction_band=n_points,
        )

        # masking invalid losses
        tsdf_est = fusion_output["tsdf_est"]
        feature_est = fusion_output["feature_est"]

        integrator_input = self._prepare_volume_update(
            extracted_values[batch["sensor"]],
            tsdf_est,
            feature_est,
            filtered_frame,
            batch["sensor"],
        )

        tsdf, features, weights, indices = self._integrator.forward(
            integrator_input,
            database[scene_id]["tsdf_" + batch["sensor"]].to(device),
            database[scene_id]["features_" + batch["sensor"]].to(device),
            database[scene_id]["weights_" + batch["sensor"]].to(device),
        )

        del indices, integrator_input

        database.tsdf[batch["sensor"]][scene_id].volume = tsdf.cpu().detach().numpy()
        database.fusion_weights[batch["sensor"]][scene_id] = (
            weights.cpu().detach().numpy()
        )
        database.features[batch["sensor"]][scene_id].volume = (
            features.cpu().detach().numpy()
        )

        del tsdf, weights, features

        return

    def fuse_training(
        self, batch, database, device
    ):  # TODO: adapt to when not using features

        """
        Learned real-time depth map fusion pipeline
        """
        output = dict()

        self.device = device

        # routing
        if self.config.ROUTING.do:
            if self.config.FILTERING_MODEL.model == "tsdf_early_fusion":
                depth, conf = self._routing(batch)

                frame = depth.squeeze_(1)
                confidence = None  # Need to implement this if I want to use it again
            else:
                # if batch['sensor'] == 'tof': # ONLY FOR A FUN TEST
                #     frame = batch[batch['sensor'] + '_depth'].squeeze_(1).to(device)
                #     confidence = None
                # else:
                depth, conf = self._routing(batch)
                frame = depth.squeeze_(1)
                confidence = conf  # Need to implement this if I want to use it again

        else:
            frame = batch["depth"].squeeze_(1)
            frame = frame.to(device)
            confidence = None

        if self.config.FEATURE_MODEL.w_rgb:
            rgb = batch["image"].squeeze().to(device)
        elif self.config.FEATURE_MODEL.w_intensity_gradient:
            i = batch["intensity"].squeeze()
            g = batch["gradient"].squeeze()
            rgb = torch.cat((i, g), dim=0).to(device)
        else:
            rgb = None

        if self.config.FEATURE_MODEL.stereo_warp_right:
            rgb_warp = batch["right_warped_rgb_stereo"].to(device)
        else:
            rgb_warp = None

        mask = batch["mask"].to(device)  # putting extractor on gpu

        filtered_frame = torch.where(mask == 0, torch.zeros_like(frame), frame)
        del mask

        b, h, w = frame.shape

        # get current tsdf values
        scene_id = batch["frame_id"][0].split("/")[0]

        extracted_values = dict()
        try:
            intrinsics = batch["intrinsics" + "_" + batch["sensor"]]
        except KeyError:
            intrinsics = batch["intrinsics"]

        extracted_values[batch["sensor"]] = self._extractor[batch["sensor"]].forward(
            frame,
            batch["extrinsics"],
            intrinsics,
            database[scene_id]["tsdf" + "_" + batch["sensor"]],
            database[scene_id]["features_" + batch["sensor"]],
            database[scene_id]["origin"],
            database[scene_id]["resolution"],
            self.config.SETTINGS.gpu,
            database[scene_id]["weights" + "_" + batch["sensor"]],
        )

        # TODO: make function that extracts only the gt values for speed up during training
        extracted_values_gt = self._extractor[batch["sensor"]].forward(
            frame,
            batch["extrinsics"],
            intrinsics,
            database[scene_id]["gt"],
            database[scene_id]["features_" + batch["sensor"]],
            database[scene_id]["origin"],
            database[scene_id]["resolution"],
            self.config.SETTINGS.gpu,
            database[scene_id]["weights_" + batch["sensor"]],
        )

        tsdf_target = extracted_values_gt["fusion_values"]
        del extracted_values_gt

        try:
            n_points = eval("self.config.FUSION_MODEL.n_points_" + batch["sensor"])
        except AttributeError:
            n_points = self.config.FUSION_MODEL.n_points
        tsdf_input, feature_input = self._prepare_fusion_input(
            frame,
            extracted_values,
            batch["sensor"],
            confidence,
            n_points,
            rgb,
            rgb_warp,
        )
        del rgb, frame

        fusion_output = self._fusion(
            tsdf_input,
            feature_input,
            extracted_values[batch["sensor"]],
            batch["sensor"],
            batch["fusionNet"],
            extraction_band=n_points,
        )

        del tsdf_input, feature_input

        # reshaping target
        tsdf_target = tsdf_target.view(b, h * w, n_points)

        # masking invalid losses
        tsdf_est = fusion_output["tsdf_est"]
        feature_est = fusion_output["feature_est"]
        if self.config.FUSION_MODEL.use_fusion_net:
            tsdf_fused = fusion_output["tsdf_fused"]
            tsdf_fused = masking(tsdf_fused, filtered_frame.view(b, h * w, 1))
            output["tsdf_fused"] = tsdf_fused
            del tsdf_fused

        del fusion_output

        tsdf_target = masking(tsdf_target, filtered_frame.view(b, h * w, 1))
        output["tsdf_target"] = tsdf_target
        del tsdf_target

        integrator_input = self._prepare_volume_update(
            extracted_values[batch["sensor"]],
            tsdf_est,
            feature_est,
            filtered_frame,
            batch["sensor"],
        )

        del extracted_values, tsdf_est, feature_est, filtered_frame

        tsdf, features, weights, indices = self._integrator.forward(
            integrator_input,
            database[scene_id]["tsdf_" + batch["sensor"]].to(device),
            database[scene_id]["features_" + batch["sensor"]].to(device),
            database[scene_id]["weights_" + batch["sensor"]].to(device),
        )

        del integrator_input

        database.tsdf[batch["sensor"]][scene_id].volume = tsdf.cpu().detach().numpy()
        database.fusion_weights[batch["sensor"]][scene_id] = (
            weights.cpu().detach().numpy()
        )
        database.features[batch["sensor"]][scene_id].volume = (
            features.cpu().detach().numpy()
        )

        output["tsdf"] = tsdf
        output["weights"] = weights
        output["features"] = features
        output["indices"] = indices

        del tsdf, weights, features

        return output


def masking(x, values, threshold=0.0, option="ueq"):

    if option == "leq":

        if x.dim() == 2:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == "geq":

        if x.dim() == 2:
            valid = (values >= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values >= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == "eq":

        if x.dim() == 2:
            valid = (values == threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values == threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == "ueq":

        if x.dim() == 2:
            valid = (values != threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values != threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    return xmasked
