import torch

from torch import nn


class FilteringNet(nn.Module):
    def __init__(self, config):

        super(FilteringNet, self).__init__()

        self.config = config
        self.trunc_value = config.DATA.trunc_value
        self.sensors = config.DATA.input
        self.feature_to_weight_head = (
            config.FILTERING_MODEL.CONV3D_MODEL.features_to_weight_head
        )
        self.weight_to_weight_head = (
            config.FILTERING_MODEL.CONV3D_MODEL.weights_to_weight_head
        )
        self.sdf_to_weight_head = config.FILTERING_MODEL.CONV3D_MODEL.sdf_to_weight_head
        self.weighting_complexity = (
            config.FILTERING_MODEL.CONV3D_MODEL.weighting_complexity
        )
        self.activation = eval(config.FILTERING_MODEL.CONV3D_MODEL.activation)
        self.n_features = config.FEATURE_MODEL.n_features
        self.alpha_supervision = config.LOSS.alpha_supervision
        self.alpha_single_sensor_supervision = (
            config.LOSS.alpha_single_sensor_supervision
        )
        bias_wn = config.FILTERING_MODEL.CONV3D_MODEL.bias
        self.outlier_channel = config.FILTERING_MODEL.CONV3D_MODEL.outlier_channel

        # alpha layer
        if self.weighting_complexity == "1layer":
            self.weight_decoder = nn.Conv3d(
                len(self.sensors)
                * (
                    self.sdf_to_weight_head
                    + self.n_features * self.feature_to_weight_head
                    + self.weight_to_weight_head
                ),
                1,
                1,
                padding=0,
                bias=bias_wn,
            )
        elif self.weighting_complexity == "2layer":
            self.weight_decoder = nn.Sequential(
                nn.Conv3d(
                    len(self.sensors)
                    * (
                        self.sdf_to_weight_head
                        + self.n_features * self.feature_to_weight_head
                        + self.weight_to_weight_head
                    ),
                    16,
                    3,
                    padding=1,
                    padding_mode="replicate",
                    bias=bias_wn,
                ),
                self.activation,
                nn.Conv3d(16, 1 + self.outlier_channel, 1, padding=0, bias=bias_wn),
            )
        elif self.weighting_complexity == "3layer":
            self.weight_decoder = nn.Sequential(
                nn.Conv3d(
                    len(self.sensors)
                    * (
                        self.sdf_to_weight_head
                        + self.n_features * self.feature_to_weight_head
                        + self.weight_to_weight_head
                    ),
                    32,
                    3,
                    padding=1,
                    padding_mode="replicate",
                    bias=bias_wn,
                ),
                self.activation,
                nn.Conv3d(32, 16, 3, padding=1, padding_mode="replicate", bias=bias_wn),
                self.activation,
                nn.Conv3d(16, 1, 1, padding=0, bias=bias_wn),
            )

        elif self.weighting_complexity == "4layer":
            self.weight_decoder = nn.Sequential(
                nn.Conv3d(
                    len(self.sensors)
                    * (
                        self.sdf_to_weight_head
                        + self.n_features * self.feature_to_weight_head
                        + self.weight_to_weight_head
                    ),
                    32,
                    3,
                    padding=1,
                    padding_mode="replicate",
                    bias=bias_wn,
                ),
                self.activation,
                nn.Conv3d(32, 32, 3, padding=1, padding_mode="replicate", bias=bias_wn),
                self.activation,
                nn.Conv3d(32, 16, 3, padding=1, padding_mode="replicate", bias=bias_wn),
                self.activation,
                nn.Conv3d(16, 1 + self.outlier_channel, 1, padding=0, bias=bias_wn),
            )
        elif self.weighting_complexity == "5layer":
            self.weight_decoder = nn.Sequential(
                nn.Conv3d(
                    len(self.sensors)
                    * (
                        self.sdf_to_weight_head
                        + self.n_features * self.feature_to_weight_head
                        + self.weight_to_weight_head
                    ),
                    32,
                    3,
                    padding=1,
                    padding_mode="replicate",
                    bias=bias_wn,
                ),
                self.activation,
                nn.Conv3d(32, 32, 3, padding=1, padding_mode="replicate", bias=bias_wn),
                self.activation,
                nn.Conv3d(32, 32, 3, padding=1, padding_mode="replicate", bias=bias_wn),
                self.activation,
                nn.Conv3d(32, 16, 3, padding=1, padding_mode="replicate", bias=bias_wn),
                self.activation,
                nn.Conv3d(16, 1 + self.outlier_channel, 1, padding=0, bias=bias_wn),
            )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, neighborhood):
        weight = dict()
        sdf = dict()
        enc = dict()
        output = dict()

        for sensor_ in self.sensors:
            sdf[sensor_] = neighborhood[sensor_][:, 0, :, :, :]

            weight[sensor_] = neighborhood[sensor_][:, 1, :, :, :].unsqueeze(1)

        for sensor_ in self.sensors:
            output["tsdf_" + sensor_] = sdf[sensor_].squeeze()
            output[sensor_ + "_init"] = weight[sensor_].squeeze() > 0

        input_ = None
        alpha_val = dict()

        for k, sensor_ in enumerate(self.config.DATA.input):
            inp = None
            if self.sdf_to_weight_head:
                if inp is None:
                    inp = neighborhood[sensor_][:, 0, :, :, :].unsqueeze(1)
                else:
                    inp = torch.cat(
                        (inp, neighborhood[sensor_][:, 0, :, :, :].unsqueeze(1)),
                        dim=1,
                    )
            if self.feature_to_weight_head:
                if inp is None:
                    inp = neighborhood[sensor_][:, 2:, :, :, :]
                else:
                    inp = torch.cat((inp, neighborhood[sensor_][:, 2:, :, :, :]), dim=1)
            if self.weight_to_weight_head:
                if self.config.FILTERING_MODEL.CONV3D_MODEL.tanh_weight:
                    if self.config.FILTERING_MODEL.CONV3D_MODEL.inverted_weight:
                        weights = torch.ones_like(
                            neighborhood[sensor_][:, 1, :, :, :].unsqueeze(1)
                        ) - self.tanh(neighborhood[sensor_][:, 1, :, :, :])
                    else:
                        weights = self.tanh(
                            neighborhood[sensor_][:, 1, :, :, :]
                        ).unsqueeze(1)
                else:
                    weights = neighborhood[sensor_][:, 1, :, :, :].unsqueeze(1)

                if inp is None:
                    inp = weights
                else:
                    inp = torch.cat((inp, weights), dim=1)

            if input_ is None:
                input_ = inp
            else:
                input_ = torch.cat((input_, inp), dim=1)

            if k == 0:
                alpha_val[sensor_] = torch.zeros_like(sdf[sensor_])
            else:
                alpha_val[sensor_] = torch.ones_like(sdf[sensor_])

        if input_.isnan().sum() > 0:
            print("Input isnan: ", input_.isnan().sum())

        alpha = self.sigmoid(self.weight_decoder(input_))

        if alpha.isnan().sum() > 0 or alpha.isinf().sum() > 0:
            print("alpha nan: ", alpha.isnan().sum())
            print("alpha inf: ", alpha.isinf().sum())
            return None

        if (
            neighborhood["test_mode"]
            or self.alpha_supervision
            or self.alpha_single_sensor_supervision
        ):
            output["sensor_weighting"] = alpha.squeeze()

        if self.outlier_channel:
            alpha_sdf = alpha[:, 0, :, :, :]
        else:
            alpha_sdf = alpha

        # this step is to not filter the voxels where we only have one sensor observation.
        # Note that we save the variable alpha and not alpha_sdf so we can still
        # use the outlier filter as usual.
        for sensor_ in self.config.DATA.input:
            alpha_sdf = torch.where(weight[sensor_] == 0, alpha_val[sensor_], alpha_sdf)

        sdf_final = None

        for k, sensor_ in enumerate(self.config.DATA.input):
            if k == 0:
                sdf_final = alpha_sdf * sdf[sensor_]
            else:
                sdf_final += (1 - alpha_sdf) * sdf[sensor_]

        output["tsdf"] = sdf_final.squeeze()

        return output
