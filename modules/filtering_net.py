import torch

from torch import nn


class DoubleConv(nn.Module):
    """Double Convolution block for the filtering network"""

    def __init__(self, c_in, c_out, activation, bias=True):

        super(DoubleConv, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                c_in,
                int(c_out / 2),
                3,
                padding=1,
                padding_mode="replicate",
                bias=bias,
            ),
            eval(activation),
            nn.Conv3d(
                int(c_out / 2),
                c_out,
                3,
                padding=1,
                padding_mode="replicate",
                bias=bias,
            ),
            eval(activation),
        )

    def forward(self, x):
        return self.block(x)


class RefinementEncoder(nn.Module):
    def __init__(self, config):

        super(RefinementEncoder, self).__init__()

        self.tanh_weight = config.FILTERING_MODEL.CONV3D_MODEL.tanh_weight
        self.network_depth = (
            config.FILTERING_MODEL.CONV3D_MODEL.REFINEMENT.network_depth
        )
        self.activation = config.FILTERING_MODEL.CONV3D_MODEL.activation
        self.n_features = config.FEATURE_MODEL.n_features
        self.w_features = (
            config.FILTERING_MODEL.CONV3D_MODEL.REFINEMENT.features_to_sdf_enc
        )
        bias = config.FILTERING_MODEL.CONV3D_MODEL.REFINEMENT.bias

        # add encoder blocks
        if self.w_features:
            self.enc_1 = DoubleConv(
                2 + self.n_features,
                8,
                self.activation,
                bias,
            )
        else:
            self.enc_1 = DoubleConv(2, 8, self.activation, bias)

        self.enc_2 = DoubleConv(8, 16, self.activation, bias)

        self.dec_1 = DoubleConv(
            24,  # 16 if not using first residual connection or 24 if using the first residual connection
            16,
            self.activation,
            bias,
        )

        if self.network_depth > 1:
            # encoder block
            self.enc_3 = DoubleConv(16, 32, self.activation, bias)

            # decoder block
            self.dec_2 = DoubleConv(48, 16, self.activation, bias)

        if self.network_depth > 2:
            # encoder block
            self.enc_4 = DoubleConv(32, 64, self.activation, bias)

            # decoder block
            self.dec_3 = DoubleConv(96, 32, self.activation, bias)

        # max pooling layer
        self.mp = nn.MaxPool3d(2)
        # upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # tanh activation
        self.tanh = nn.Tanh()

    def forward(self, neighborhood):

        if self.tanh_weight:
            neighborhood[0, 1, :, :, :] = self.tanh(neighborhood[0, 1, :, :, :])

        if not self.w_features:
            neighborhood = neighborhood[:, :2, :, :, :]
        e1 = self.enc_1(neighborhood)
        x = self.mp(e1)
        # x = e1
        e2 = self.enc_2(x)

        if self.network_depth > 1:
            x = self.mp(e2)
            # x = e2
            e3 = self.enc_3(x)

        if self.network_depth > 2:
            x = self.mp(e3)
            # x = e3
            e4 = self.enc_4(x)

            x = self.up(e4)
            # x = e4
            e3 = torch.cat([x, e3], dim=1)
            e3 = self.dec_3(e3)

        if self.network_depth > 1:
            x = self.up(e3)
            # x = e3
            e2 = torch.cat([x, e2], dim=1)
            e2 = self.dec_2(e2)

        x = self.up(e2)
        # x = e2
        x = torch.cat([x, e1], dim=1)
        x = self.dec_1(x)

        return x


class Weighting_Encoder(nn.Module):
    def __init__(self, config):

        super(Weighting_Encoder, self).__init__()

        self.activation = config.FILTERING_MODEL.CONV3D_MODEL.activation
        self.n_features = config.FEATURE_MODEL.n_features
        self.features_to_weight_head = config.FILTERING_MODEL.features_to_weight_head
        self.network_depth = config.FILTERING_MODEL.CONV3D_MODEL.network_depth
        bias_wn = config.FILTERING_MODEL.CONV3D_MODEL.bias

        # add encoder blocks
        self.enc_1 = DoubleConv(
            self.n_features,
            8,
            self.activation,
            bias=bias_wn,
        )

        self.enc_2 = DoubleConv(
            8,
            16,
            self.activation,
            bias=bias_wn,
        )

        self.dec_1 = DoubleConv(
            24,  # 16 if not using first residual connection or 24 if using the first residual connection
            16,
            self.activation,
            bias=bias_wn,
        )

        if self.network_depth > 1:
            # encoder block
            self.enc_3 = DoubleConv(
                16,
                32,
                self.activation,
                bias=bias_wn,
            )

            # decoder block
            self.dec_2 = DoubleConv(
                48,
                16,
                self.activation,
                bias=bias_wn,
            )

        if self.network_depth > 2:
            # encoder block
            self.enc_4 = DoubleConv(
                32,
                64,
                self.activation,
                bias=bias_wn,
            )

            # decoder block
            self.dec_3 = DoubleConv(
                96,
                32,
                self.activation,
                bias=bias_wn,
            )

        # max pooling layer
        self.mp = nn.MaxPool3d(2)
        # upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # tanh activation
        self.tanh = nn.Tanh()

    def forward(self, neighborhood):

        e1 = self.enc_1(neighborhood)
        # print('e1', e1)
        x = self.mp(e1)
        # x = e1
        e2 = self.enc_2(x)

        if self.network_depth > 1:
            x = self.mp(e2)
            # x = e2
            e3 = self.enc_3(x)

        if self.network_depth > 2:
            x = self.mp(e3)
            # x = e3
            e4 = self.enc_4(x)

            x = self.up(e4)
            # x = e4
            e3 = torch.cat([x, e3], dim=1)
            e3 = self.dec_3(e3)

        if self.network_depth > 1:
            x = self.up(e3)
            # x = e3
            e2 = torch.cat([x, e2], dim=1)
            e2 = self.dec_2(e2)

        x = self.up(e2)
        # x = e2
        x = torch.cat([x, e1], dim=1)
        x = self.dec_1(x)

        return x


class Weighting_Decoder(nn.Module):
    def __init__(self, config):

        super(Weighting_Decoder, self).__init__()

        self.config = config
        self.sensors = config.DATA.input
        self.sdf_enc_to_weight_head = (
            config.FILTERING_MODEL.CONV3D_MODEL.REFINEMENT.sdf_enc_to_weight_head
        )
        self.activation = eval(config.FILTERING_MODEL.CONV3D_MODEL.activation)
        self.n_features = config.FEATURE_MODEL.n_features
        bias_wn = config.FILTERING_MODEL.CONV3D_MODEL.bias

        self.encoder = nn.ModuleDict()
        for sensor_ in config.DATA.input:
            self.encoder[sensor_] = Weighting_Encoder(config)

        self.decoder = nn.Sequential(
            nn.Conv3d(
                len(self.sensors) * 16 * self.sdf_enc_to_weight_head
                + len(self.sensors) * 16,
                32,
                1,
                padding=0,
                bias=bias_wn,
            ),
            self.activation,
            nn.Conv3d(32, 16, 1, padding=0, bias=bias_wn),
            self.activation,
            nn.Conv3d(16, 1, 1, padding=0, bias=bias_wn),
        )

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, neighborhood):
        # print(self.decoder[0].weight)
        sdf_enc_input = dict()
        feature_encoding = dict()

        for k, sensor_ in enumerate(self.sensors):
            start = k * (
                self.n_features + 16 * self.sdf_enc_to_weight_head
            )  # 16 because we use 16 features to decode the sdf for each sensor
            end = (k + 1) * (self.n_features + 16 * self.sdf_enc_to_weight_head)
            sensor_input = neighborhood[:, start:end, :, :, :]
            if self.sdf_enc_to_weight_head:
                sdf_enc_input[sensor_] = sensor_input[:, :16, :, :, :]
                sensor_input = sensor_input[:, 16:, :, :, :]

            feature_encoding[sensor_] = self.encoder[sensor_](sensor_input)

        input_ = None
        for k, sensor_ in enumerate(self.sensors):
            if k == 0:
                if self.sdf_enc_to_weight_head:
                    input_ = torch.cat(
                        (sdf_enc_input[sensor_], feature_encoding[sensor_]), dim=1
                    )
                else:
                    input_ = feature_encoding[sensor_]
            else:
                if self.sdf_enc_to_weight_head:
                    inp = torch.cat(
                        (sdf_enc_input[sensor_], feature_encoding[sensor_]), dim=1
                    )
                    input_ = torch.cat((input_, inp), dim=1)
                else:
                    input_ = torch.cat((input_, feature_encoding[sensor_]), dim=1)

        x = self.decoder(input_)

        return x


class FilteringNet(nn.Module):
    def __init__(self, config):

        super(FilteringNet, self).__init__()

        self.config = config
        self.output_scale = config.FILTERING_MODEL.CONV3D_MODEL.REFINEMENT.output_scale
        self.trunc_value = config.DATA.trunc_value
        self.sensors = config.DATA.input
        self.feature_to_weight_head = (
            config.FILTERING_MODEL.CONV3D_MODEL.features_to_weight_head
        )
        self.sdf_enc_to_weight_head = (
            config.FILTERING_MODEL.CONV3D_MODEL.REFINEMENT.sdf_enc_to_weight_head
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
        self.residual_learning = (
            config.FILTERING_MODEL.CONV3D_MODEL.REFINEMENT.residual_learning
        )
        self.use_refinement = config.FILTERING_MODEL.CONV3D_MODEL.use_refinement
        self.alpha_supervision = config.LOSS.alpha_supervision
        self.alpha_single_sensor_supervision = (
            config.LOSS.alpha_single_sensor_supervision
        )
        bias = config.FILTERING_MODEL.CONV3D_MODEL.REFINEMENT.bias
        bias_wn = config.FILTERING_MODEL.CONV3D_MODEL.bias
        self.refinement_model = (
            config.FILTERING_MODEL.CONV3D_MODEL.REFINEMENT.refinement_model
        )
        self.outlier_channel = config.FILTERING_MODEL.CONV3D_MODEL.outlier_channel

        if self.use_refinement:
            self.encoder = nn.ModuleDict()
            self.sdf_layer = nn.ModuleDict()
            for sensor_ in config.DATA.input:
                if self.refinement_model == "simple":
                    self.sdf_layer[sensor_] = nn.Conv3d(
                        1, 1, 3, padding=1, padding_mode="replicate", bias=bias
                    )
                else:
                    self.encoder[sensor_] = RefinementEncoder(config)
                    self.sdf_layer[sensor_] = nn.Conv3d(16, 1, 1, padding=0, bias=bias)

        # alpha layer
        if self.weighting_complexity == "1layer":
            self.weight_decoder = nn.Conv3d(
                16 * len(self.sensors) * self.sdf_enc_to_weight_head
                + len(self.sensors)
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
                    16 * len(self.sensors) * self.sdf_enc_to_weight_head
                    + len(self.sensors)
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
                    16 * len(self.sensors) * self.sdf_enc_to_weight_head
                    + len(self.sensors)
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
                    16 * len(self.sensors) * self.sdf_enc_to_weight_head
                    + len(self.sensors)
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
                    16 * len(self.sensors) * self.sdf_enc_to_weight_head
                    + len(self.sensors)
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
        elif self.weighting_complexity == "unet_style":
            self.weight_decoder = Weighting_Decoder(config)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, neighborhood):
        weight = dict()
        sdf = dict()
        enc = dict()
        output = dict()

        for sensor_ in self.sensors:
            if self.use_refinement:
                if self.refinement_model == "simple":
                    sdf[sensor_] = self.output_scale * self.tanh(
                        self.sdf_layer[sensor_](
                            neighborhood[sensor_][:, 0, :, :, :].unsqueeze(1)
                        )
                    )
                else:
                    enc[sensor_] = self.encoder[sensor_](neighborhood[sensor_])
                    sdf[sensor_] = self.output_scale * self.tanh(
                        self.sdf_layer[sensor_](enc[sensor_])
                    )

                sdf[sensor_] = torch.clamp(
                    sdf[sensor_], -self.trunc_value, self.trunc_value
                )
                if sdf[sensor_].isnan().sum() > 0 or sdf[sensor_].isinf().sum() > 0:
                    print(sensor_)
                    print("sdf[sensor_] nan: ", sdf[sensor_].isnan().sum())
                    print("sdf[sensor_] inf: ", sdf[sensor_].isinf().sum())
                if self.residual_learning:
                    sdf[sensor_] += neighborhood[sensor_][:, 0, :, :, :]
                    sdf[sensor_] = torch.clamp(
                        sdf[sensor_], -self.trunc_value, self.trunc_value
                    )
            else:
                sdf[sensor_] = neighborhood[sensor_][:, 0, :, :, :]

            weight[sensor_] = neighborhood[sensor_][:, 1, :, :, :].unsqueeze(1)

        for sensor_ in self.sensors:
            output["tsdf_" + sensor_] = sdf[sensor_].squeeze()
            output[sensor_ + "_init"] = weight[sensor_].squeeze() > 0

        input_ = None
        alpha_val = dict()

        for k, sensor_ in enumerate(self.config.DATA.input):
            inp = None
            if self.sdf_enc_to_weight_head:
                inp = enc[sensor_]
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
                    inp = torch.cat(
                        (inp, neighborhood[sensor_][:, 2:, :, :, :]), dim=1
                    )
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
            alpha_sdf = alpha[
                :, 0, :, :, :
            ] 
        else:
            alpha_sdf = alpha

        # this step is to not filter the voxels where we only have one sensor observing
        # note that we save the variable alpha and not alpha_sdf so we can still
        # use the outlier filter as usual. During test time this step is important to avoid
        # that we make a smooth decision where only one sensor integrates.
        for sensor_ in self.config.DATA.input:
            alpha_sdf = torch.where(
                weight[sensor_] == 0, alpha_val[sensor_], alpha_sdf
            )

        sdf_final = None

        for k, sensor_ in enumerate(self.config.DATA.input):
            if k == 0:
                sdf_final = alpha_sdf * sdf[sensor_]
            else:
                sdf_final += (1 - alpha_sdf) * sdf[sensor_]

        output["tsdf"] = sdf_final.squeeze()

        return output
