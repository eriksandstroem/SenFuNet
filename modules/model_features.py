import torch
from torch import nn

from torch.nn.functional import normalize


class EncoderBlock(nn.Module):
    """Encoder block for the fusion network in NeuralFusion"""

    def __init__(self, c_in, c_out, activation, resolution, layernorm):

        super(EncoderBlock, self).__init__()

        if layernorm:
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                activation,
                nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                activation,
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                activation,
                nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                activation,
            )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder block for the fusion network in NeuralFusion"""

    def __init__(self, c_in, c_out, activation, resolution, layernorm):

        super(DecoderBlock, self).__init__()

        if layernorm:
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                activation,
                nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                activation,
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                activation,
                nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                activation,
            )

    def forward(self, x):
        return self.block(x)


class FeatureNet(nn.Module):
    """Network used in NeuralFusion"""

    def __init__(self, config, sensor):

        super(FeatureNet, self).__init__()

        try:
            self.n_points = eval("config.n_points_" + sensor)
        except AttributeError:
            self.n_points = config.n_points

        self.n_features = config.n_features - config.append_depth

        self.normalize = config.normalize
        self.w_rgb = config.w_rgb
        self.w_stereo_warp_right = config.stereo_warp_right
        self.w_intensity_gradient = config.w_intensity_gradient
        self.confidence = config.confidence

        # layer settings
        n_channels_input = self.n_features
        n_channels_output = self.n_features
        self.n_layers = config.n_layers
        self.height = config.resy
        self.width = config.resx
        resolution = (self.height, self.width)
        enc_activation = eval(config.enc_activation)
        dec_activation = eval(config.dec_activation)
        self.tsdf_out = self.n_points
        layernorm = config.layernorm
        self.append_depth = config.append_depth

        # define network submodules (encoder/decoder)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if sensor == "tof":
            n_channels_first = (
                config.depth
                + 3 * int(self.w_rgb) * config.w_rgb_tof
                + 2 * int(self.w_intensity_gradient)
                + int(self.confidence)
            )
        elif sensor == "stereo":
            n_channels_first = (
                config.depth
                + 3 * int(self.w_rgb)
                + 2 * int(self.w_intensity_gradient)
                + 3 * int(self.w_stereo_warp_right)
                + int(self.confidence)
            )
        else:
            n_channels_first = (
                config.depth
                + 3 * int(self.w_rgb)
                + 2 * int(self.w_intensity_gradient)
                + int(self.confidence)
            )

        # add first encoder block
        self.encoder.append(
            EncoderBlock(
                n_channels_first,
                n_channels_input,
                enc_activation,
                resolution,
                layernorm,
            )
        )
        # add first decoder block
        if sensor == "stereo":
            self.decoder.append(
                DecoderBlock(
                    (self.n_layers) * n_channels_input
                    + config.depth
                    + 3 * int(self.w_rgb)
                    + 2 * int(self.w_intensity_gradient)
                    + 3 * int(self.w_stereo_warp_right)
                    + int(self.confidence),
                    self.n_layers * n_channels_output,
                    dec_activation,
                    resolution,
                    layernorm,
                )
            )
        elif sensor == "tof":
            self.decoder.append(
                DecoderBlock(
                    (self.n_layers) * n_channels_input
                    + config.depth
                    + 3 * int(self.w_rgb) * config.w_rgb_tof
                    + 2 * int(self.w_intensity_gradient)
                    + int(self.confidence),
                    self.n_layers * n_channels_output,
                    dec_activation,
                    resolution,
                    layernorm,
                )
            )
        else:
            self.decoder.append(
                DecoderBlock(
                    (self.n_layers) * n_channels_input
                    + config.depth
                    + 3 * int(self.w_rgb)
                    + 2 * int(self.w_intensity_gradient)
                    + int(self.confidence),
                    self.n_layers * n_channels_output,
                    dec_activation,
                    resolution,
                    layernorm,
                )
            )

        # adding model layers
        for l in range(1, self.n_layers):
            self.encoder.append(
                EncoderBlock(
                    n_channels_first + l * n_channels_input,
                    n_channels_input,
                    enc_activation,
                    resolution,
                    layernorm,
                )
            )

            self.decoder.append(
                DecoderBlock(
                    ((self.n_layers + 1) - l) * n_channels_output,
                    ((self.n_layers + 1) - (l + 1)) * n_channels_output,
                    dec_activation,
                    resolution,
                    layernorm,
                )
            )

        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.append_depth:
            if self.w_rgb:
                d = x[:, 0, :, :].unsqueeze(1)
            else:
                d = x

        # encoding

        for enc in self.encoder:
            xmid = enc(x)
            if xmid.isnan().sum() > 0 or xmid.isinf().sum() > 0:
                print("xmid nan: ", xmid.isnan().sum())
                print("xmid inf: ", xmid.isinf().sum())
            x = torch.cat([x, xmid], dim=1)

        # decoding
        for dec in self.decoder:
            x = dec(x)

        if self.normalize:
            x = normalize(x, p=2, dim=1)

        if self.append_depth:
            x = torch.cat([x, d], dim=1)

        output = dict()

        output["feature"] = x

        return output


class FeatureResNet(nn.Module):
    """Residual Network"""

    def __init__(self, config, sensor):

        super(FeatureResNet, self).__init__()

        try:
            self.n_points = eval("config.n_points_" + sensor)
        except AttributeError:
            self.n_points = config.n_points

        self.n_features = config.n_features - config.append_depth

        self.normalize = config.normalize
        self.w_rgb = config.w_rgb
        self.w_stereo_warp_right = config.stereo_warp_right
        self.w_intensity_gradient = config.w_intensity_gradient
        self.confidence = config.confidence

        # layer settings
        n_channels_input = self.n_features
        self.n_layers = config.n_layers
        self.height = config.resy
        self.width = config.resx
        resolution = (self.height, self.width)
        enc_activation = eval(config.enc_activation)
        self.tsdf_out = self.n_points
        layernorm = config.layernorm
        self.append_depth = config.append_depth

        # define network submodules (encoder/decoder)
        self.encoder = nn.ModuleList()

        if sensor == "tof":
            n_channels_first = (
                config.depth
                + 3 * int(self.w_rgb) * config.w_rgb_tof
                + 2 * int(self.w_intensity_gradient)
                + int(self.confidence)
            )
        elif (
            sensor == "stereo"
        ):  # I did not feed rgb to sgm_stereo. This line should have been sensor.endswith("stereo"):
            n_channels_first = (
                config.depth
                + 3 * int(self.w_rgb)
                + 2 * int(self.w_intensity_gradient)
                + 3 * int(self.w_stereo_warp_right)
                + int(self.confidence)
            )
        else:
            n_channels_first = (
                config.depth
                + 3 * int(self.w_rgb)
                + 2 * int(self.w_intensity_gradient)
                + int(self.confidence)
            )

        # add first encoder block
        self.encoder.append(
            EncoderBlock(
                n_channels_first,
                n_channels_input,
                enc_activation,
                resolution,
                layernorm,
            )
        )

        # adding model layers
        for l in range(1, self.n_layers):
            self.encoder.append(
                EncoderBlock(
                    n_channels_input,
                    n_channels_input,
                    enc_activation,
                    resolution,
                    layernorm,
                )
            )

        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.append_depth:
            if self.w_rgb:
                d = x[:, 0, :, :].unsqueeze(1)
            else:
                d = x

        # encoding

        for k, enc in enumerate(self.encoder):
            xmid = enc(x)
            if xmid.isnan().sum() > 0 or xmid.isinf().sum() > 0:
                print("xmid nan: ", xmid.isnan().sum())
                print("xmid inf: ", xmid.isinf().sum())

            if k > 0:
                x = x + xmid
            else:
                x = xmid

        if self.normalize:
            x = normalize(x, p=2, dim=1)

        if self.append_depth:
            x = torch.cat([x, d], dim=1)

        output = dict()

        output["feature"] = x

        return output
