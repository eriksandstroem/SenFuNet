import torch
import numpy as np
from torch import nn

from torch.nn.functional import normalize


class EncoderBlock(nn.Module):
    '''Encoder block for the fusion network in NeuralFusion'''

    def __init__(self, c_in, c_out, activation, resolution):

        super(EncoderBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                                   nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                   activation,
                                   nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                                   nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                   activation)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    '''Decoder block for the fusion network in NeuralFusion'''

    def __init__(self, c_in, c_out, activation, resolution):

        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                                   nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                   activation,
                                   nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                                   nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                   activation)

    def forward(self, x):
        return self.block(x)


class FeatureNet(nn.Module):
    """Fusion Network used in NeuralFusion"""

    def __init__(self, config, sensor):

        super(FeatureNet, self).__init__()

        self.n_points = eval('config.n_points_' + sensor)
        self.n_features = config.n_features

        # layer settings
        n_channels_input = self.n_points * (self.n_features + int(config.use_count))
        n_channels_output = self.n_points * self.n_features
        self.n_layers = config.n_layers
        self.height = config.resy
        self.width = config.resx
        resolution = (self.height, self.width)
        activation = eval(config.activation)
        self.tsdf_out = self.n_points
        self.scale = config.output_scale

        # define network submodules (encoder/decoder)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        n_channels_first = n_channels_input + config.depth

        # add first encoder block
        self.encoder.append(EncoderBlock(n_channels_first,
                                         n_channels_input,
                                         activation,
                                         resolution))
        # add first decoder block
        self.decoder.append(DecoderBlock((self.n_layers + 1) * n_channels_input + config.depth,
                                         self.n_layers * n_channels_output,
                                         activation,
                                         resolution))

        # adding model layers
        for l in range(1, self.n_layers):
            self.encoder.append(EncoderBlock(n_channels_first + l * n_channels_input,
                                             n_channels_input,
                                             activation,
                                             resolution))

            self.decoder.append(DecoderBlock(((self.n_layers + 1) - l) * n_channels_output,
                                             ((self.n_layers + 1) - (l + 1)) * n_channels_output,
                                             activation,
                                             resolution))

        self.tanh = nn.Tanh()

    def forward(self, x):

        # encoding
        for enc in self.encoder:
            xmid = enc(x)
            x = torch.cat([x, xmid], dim=1)

        # decoding
        for dec in self.decoder:
            x = dec(x)

        # normalization
        x = x.view(1, self.n_features, self.n_points, self.height, self.width)
        x = normalize(x, p=2, dim=1)
        x = x.view(1, self.n_features * self.n_points, self.height, self.width)

        return x


