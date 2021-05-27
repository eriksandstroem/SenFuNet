import torch
import numpy as np
from torch import nn

from torch.nn.functional import normalize


class EncoderBlock(nn.Module):
    '''Encoder block for the fusion network in NeuralFusion'''

    def __init__(self, c_in, c_out, activation, resolution, layernorm):

        super(EncoderBlock, self).__init__()

        if layernorm:
            self.block = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                                       nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                       activation,
                                       nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                                       nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                       activation)
        else:
            self.block = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                                       activation,
                                       nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                                       activation)

    def forward(self, x):
        # print('input enc: ', x.isnan().sum())
        # print(self.block[0].weight)
        # print('output enc: ', self.block(x).isnan().sum())
        return self.block(x)


class DecoderBlock(nn.Module):
    '''Decoder block for the fusion network in NeuralFusion'''

    def __init__(self, c_in, c_out, activation, resolution, layernorm):

        super(DecoderBlock, self).__init__()

        if layernorm:
            self.block = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                                       nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                       activation,
                                       nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                                       nn.LayerNorm([resolution[0], resolution[1]], elementwise_affine=True),
                                       activation)
        else:
            self.block = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), padding=1),
                                       activation,
                                       nn.Conv2d(c_out, c_out, (3, 3), padding=1),
                                       activation)

    def forward(self, x):
        return self.block(x)


class FeatureNet(nn.Module):
    """Fusion Network used in NeuralFusion"""

    def __init__(self, config, sensor):

        super(FeatureNet, self).__init__()

        try:
            self.n_points = eval('config.n_points_' + sensor)
        except:
            self.n_points = config.n_points


        self.n_features = config.n_features - config.append_depth
        if self.n_features == 0: # then we don't use the feature net at all, but to stop error we set it to one
            self.n_features = 1
        self.normalize = config.normalize
        self.w_rgb = config.w_rgb
        self.w_intensity_gradient = config.w_intensity_gradient


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

        n_channels_first = config.depth + 3*int(self.w_rgb) + 2*int(self.w_intensity_gradient)

        # add first encoder block
        self.encoder.append(EncoderBlock(n_channels_first,
                                         n_channels_input,
                                         enc_activation,
                                         resolution,
                                         layernorm))
        # add first decoder block
        self.decoder.append(DecoderBlock((self.n_layers) * n_channels_input + config.depth + 3*int(self.w_rgb) + 2*int(self.w_intensity_gradient),
                                         self.n_layers * n_channels_output,
                                         dec_activation,
                                         resolution,
                                         layernorm))

        # adding model layers
        for l in range(1, self.n_layers):
            self.encoder.append(EncoderBlock(n_channels_first + l * n_channels_input,
                                             n_channels_input,
                                             enc_activation,
                                             resolution,
                                             layernorm))

            self.decoder.append(DecoderBlock(((self.n_layers + 1) - l) * n_channels_output,
                                             ((self.n_layers + 1) - (l + 1)) * n_channels_output,
                                             dec_activation,
                                             resolution,
                                             layernorm))

        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.append_depth:
            d = x
        # encoding

        for enc in self.encoder:
            xmid = enc(x)
            # print(xmid)
            if xmid.isnan().sum() > 0 or xmid.isinf().sum() > 0:
                print('xmid nan: ', xmid.isnan().sum())
                print('xmid inf: ', xmid.isinf().sum())
            # print('enc: ', xmid.isnan().sum())
            x = torch.cat([x, xmid], dim=1)

        # print('enc isnan: ', x.isnan().sum())

        # decoding
        for dec in self.decoder:
            x = dec(x)

        # print('dec isnan: ', x.isnan().sum())

        if self.normalize:
            x = normalize(x, p=2, dim=1)

        # line below only for specific test
        # x = torch.zeros_like(x)

        if self.append_depth:
            x = torch.cat([x, d], dim=1)
  
        return x


class FeatureResNet(nn.Module):
    """Fusion Network used in NeuralFusion"""

    def __init__(self, config, sensor):

        super(FeatureResNet, self).__init__()

        try:
            self.n_points = eval('config.n_points_' + sensor)
        except:
            self.n_points = config.n_points


        self.n_features = config.n_features - config.append_depth
        if self.n_features == 0: # then we don't use the feature net at all, but to stop error we set it to one
            self.n_features = 1
        self.normalize = config.normalize
        self.w_rgb = config.w_rgb
        self.w_intensity_gradient = config.w_intensity_gradient


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

        n_channels_first = config.depth + 3*int(self.w_rgb) + 2*int(self.w_intensity_gradient)

        # add first encoder block
        self.encoder.append(EncoderBlock(n_channels_first,
                                         n_channels_input,
                                         enc_activation,
                                         resolution,
                                         layernorm))


        # adding model layers
        for l in range(1, self.n_layers):
            self.encoder.append(EncoderBlock(n_channels_input,
                                             n_channels_input,
                                             enc_activation,
                                             resolution,
                                             layernorm))


        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.append_depth:
            d = x
        # encoding

        for k, enc in enumerate(self.encoder):
            xmid = enc(x)
            # print(xmid)
            if xmid.isnan().sum() > 0 or xmid.isinf().sum() > 0:
                print('xmid nan: ', xmid.isnan().sum())
                print('xmid inf: ', xmid.isinf().sum())
            # print('enc: ', xmid.isnan().sum())
            if k > 0:
                x = x + xmid
            else:
                x = xmid


        if self.normalize:
            x = normalize(x, p=2, dim=1)




        if self.append_depth:
            x = torch.cat([x, d], dim=1)
  
        return x


