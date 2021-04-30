import torch

from torch import nn

class DoubleConv(nn.Module):
    '''Double Convolution block for the filtering network'''

    def __init__(self, c_in, c_out, activation, grouping_strategy, nbr_groups):

        super(DoubleConv, self).__init__()

        if grouping_strategy == 'relative':
            self.block = nn.Sequential(nn.GroupNorm(num_groups=int(c_in/2), num_channels=c_in),
                                       nn.Conv3d(c_in, int(c_out/2), 3, padding=1, padding_mode='replicate', groups=int(c_in/2)),
                                       eval(activation),
                                       nn.GroupNorm(num_groups=int(c_out/4), num_channels=int(c_out/2)),
                                       nn.Conv3d(int(c_out/2), c_out, 3, padding=1, padding_mode='replicate', groups=int(c_out/4)),
                                       eval(activation))
        elif grouping_strategy == 'absolute':
            self.block = nn.Sequential(#nn.GroupNorm(num_groups=nbr_groups, num_channels=c_in),
                                       nn.Conv3d(c_in, int(c_out/2), 3, padding=1, padding_mode='replicate', bias=True),
                                       eval(activation),
                                       # nn.GroupNorm(num_groups=nbr_groups, num_channels=int(c_out/2)),
                                       nn.Conv3d(int(c_out/2), c_out, 3, padding=1, padding_mode='replicate', bias=True),
                                       eval(activation))
            # self.block = nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, padding_mode='replicate', bias=True),
            #                            eval(activation))

    def forward(self, x):
        return self.block(x)

class FilteringNetEncoder(nn.Module):
    def __init__(self, config):

        super(FilteringNetEncoder, self).__init__()

        self.tanh_weight = config.FILTERING_MODEL.tanh_weight
        self.network_depth = config.FILTERING_MODEL.network_depth
        self.grouping_strategy = config.FILTERING_MODEL.grouping_strategy
        self.nbr_groups = config.FILTERING_MODEL.nbr_groups
        self.activation = config.FILTERING_MODEL.activation
        self.n_features = config.FEATURE_MODEL.n_features
        self.w_features = config.FILTERING_MODEL.w_features

        # add encoder blocks
        if self.w_features:
            self.enc_1 = DoubleConv(2 + self.n_features,
                                8,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)
        else:
            self.enc_1 = DoubleConv(2,
                                8,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

        self.enc_2 = DoubleConv(8,
                            16,
                            self.activation,
                            self.grouping_strategy,
                            self.nbr_groups)

        self.dec_1 = DoubleConv(24, # 16 if not using first residual connection or 24 if using the first residual connection
                            16,
                            self.activation,
                            self.grouping_strategy,
                            self.nbr_groups)

        if self.network_depth > 1:
            # encoder block
            self.enc_3 = DoubleConv(16,
                                32,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

            # decoder block
            self.dec_2 = DoubleConv(48,
                                16,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

        if self.network_depth > 2:
            # encoder block
            self.enc_4 = DoubleConv(32,
                                64,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

            # decoder block
            self.dec_3 = DoubleConv(96,
                                32,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

        # max pooling layer
        self.mp = nn.MaxPool3d(2)
        # upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # tanh activation
        self.tanh = nn.Tanh()

    def forward(self, neighborhood):
        # stereo_tsdf = neighborhood['stereo']
        # print(tof_tsdf.shape)
        if self.tanh_weight:
            neighborhood[0, 1, :, :, :] = self.tanh(neighborhood[0, 1, :, :, :])

        if not self.w_features:
            neighborhood = neighborhood[:, :2, :, :, :]

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

class FilteringNet(nn.Module):

    def __init__(self, config):

        super(FilteringNet, self).__init__()

        self.tanh_weight = config.FILTERING_MODEL.tanh_weight
        self.network_depth = config.FILTERING_MODEL.network_depth
        self.grouping_strategy = config.FILTERING_MODEL.grouping_strategy
        self.nbr_groups = config.FILTERING_MODEL.nbr_groups
        self.activation = config.FILTERING_MODEL.activation
        self.n_features = config.FEATURE_MODEL.n_features
        self.w_features = config.FILTERING_MODEL.w_features
        self.output_scale = config.FILTERING_MODEL.output_scale
        self.trunc_value = config.DATA.trunc_value

        # add encoder blocks
        if self.w_features:
            self.enc_1 = DoubleConv(2 + self.n_features,
                                8,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)
        else:
            self.enc_1 = DoubleConv(2,
                                8,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

        self.enc_2 = DoubleConv(8,
                            16,
                            self.activation,
                            self.grouping_strategy,
                            self.nbr_groups)

        self.dec_1 = DoubleConv(24, # 16 if not using first residual connection or 24 if using the first residual connection
                            16,
                            self.activation,
                            self.grouping_strategy,
                            self.nbr_groups)

        if self.network_depth > 1:
            # encoder block
            self.enc_3 = DoubleConv(16,
                                32,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

            # decoder block
            self.dec_2 = DoubleConv(48,
                                16,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

        if self.network_depth > 2:
            # encoder block
            self.enc_4 = DoubleConv(32,
                                64,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

            # decoder block
            self.dec_3 = DoubleConv(96,
                                32,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups)

        # max pooling layer
        self.mp = nn.MaxPool3d(2)
        # upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # tanh activation
        self.tanh = nn.Tanh()

        # final decoding layer
        self.sdf_layer = nn.Conv3d(16, 1, 1, padding=0, bias=True)

    def forward(self, neighborhood):

        neighborhood = neighborhood['tof']

                # stereo_tsdf = neighborhood['stereo']
        # print(tof_tsdf.shape)
        if self.tanh_weight:
            neighborhood[0, 1, :, :, :] = self.tanh(neighborhood[0, 1, :, :, :])

        if not self.w_features:
            neighborhood = neighborhood[:, :2, :, :, :]

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

        x = self.output_scale*self.sdf_layer(x)

        x = torch.clamp(x,
                        -self.trunc_value,
                        self.trunc_value)

        # x = neighborhood['tof'][:, :, :, 0]
        output = dict()
        output['tsdf'] = x.squeeze()

        return output


class FilteringNetFuseSensors(nn.Module):

    def __init__(self, config):

        super(FilteringNetFuseSensors, self).__init__()

        self.output_scale = config.FILTERING_MODEL.output_scale
        self.trunc_value = config.DATA.trunc_value
   
        self.encoder_tof = FilteringNetEncoder(config)
        self.encoder_stereo = FilteringNetEncoder(config)

        # final decoding layer
        self.sdf_layer_tof = nn.Conv3d(16, 1, 1, padding=0, bias=True)
        self.sdf_layer_stereo = nn.Conv3d(16, 1, 1, padding=0, bias=True)

        # alpha layer
        self.weight_decoder = nn.Conv3d(32, 1, 1, padding=0, bias=True)

    def forward(self, neighborhood):
        tof_weight = neighborhood['tof'][:, 1, :, :, :].unsqueeze(1)
        stereo_weight = neighborhood['stereo'][:, 1, :, :, :].unsqueeze(1)

        tof_enc = self.encoder_tof(neighborhood['tof'])
        stereo_enc = self.encoder_tof(neighborhood['stereo'])

        del neighborhood
        # compute refined per sensor sdf values
        tof_sdf = self.output_scale*self.sdf_layer_tof(tof_enc)
        stereo_sdf = self.output_scale*self.sdf_layer_stereo(stereo_enc)

        tof_sdf = torch.clamp(tof_sdf,
                               -self.trunc_value,
                               self.trunc_value)

        stereo_sdf = torch.clamp(stereo_sdf,
                               -self.trunc_value,
                               self.trunc_value)

        # compute weighting of sensors using the per sensor encoding
        alpha = self.weight_decoder(torch.cat((tof_enc, stereo_enc), dim=1))
        alpha = torch.where(stereo_weight == 0, torch.ones_like(alpha), alpha)
        alpha = torch.where(tof_weight == 0, torch.zeros_like(alpha), alpha)

        sdf = alpha * tof_sdf + (1 - alpha) * stereo_sdf
   
        # this line should not be needed
        sdf = torch.clamp(sdf,
                            -self.trunc_value,
                            self.trunc_value)

        output = dict()
        output['tsdf'] = sdf.squeeze()
        output['tof_init'] = tof_weight.squeeze() > 0
        output['stereo_init'] = stereo_weight.squeeze() > 0
        output['tsdf_tof'] = tof_sdf.squeeze()
        output['tsdf_stereo'] = stereo_sdf.squeeze()

        return output

