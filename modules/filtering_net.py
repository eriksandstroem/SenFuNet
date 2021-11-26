import torch

from torch import nn

class DoubleConv(nn.Module):
    '''Double Convolution block for the filtering network'''

    def __init__(self, c_in, c_out, activation, grouping_strategy, nbr_groups, bias=True):

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
                                       nn.Conv3d(c_in, int(c_out/2), 3, padding=1, padding_mode='replicate', bias=bias),
                                       eval(activation),
                                       # nn.GroupNorm(num_groups=nbr_groups, num_channels=int(c_out/2)),
                                       nn.Conv3d(int(c_out/2), c_out, 3, padding=1, padding_mode='replicate', bias=bias),
                                       eval(activation))
            # self.block = nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1, padding_mode='replicate', bias=True),
            #                            eval(activation))

    def forward(self, x):
        return self.block(x)

class FilteringNetEncoder(nn.Module):
    def __init__(self, config):

        super(FilteringNetEncoder, self).__init__()

        self.tanh_weight = config.FILTERING_MODEL.tanh_weight
        self.network_depth = config.FILTERING_MODEL.CONV3D_MODEL.network_depth
        self.grouping_strategy = config.FILTERING_MODEL.CONV3D_MODEL.grouping_strategy
        self.nbr_groups = config.FILTERING_MODEL.CONV3D_MODEL.nbr_groups
        self.activation = config.FILTERING_MODEL.CONV3D_MODEL.activation
        self.n_features = config.FEATURE_MODEL.n_features
        self.w_features = config.FILTERING_MODEL.features_to_sdf_enc
        bias = config.FILTERING_MODEL.CONV3D_MODEL.bias

        # add encoder blocks
        if self.w_features:
            self.enc_1 = DoubleConv(2 + self.n_features,
                                8,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups,
                                bias)
        else:
            self.enc_1 = DoubleConv(2,
                                8,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups,
                                bias)

        self.enc_2 = DoubleConv(8,
                            16,
                            self.activation,
                            self.grouping_strategy,
                            self.nbr_groups,
                            bias)

        self.dec_1 = DoubleConv(24, # 16 if not using first residual connection or 24 if using the first residual connection
                            16,
                            self.activation,
                            self.grouping_strategy,
                            self.nbr_groups,
                            bias)

        if self.network_depth > 1:
            # encoder block
            self.enc_3 = DoubleConv(16,
                                32,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups,
                                bias)

            # decoder block
            self.dec_2 = DoubleConv(48,
                                16,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups,
                                bias)

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
                                self.nbr_groups,
                                bias)

        # max pooling layer
        self.mp = nn.MaxPool3d(2)
        # upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # tanh activation
        self.tanh = nn.Tanh()

    def forward(self, neighborhood):
        # stereo_tsdf = neighborhood['stereo']
        # print(tof_tsdf.shape)
        # print(neighborhood[0, :, 32, 32, 32])

        if self.tanh_weight:
            neighborhood[0, 1, :, :, :] = self.tanh(neighborhood[0, 1, :, :, :])

        if not self.w_features: # this line should not be needed since I don't feed the features if
        # we dont select features in the config file. Nope, I still need it at test time since get_local_grids function
        # still feeds the features
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

class Weighting_Decoder(nn.Module):
    def __init__(self, config):

        super(Weighting_Decoder, self).__init__()

        # self.network_depth = config.FILTERING_MODEL.CONV3D_MODEL.network_depth
        self.activation = eval(config.FILTERING_MODEL.CONV3D_MODEL.activation)
        self.n_features = config.FEATURE_MODEL.n_features

        self.features_to_weight_head = config.FILTERING_MODEL.features_to_weight_head
        self.sdf_enc_to_weight_head = config.FILTERING_MODEL.sdf_enc_to_weight_head
        self.network_depth = 1 # for now

        # add encoder blocks
        self.enc_1 = nn.Sequential(nn.Conv3d(len(config.DATA.input)*(16*self.sdf_enc_to_weight_head + self.features_to_weight_head*self.n_features),
                            24, 3, padding=1, padding_mode='replicate'),
                            self.activation)

        self.enc_2 = nn.Sequential(nn.Conv3d(24,
                            16, 3, padding=1, padding_mode='replicate'),
                            self.activation)

        self.dec_1 = nn.Sequential(nn.Conv3d(40,
                            24, 3, padding=1, padding_mode='replicate'),
                            self.activation)

        self.dec_last = nn.Sequential(nn.Conv3d(24,
                            12, 1, padding=0),
                            self.activation,
                            nn.Conv3d(12,
                            1, 1, padding=0))
                                    

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
        x = self.dec_last(x)

        return x

class Weighting_Encoder(nn.Module):
    def __init__(self, config):

        super(Weighting_Encoder, self).__init__()

        # self.network_depth = config.FILTERING_MODEL.CONV3D_MODEL.network_depth
        self.activation = config.FILTERING_MODEL.CONV3D_MODEL.activation
        self.n_features = config.FEATURE_MODEL.n_features
        self.grouping_strategy = config.FILTERING_MODEL.CONV3D_MODEL.grouping_strategy
        self.nbr_groups = config.FILTERING_MODEL.CONV3D_MODEL.nbr_groups
        self.features_to_weight_head = config.FILTERING_MODEL.features_to_weight_head
        self.sdf_enc_to_weight_head = config.FILTERING_MODEL.sdf_enc_to_weight_head
        self.network_depth = config.FILTERING_MODEL.CONV3D_MODEL.network_depth_sensor_weighting
        bias_wn = config.FILTERING_MODEL.CONV3D_MODEL.bias_weight_net

        # add encoder blocks
        self.enc_1 = DoubleConv(self.n_features,
                            8,
                            self.activation,
                            self.grouping_strategy,
                            self.nbr_groups,
                            bias=bias_wn)

        self.enc_2 = DoubleConv(8,
                            16,
                            self.activation,
                            self.grouping_strategy,
                            self.nbr_groups,
                            bias=bias_wn)

        self.dec_1 = DoubleConv(24, # 16 if not using first residual connection or 24 if using the first residual connection
                            16,
                            self.activation,
                            self.grouping_strategy,
                            self.nbr_groups,
                            bias=bias_wn)

        if self.network_depth > 1:
            # encoder block
            self.enc_3 = DoubleConv(16,
                                32,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups,
                                bias=bias_wn)

            # decoder block
            self.dec_2 = DoubleConv(48,
                                16,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups,
                                bias=bias_wn)

        if self.network_depth > 2:
            # encoder block
            self.enc_4 = DoubleConv(32,
                                64,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups,
                                bias=bias_wn)

            # decoder block
            self.dec_3 = DoubleConv(96,
                                32,
                                self.activation,
                                self.grouping_strategy,
                                self.nbr_groups,
                                bias=bias_wn)

        # max pooling layer
        self.mp = nn.MaxPool3d(2)
        # upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
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

class concat_5layer(nn.Module):

    def __init__(self, config):

        super(concat_5layer, self).__init__()

        self.config = config
        self.sensors = config.DATA.input
        self.feature_to_weight_head = config.FILTERING_MODEL.features_to_weight_head
        self.sdf_enc_to_weight_head = config.FILTERING_MODEL.sdf_enc_to_weight_head 
        self.weight_to_weight_head = config.FILTERING_MODEL.weights_to_weight_head 
        self.activation = eval(config.FILTERING_MODEL.CONV3D_MODEL.activation) 
        self.n_features = config.FEATURE_MODEL.n_features
        bias_wn = config.FILTERING_MODEL.CONV3D_MODEL.bias_weight_net
        nbr_input = 16*len(self.sensors)*self.sdf_enc_to_weight_head + len(self.sensors)*(self.n_features + self.weight_to_weight_head)

        self.weight_encoder1 = nn.Sequential(nn.Conv3d(nbr_input, 32, 3, 
                                                    padding=1, padding_mode='replicate', bias=bias_wn),
                                    self.activation)
        self.weight_encoder2 = nn.Sequential(nn.Conv3d(32 + nbr_input, 32, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                    self.activation)
        self.weight_encoder3 = nn.Sequential(nn.Conv3d(64 + nbr_input, 32, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                    self.activation)
        self.weight_decoder1 = nn.Sequential(nn.Conv3d(96 + nbr_input, 32, 1, padding=0, bias=bias_wn),
                                    self.activation)
        self.weight_decoder2 = nn.Conv3d(32, 1, 1, padding=0, bias=bias_wn)


    def forward(self, input_):

        x1 = self.weight_encoder1(input_)
        x2 = torch.cat((x1, input_), dim=1)
        x3 = self.weight_encoder2(x2)
        x4 = torch.cat((x3, x2), dim=1)
        x5 = self.weight_encoder3(x4)
        x6 = torch.cat((x5, x4), dim=1)
        x7 = self.weight_decoder1(x6)
        x = self.weight_decoder2(x7)

        return x

class residual_5layer(nn.Module):

    def __init__(self, config):

        super(residual_5layer, self).__init__()

        self.config = config
        self.sensors = config.DATA.input
        self.feature_to_weight_head = config.FILTERING_MODEL.features_to_weight_head
        self.sdf_enc_to_weight_head = config.FILTERING_MODEL.sdf_enc_to_weight_head 
        self.weight_to_weight_head = config.FILTERING_MODEL.weights_to_weight_head 
        self.activation = eval(config.FILTERING_MODEL.CONV3D_MODEL.activation) 
        self.n_features = config.FEATURE_MODEL.n_features
        bias_wn = config.FILTERING_MODEL.CONV3D_MODEL.bias_weight_net

        self.weight_decoder1 = nn.Sequential(nn.Conv3d(16*len(self.sensors)*self.sdf_enc_to_weight_head + len(self.sensors)*(self.n_features + self.weight_to_weight_head), 32, 3, 
                                                    padding=1, padding_mode='replicate', bias=bias_wn),
                                    self.activation)
        self.weight_decoder2 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                    self.activation)
        self.weight_decoder3 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                    self.activation)
        self.weight_decoder4 = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                    self.activation)
        self.weight_decoder5 = nn.Conv3d(16, 1, 1, padding=0, bias=bias_wn)


    def forward(self, input_):

        x1 = self.weight_decoder1(input_)
        x2 = self.weight_decoder2(x1)
        x3 = x1 + x2
        x4 = self.weight_decoder3(x3)
        x5 = x3 + x4
        x6 = self.weight_decoder4(x5)
        x = self.weight_decoder5(x6)

        return x

class Weighting_Decoder2(nn.Module):

    def __init__(self, config):

        super(Weighting_Decoder2, self).__init__()

        self.config = config
        self.output_scale = config.FILTERING_MODEL.output_scale
        self.trunc_value = config.DATA.trunc_value
        self.test_mode = config.SETTINGS.test_mode
        self.sensors = config.DATA.input
        self.feature_to_weight_head = config.FILTERING_MODEL.features_to_weight_head
        self.sdf_enc_to_weight_head = config.FILTERING_MODEL.sdf_enc_to_weight_head 
        self.weighting_complexity = config.FILTERING_MODEL.CONV3D_MODEL.weighting_complexity
        self.activation = eval(config.FILTERING_MODEL.CONV3D_MODEL.activation) 
        self.n_features = config.FEATURE_MODEL.n_features
        bias_wn = config.FILTERING_MODEL.CONV3D_MODEL.bias_weight_net

        self.encoder = nn.ModuleDict()
        for sensor_ in config.DATA.input:
            self.encoder[sensor_] = Weighting_Encoder(config)

        self.decoder = nn.Sequential(nn.Conv3d(len(self.sensors)*16*self.sdf_enc_to_weight_head + len(self.sensors)*16, 32, 1, 
                                                                padding=0, bias=bias_wn),
                                                self.activation,
                                                nn.Conv3d(32, 16, 1, padding=0, bias=bias_wn),
                                                self.activation,
                                                nn.Conv3d(16, 1, 1, padding=0, bias=bias_wn))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, neighborhood):
        # print(self.decoder[0].weight)
        sdf_enc_input = dict()
        feature_encoding = dict()

        for k, sensor_ in enumerate(self.sensors):
            start  = k*(self.n_features + 16*self.sdf_enc_to_weight_head) # 16 because we use 16 features to decode the sdf for each sensor
            end  = (k + 1)*(self.n_features + 16*self.sdf_enc_to_weight_head)
            sensor_input = neighborhood[:, start:end, :, :, :]
            if self.sdf_enc_to_weight_head:
                sdf_enc_input[sensor_] = sensor_input[:, :16, :, :, :]
                sensor_input = sensor_input[:, 16:, :, :, :]

            # print(sensor_input[:, :, 0, 0, 0])
            feature_encoding[sensor_] = self.encoder[sensor_](sensor_input)

        input_ = None
        for k, sensor_ in enumerate(self.sensors):
            if k == 0:
                if self.sdf_enc_to_weight_head:
                    input_ = torch.cat((sdf_enc_input[sensor_], feature_encoding[sensor_]), dim=1)
                else:
                    input_ = feature_encoding[sensor_]
            else:
                if self.sdf_enc_to_weight_head:
                    inp = torch.cat((sdf_enc_input[sensor_], feature_encoding[sensor_]), dim=1)
                    input_ = torch.cat((input_, inp), dim=1)
                else:
                    input_ = torch.cat((input_, feature_encoding[sensor_]), dim=1)


        x = self.decoder(input_)

        return x

class FilteringNet(nn.Module):

    def __init__(self, config):

        super(FilteringNet, self).__init__()

        self.config = config
        self.output_scale = config.FILTERING_MODEL.output_scale
        self.trunc_value = config.DATA.trunc_value
        self.test_mode = config.SETTINGS.test_mode
        self.sensors = config.DATA.input
        self.feature_to_weight_head = config.FILTERING_MODEL.features_to_weight_head
        self.sdf_enc_to_weight_head = config.FILTERING_MODEL.sdf_enc_to_weight_head 
        self.weight_to_weight_head = config.FILTERING_MODEL.weights_to_weight_head 
        self.sdf_to_weight_head = config.FILTERING_MODEL.sdf_to_weight_head
        self.weighting_complexity = config.FILTERING_MODEL.CONV3D_MODEL.weighting_complexity
        self.activation = eval(config.FILTERING_MODEL.CONV3D_MODEL.activation) 
        self.n_features = config.FEATURE_MODEL.n_features
        self.residual_learning = config.FILTERING_MODEL.residual_learning
        self.use_outlier_filter = config.FILTERING_MODEL.use_outlier_filter
        self.alpha_force = config.FILTERING_MODEL.alpha_force
        self.alpha_supervision = config.LOSS.alpha_supervision
        self.alpha_single_sensor_supervision = config.LOSS.alpha_single_sensor_supervision
        bias = config.FILTERING_MODEL.CONV3D_MODEL.bias
        bias_wn = config.FILTERING_MODEL.CONV3D_MODEL.bias_weight_net
        self.outlier_filter_model = config.FILTERING_MODEL.CONV3D_MODEL.outlier_filter_model
        self.add_outlier_channel = config.FILTERING_MODEL.outlier_channel
        
        if self.use_outlier_filter:
            self.encoder = nn.ModuleDict()
            self.sdf_layer = nn.ModuleDict()
            for sensor_ in config.DATA.input:
                if self.outlier_filter_model == 'simple':
                    self.sdf_layer[sensor_] = nn.Conv3d(1, 1, 3, padding=1, padding_mode='replicate', bias=bias)
                else:
                    self.encoder[sensor_] = FilteringNetEncoder(config)
                    self.sdf_layer[sensor_] = nn.Conv3d(16, 1, 1, padding=0, bias=bias)
                # if self.residual_learning:
                #     self.encoder[sensor_].apply(init_weights)
                #     self.sdf_layer[sensor_].apply(init_weights)




        # alpha layer
        if len(config.DATA.input) > 1:
            if self.weighting_complexity == '1layer':
                self.weight_decoder = nn.Conv3d(16*len(self.sensors)*self.sdf_enc_to_weight_head + len(self.sensors)*(self.n_features + self.weight_to_weight_head), 1, 1, padding=0, bias=bias_wn)
            elif self.weighting_complexity == '3layer':
                self.weight_decoder = nn.Sequential(nn.Conv3d(16*len(self.sensors)*self.sdf_enc_to_weight_head + len(self.sensors)*(self.sdf_to_weight_head + self.n_features*self.feature_to_weight_head + self.weight_to_weight_head), 32, 3, 
                                                                    padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(32, 16, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(16, 1, 1, padding=0, bias=bias_wn))
            elif self.weighting_complexity == '3layer_1extra1conv':
                self.weight_decoder = nn.Sequential(nn.Conv3d(16*len(self.sensors)*self.sdf_enc_to_weight_head + len(self.sensors)*(self.n_features*self.feature_to_weight_head + self.weight_to_weight_head), 32, 3, 
                                                padding=1, padding_mode='replicate', bias=bias_wn),
                                                self.activation,
                                                nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                                                    self.activation,
                                                nn.Conv3d(32, 16, 1, padding=0, bias=bias_wn),
                                                self.activation,
                                                nn.Conv3d(16, 1, 1, padding=0, bias=bias_wn))
            elif self.weighting_complexity == '5layer':
                self.weight_decoder = nn.Sequential(nn.Conv3d(16*len(self.sensors)*self.sdf_enc_to_weight_head + len(self.sensors)*(self.n_features + self.weight_to_weight_head), 32, 3, 
                                                                    padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(32, 16, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(16, 1 + self.add_outlier_channel, 1, padding=0, bias=bias_wn))
            elif self.weighting_complexity == '4layer':
                self.weight_decoder = nn.Sequential(nn.Conv3d(16*len(self.sensors)*self.sdf_enc_to_weight_head + len(self.sensors)*(self.n_features + self.weight_to_weight_head), 32, 3, 
                                                                    padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(32, 16, 3, padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(16, 1 + self.add_outlier_channel, 1, padding=0, bias=bias_wn))
            elif self.weighting_complexity == '2layer':
                self.weight_decoder = nn.Sequential(nn.Conv3d(16*len(self.sensors)*self.sdf_enc_to_weight_head + len(self.sensors)*(self.n_features + self.weight_to_weight_head), 16, 3, \
                                                                    padding=1, padding_mode='replicate', bias=bias_wn),
                                                    self.activation,
                                                    nn.Conv3d(16, 1 + self.add_outlier_channel, 1, padding=0, bias=bias_wn))
            elif self.weighting_complexity == 'residual_5layer':
                self.weight_decoder = residual_5layer(config)
            elif self.weighting_complexity == 'concat_5layer':
                self.weight_decoder = concat_5layer(config)
            elif self.weighting_complexity == 'unet':
                self.weight_decoder = Weighting_Decoder(config)
            elif self.weighting_complexity == 'encode_features_first':
                self.weight_decoder = Weighting_Decoder2(config)
                # the negatith concatenating the sdf encoding and the features is that 
                # the sdf encoding already is neighborhood aware with a receptive field big enough.
                # applying more convolutions to this encoding does not make sense. But we need to apply 
                # processing steps to the 2D features to make then neighborhood aware. This means that 
                # it makes more sense to process the 2D features through a unet separately and then
                # feed them with the sdf encoding through a final prediction layer/s.

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, neighborhood):
        weight = dict()
        sdf = dict()
        enc = dict()
        output = dict()

        for sensor_ in self.sensors:
            if self.use_outlier_filter:
                if self.outlier_filter_model == 'simple':
                    sdf[sensor_] = self.output_scale*self.tanh(self.sdf_layer[sensor_](neighborhood[sensor_][:, 0, :, :, :].unsqueeze(1)))
                else:  
                    enc[sensor_] = self.encoder[sensor_](neighborhood[sensor_])
                    sdf[sensor_] = self.output_scale*self.tanh(self.sdf_layer[sensor_](enc[sensor_]))

                sdf[sensor_] = torch.clamp(sdf[sensor_],
                                   -self.trunc_value,
                                   self.trunc_value)
                if sdf[sensor_].isnan().sum() > 0 or sdf[sensor_].isinf().sum() > 0:
                    print(sensor_)
                    print('sdf[sensor_] nan: ', sdf[sensor_].isnan().sum())
                    print('sdf[sensor_] inf: ', sdf[sensor_].isinf().sum())
                if self.residual_learning:
                    sdf[sensor_] += neighborhood[sensor_][:, 0, :, :, :]
                    sdf[sensor_] = torch.clamp(sdf[sensor_],
                                -self.trunc_value,
                                self.trunc_value)
            else:
                sdf[sensor_] = neighborhood[sensor_][:, 0, :, :, :]
  

            weight[sensor_] = neighborhood[sensor_][:, 1, :, :, :].unsqueeze(1)

        if len(self.sensors) == 2:
            for sensor_ in self.sensors:
                output['tsdf_' + sensor_] = sdf[sensor_].squeeze()
                output[sensor_ + '_init'] = weight[sensor_].squeeze() > 0
    
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
                        inp = torch.cat((inp, neighborhood[sensor_][:, 0, :, :, :].unsqueeze(1)), dim=1)
                if self.feature_to_weight_head:
                    if inp is None:
                        inp = neighborhood[sensor_][:, 2:, :, :, :]
                    else:
                        inp = torch.cat((inp, neighborhood[sensor_][:, 2:, :, :, :]), dim=1)
                if self.weight_to_weight_head:
                    if self.config.FILTERING_MODEL.tanh_weight:
                        if self.config.FILTERING_MODEL.inverted_weight:
                            weights = torch.ones_like(neighborhood[sensor_][:, 1, :, :, :].unsqueeze(1)) - self.tanh(neighborhood[sensor_][:, 1, :, :, :])
                        else:
                            weights = self.tanh(neighborhood[sensor_][:, 1, :, :, :]).unsqueeze(1)
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

                    # if self.sdf_enc_to_weight_head and self.feature_to_weight_head:
                    #     input_ = torch.cat((enc[sensor_], neighborhood[sensor_][:, 2:, :, :, :]), dim=1)
                    # elif self.feature_to_weight_head:
                    #     input_ = neighborhood[sensor_][:, 2:, :, :, :]
                    # elif self.sdf_enc_to_weight_head:
                    #     input_ = enc[sensor_]
                    alpha_val[sensor_] = torch.zeros_like(sdf[sensor_])
                else:
                    # if self.sdf_enc_to_weight_head and self.feature_to_weight_head:
                    #     inp = torch.cat((enc[sensor_], neighborhood[sensor_][:, 2:, :, :, :]), dim=1)
                    # elif self.feature_to_weight_head:
                    #     inp = neighborhood[sensor_][:, 2:, :, :, :]
                    # elif self.sdf_enc_to_weight_head:
                    #     inp = enc[sensor_]
                    # input_ = torch.cat((input_, inp), dim=1)
                    alpha_val[sensor_] = torch.ones_like(sdf[sensor_]) 

            if input_.isnan().sum() > 0:
                print('Input isnan: ', input_.isnan().sum())

            # for layer in self.weight_decoder:
            #     if isinstance(layer, nn.Conv3d):
            #         if layer.weight.isnan().sum() > 0 or layer.bias.isnan().sum() > 0:
            #             print(layer)
            #             print('layer bias nan: ', layer.bias.isnan().sum())
            #             print('layer weight nan: ', layer.weight.isnan().sum())
            # print('inputsum: ', input_.sum())
            alpha = self.sigmoid(self.weight_decoder(input_))

            # print('alphasum: ', alpha.sum())
            if alpha.isnan().sum() > 0 or alpha.isinf().sum() > 0:
                print('alpha nan: ', alpha.isnan().sum())
                print('alpha inf: ', alpha.isinf().sum())
                return None

            if self.test_mode or self.alpha_supervision or self.alpha_single_sensor_supervision:
                output['sensor_weighting'] = alpha.squeeze()

            if self.add_outlier_channel:
                alpha_sdf = alpha[:, 0, :, :, :] # make sure that alpha remains the same!
            else:
                alpha_sdf = alpha

            # print('input_ ', input_)
            # print('alpha: ', alpha)


            # this step is to not filter the voxels where we only have one sensor observing
            # note that we save the variable alpha and not alpha_sdf later so we can still
            # use the outlier filter as usual. During test time this step is important to avoid
            # that we make a smooth decision where only one sensor integrates. This step is 
            # important when we want to keep the single sensor observation during test time.
            for sensor_ in self.config.DATA.input:
                alpha_sdf = torch.where(weight[sensor_] == 0, alpha_val[sensor_], alpha_sdf)

            sdf_final = None

            for k, sensor_ in enumerate(self.config.DATA.input):
                if k == 0:
                    sdf_final = alpha_sdf * sdf[sensor_]
                else:
                    sdf_final += (1 - alpha_sdf) * sdf[sensor_]

            # print('sdf final: ', sdf_final.sum())
            # print('after sdf final computation')
            # for sensor_ in self.config.DATA.input:
            #     print(sdf[sensor_].sum())

            # input_ = torch.cat((enc['tof'], enc['stereo']), dim=1)
            # alpha = self.sigmoid(self.weight_decoder(input_))
            
            # alpha = torch.where(weight['stereo'] == 0, torch.ones_like(alpha), alpha)
            # alpha = torch.where(weight['tof'] == 0, torch.zeros_like(alpha), alpha)


            # sdf = alpha * sdf['tof'] + (1 - alpha) * sdf['stereo']
            output['tsdf'] = sdf_final.squeeze()

        elif len(self.sensors) > 2:
            raise NotImplementedError
        else:
            output['tsdf'] = sdf[list(sdf.keys())[0]].squeeze()

        return output




def init_weights(model):
    if isinstance(model, nn.Conv3d):
        nn.init.uniform_(model.weight.data, -0.001, 0.001)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)

