# import torch

# from torch import nn

# class FilteringNet(nn.Module):

#     def __init__(self, config):

#         super(FilteringNet, self).__init__()

#         self.config = config
#         self.side_length_in = config.FILTERING_MODEL.MLP_MODEL.neighborhood
#         self.side_length_out = config.FILTERING_MODEL.MLP_MODEL.neighborhood_out
#         self.n_features = config.FILTERING_MODEL.w_features*config.FEATURE_MODEL.n_features 
#         self.tanh_weight = config.FILTERING_MODEL.tanh_weight
#         self.attend_to_uninit = config.FILTERING_MODEL.MLP_MODEL.attend_to_uninit

#         self.softmax = nn.Softmax(dim=1) 
#         self.tanh = torch.nn.Tanh()

#         self.n_inputs = (2 + self.n_features)*pow(self.side_length_in, 3) # 7x7x7x2 is 686
#         self.n_outputs = pow(self.side_length_out, 3)
     
#         # self.feature_encoding = torch.nn.ModuleDict()
#         # for sensor_ in config.DATA.input:
#         #     self.feature_encoding 
#         self.feature_encoding_tof = nn.Sequential(nn.Linear(self.n_inputs, self.n_outputs),
#                                                                nn.Tanh())
#         self.feature_encoding_stereo = nn.Sequential(nn.Linear(self.n_inputs, self.n_outputs),
#                                                                nn.Tanh())
   
#         # fuse and decode sdf weights. Here I can try with center voxel conditioning as well if it does not work without

#         self.decoding1 = nn.Sequential(nn.Linear(2*(self.n_outputs + 2  + self.n_features), 108),
#                                               nn.Tanh())

#         self.decoding2 = nn.Sequential(nn.Linear(108 + 2*(2  + self.n_features), 108),
#                                           nn.Tanh())
#         self.decoding3 = nn.Sequential(nn.Linear(108 + 2*(2  + self.n_features), 54),
#                                           nn.Tanh())
#         self.final = nn.Sequential(nn.Linear(54, 54))


#     def forward(self, neighborhood):

#         tof_tsdf = neighborhood['tof']
#         stereo_tsdf = neighborhood['stereo']

#         if self.tanh_weight:
#             tof_tsdf[:, :, 1] = self.tanh(tof_tsdf[:, :, 1]) 
#             stereo_tsdf[:, :, 1] = self.tanh(stereo_tsdf[:, :, 1]) 

#         # extract output data (so that we know what sdf values to multiply with)
#         tof_tsdf = tof_tsdf.view(-1, self.side_length_in, self.side_length_in, self.side_length_in, 2 + self.n_features)
#         if self.side_length_in == 9:
#             tof_tsdf = tof_tsdf[:, 3:6, 3:6, 3:6, :]
#         elif self.side_length_in == 7:
#             tof_tsdf = tof_tsdf[:, 2:5, 2:5, 2:5, :]
#         elif self.side_length_in == 5:
#             tof_tsdf = tof_tsdf[:, 1:4, 1:4, 1:4, :]
#         tof_tsdf = tof_tsdf.contiguous().view(-1, self.n_outputs, 2 + self.n_features)

#         stereo_tsdf = stereo_tsdf.view(-1, self.side_length_in, self.side_length_in, self.side_length_in, 2 + self.n_features)
#         if self.side_length_in == 9:
#             stereo_tsdf = stereo_tsdf[:, 3:6, 3:6, 3:6, :]
#         elif self.side_length_in == 7:
#             stereo_tsdf = stereo_tsdf[:, 2:5, 2:5, 2:5, :]
#         elif self.side_length_in == 5:
#             stereo_tsdf = stereo_tsdf[:, 1:4, 1:4, 1:4, :]
#         stereo_tsdf = stereo_tsdf.contiguous().view(-1, self.n_outputs, 2 + self.n_features)
#         # print(stereo_tsdf.shape)
#         # print(tof_tsdf[:, 13, :].unsqueeze(-1).shape)
#         center_features = torch.cat((tof_tsdf[:, 13, :], stereo_tsdf[:, 13, :]), dim=1)

#         if not self.tanh_weight:
#             tof_center_weight = self.tanh(tof_tsdf[:, 13, 1]).unsqueeze(-1)
#             stereo_center_weight = self.tanh(stereo_tsdf[:, 13, 1]).unsqueeze(-1)
#             # tof_center_weight = tof_tsdf[:, 13, 1].unsqueeze(-1)
#             # stereo_center_weight = stereo_tsdf[:, 13, 1].unsqueeze(-1)
#         else:
#             tof_center_weight = tof_tsdf[:, 13, 1].unsqueeze(-1)
#             stereo_center_weight = stereo_tsdf[:, 13, 1].unsqueeze(-1)
#         # print(stereo_center_weight.shape)

#         tof_weight = tof_tsdf[:, :, 1]
#         stereo_weight = stereo_tsdf[:, :, 1]
#         tof_tsdf = tof_tsdf[:, :, 0]
#         stereo_tsdf = stereo_tsdf[:, :, 0]

#         # get center voxel features
#         center_features = torch.cat((neighborhood['tof'][:, 13, :], neighborhood['stereo'][:, 13, :]), dim=1)

#         # concatenate sensor neighborhoods
#         neighborhood_flat = torch.cat((neighborhood['tof'], neighborhood['stereo']), dim=2)
#         # flatten input to mlp
#         neighborhood_flat_tof = torch.flatten(neighborhood['tof'], 1, 2)
#         neighborhood_flat_stereo = torch.flatten(neighborhood['stereo'], 1, 2)
#         # del neighborhood
#         # compute compressed neighborhood data
#         # print(neighborhood_flat_tof.shape)

#         feat_tof = self.feature_encoding_tof.forward(neighborhood_flat_tof)
#         # print(feat_tof.shape)
#         feat_stereo = self.feature_encoding_stereo.forward(neighborhood_flat_stereo)


#         feat = torch.cat((feat_tof, feat_stereo), dim=1)

#         x = torch.cat([center_features, feat], dim=1)

#         # weight prediction network
#         x = self.decoding1.forward(x)
#         x = torch.cat([center_features, x], dim=1)
#         x = self.decoding2.forward(x)
#         x = torch.cat([center_features, x], dim=1)
#         x = self.decoding3.forward(x)
#         scores = self.final.forward(x)

#         # print('scores isnan: ', torch.isnan(scores).sum())

#         if not self.attend_to_uninit:

#             scores[:, :27] = scores[:, :27] - torch.where(tof_weight > 0, torch.zeros_like(scores[:, :27]), torch.ones_like(scores[:, :27]) * float('inf'))
#             scores[:, 27:] = scores[:, 27:] - torch.where(stereo_weight > 0, torch.zeros_like(scores[:, 27:]), torch.ones_like(scores[:, 27:]) * float('inf'))
     

#         # print('scores isinf: ', torch.isinf(scores).sum())
#         scores = self.softmax(scores)
#         # print('scores first: ', scores[0, :])

#         x = torch.mul(scores, torch.cat([tof_tsdf, stereo_tsdf], dim=1))
#         x = x.sum(dim=1)

#         output = dict()
#         output['tsdf'] = x

#         if len(self.config.DATA.input) > 1 and self.config.SETTINGS.test_mode:
#             alpha = torch.sum(scores[:, :27], dim=1)
#             output['sensor_weighting'] = alpha.squeeze()

#         return output

import torch

from torch import nn

class FilteringNet(nn.Module):

    def __init__(self, config):

        super(FilteringNet, self).__init__()

        self.config = config
        self.side_length_in = config.FILTERING_MODEL.MLP_MODEL.neighborhood
        self.side_length_out = config.FILTERING_MODEL.MLP_MODEL.neighborhood_out
        self.n_features = config.FILTERING_MODEL.w_features*config.FEATURE_MODEL.n_features 
        self.tanh_weight = config.FILTERING_MODEL.tanh_weight
        self.attend_to_uninit = config.FILTERING_MODEL.MLP_MODEL.attend_to_uninit

        self.softmax = nn.Softmax(dim=1) 
        self.tanh = torch.nn.Tanh()

        self.n_inputs = (2 + self.n_features)*pow(self.side_length_in, 3) # 7x7x7x2 is 686
        self.n_outputs = pow(self.side_length_out, 3)
     
        self.feature_encoding = torch.nn.ModuleDict()
        for sensor_ in config.DATA.input:
            self.feature_encoding[sensor_] = nn.Sequential(nn.Linear(self.n_inputs, self.n_outputs),
                                                               nn.Tanh())
   
        # fuse and decode sdf weights. Here I can try with center voxel conditioning as well if it does not work without
        nbr_sensors = len(self.config.DATA.input)
        self.decoding1 = nn.Sequential(nn.Linear(nbr_sensors*(self.n_outputs + 2  + self.n_features), nbr_sensors*54),
                                              nn.Tanh())

        self.decoding2 = nn.Sequential(nn.Linear(nbr_sensors*54 + nbr_sensors*(2  + self.n_features), nbr_sensors*54),
                                          nn.Tanh())
        self.decoding3 = nn.Sequential(nn.Linear(nbr_sensors*54 + nbr_sensors*(2  + self.n_features), nbr_sensors*27),
                                          nn.Tanh())
        self.final = nn.Sequential(nn.Linear(nbr_sensors*27, nbr_sensors*self.n_outputs))


    def forward(self, neighborhood):

        features = dict()
        center_weight = dict()
        weight = dict()
        tsdf = dict()
        center_features = None 
        encoding = None
        for k, sensor_ in enumerate(self.config.DATA.input):
            features[sensor_] = neighborhood[sensor_]

            if self.tanh_weight:
                features[sensor_][:, :, 1] = self.tanh(features[sensor_][:, :, 1])

            center_weight[sensor_] = features[sensor_][:, 13, 1].unsqueeze(-1)

            # extract output data (so that we know what sdf values to multiply with)
            features[sensor_] = features[sensor_].view(-1, self.side_length_in, self.side_length_in, self.side_length_in, 2 + self.n_features)
            if self.side_length_in == 9:
                features[sensor_] = tsdf[sensor_][:, 3:6, 3:6, 3:6, :]
            elif self.side_length_in == 7:
                features[sensor_] = features[sensor_][:, 2:5, 2:5, 2:5, :]
            elif self.side_length_in == 5:
                features[sensor_] = features[sensor_][:, 1:4, 1:4, 1:4, :]
            features[sensor_] = features[sensor_].contiguous().view(-1, self.n_outputs, 2 + self.n_features)

            weight[sensor_] = features[sensor_][:, :, 1]

            # flatten input to mlp
            neighborhood_flat = torch.flatten(neighborhood[sensor_], 1, 2)
 
            if k == 0:
                tsdf = features[sensor_][:, :, 0]
                center_features = features[sensor_][:, 13, :]
                encoding = self.feature_encoding[sensor_](neighborhood_flat)
            else:
                tsdf = torch.cat((tsdf, features[sensor_][:, :, 0]), dim=1)
                center_features = torch.cat((center_features, features[sensor_][:, 13, :]), dim=1)
                encoding = torch.cat((encoding, self.feature_encoding[sensor_](neighborhood_flat)), dim=1)

  
        x = torch.cat([center_features, encoding], dim=1)

        # weight prediction network
        x = self.decoding1.forward(x)
        x = torch.cat([center_features, x], dim=1)
        x = self.decoding2.forward(x)
        x = torch.cat([center_features, x], dim=1)
        x = self.decoding3.forward(x)
        scores = self.final.forward(x)

        if not self.attend_to_uninit:
            if len(self.config.DATA.input) > 1:
                for k, sensor_ in enumerate(self.config.DATA.input):
                    start = k*self.n_outputs
                    end = (k + 1)*self.n_outputs
                    scores[:, start:end] = scores[:, start:end] - torch.where(weight[sensor_] > 0, torch.zeros_like(scores[:, start:end]), torch.ones_like(scores[:, start:end]) * float('inf'))
            else:
                scores[:, :] = scores[:, :] - torch.where(weight[self.config.DATA.input[0]] > 0, torch.zeros_like(scores[:, :]), torch.ones_like(scores[:, :]) * float('inf'))

        scores = self.softmax(scores)
   
        x = torch.mul(scores, tsdf)
        x = x.sum(dim=1)

        output = dict()
        output['tsdf'] = x

        if len(self.config.DATA.input) > 1 and self.config.SETTINGS.test_mode:
            alpha = torch.sum(scores[:, :27], dim=1)
            output['sensor_weighting'] = alpha.squeeze()

        return output