import torch

import numpy as np

from torch import nn

class TranslationNetEncoder(torch.nn.Module):
    
    def __init__(self, config):

        super(TranslationNetEncoder, self).__init__()

        self.config = config

        self.n_features = 2 + config.FILTERING_MODEL.features_to_sdf_enc*config.FEATURE_MODEL.n_features  # + 2 for weight and tsdf 2 because two sensors. 
        activation = eval(config.FILTERING_MODEL.MLP_MODEL.activation)

        self.layer_context = torch.nn.Sequential(torch.nn.Linear((self.config.FILTERING_MODEL.MLP_MODEL.neighborhood ** 3) * self.n_features, 12*self.n_features),
                                                  # torch.nn.LayerNorm([12*self.n_features], elementwise_affine=False),
                                                  activation,
                                                  torch.nn.Linear(12*self.n_features, 8*self.n_features),
                                                  # torch.nn.LayerNorm([8*self.n_features], elementwise_affine=False),
                                                  activation)

    def forward(self, neighborhood):
        # print('1', neighborhood[0, :, :])
        neighborhood = neighborhood.contiguous().view(neighborhood.shape[0], self.config.FILTERING_MODEL.MLP_MODEL.neighborhood**3 * self.n_features)
        # print('2', neighborhood[0, :])
        # n = neighborhood.contiguous().view(neighborhood.shape[0], self.config.FILTERING_MODEL.MLP_MODEL.neighborhood**3, self.n_features)
        # print('3', n[0, :, :])
        context_features = self.layer_context(neighborhood)

        return context_features

class TranslationNetDecoder(torch.nn.Module):
    
    def __init__(self, config):

        super(TranslationNetDecoder, self).__init__()

        self.config = config

        self.n_features = 2 + config.FILTERING_MODEL.features_to_sdf_enc*config.FEATURE_MODEL.n_features  # + 2 for weight and tsdf 2 because two sensors. 
        activation = eval(config.FILTERING_MODEL.MLP_MODEL.activation)

        self.layer1 = torch.nn.Sequential(
            #torch.nn.Linear(3 + self.n_features + self.n_features, 64),
            torch.nn.Linear(8*self.n_features + self.n_features, 32),
            activation)

        self.layer2 = torch.nn.Sequential(
            # torch.nn.Linear(3 + self.n_features + 64, 32),
            torch.nn.Linear(self.n_features + 32 , 16),
            activation)

        self.layer3 = torch.nn.Sequential(
            #torch.nn.Linear(3 + self.n_features + 32, 16),
            torch.nn.Linear(self.n_features + 16, 8),
            activation)

        self.sdf_head = torch.nn.Sequential(torch.nn.Linear(8, 1),
                                                torch.nn.Tanh())

        if self.config.FILTERING_MODEL.MLP_MODEL.occ_head:
            self.occ_head = torch.nn.Sequential(torch.nn.Linear(8, 1),
                                                    torch.nn.Sigmoid())

    def forward(self, neighborhood, context_features):

        if self.config.FILTERING_MODEL.MLP_MODEL.neighborhood == 3:
            center_neighborhood = neighborhood[:, 13, :]
        else: # Neighborhood is 5
            center_neighborhood = neighborhood[:, 62, :]

        features = torch.cat([center_neighborhood, context_features], dim=1)
        features = self.layer1(features)
        features = torch.cat([center_neighborhood, features], dim=1)
        features = self.layer2(features)
        features = torch.cat([center_neighborhood, features], dim=1)
        features = self.layer3(features)
        sdf = self.sdf_head(features)
        if self.config.FILTERING_MODEL.MLP_MODEL.occ_head:
            occ = self.occ_head(features)

        sdf = torch.clamp(self.config.FILTERING_MODEL.output_scale * sdf,
                               -self.config.DATA.trunc_value,
                               self.config.DATA.trunc_value)

        return sdf, occ


class TranslationNet(torch.nn.Module):
    def __init__(self, config):

        super(TranslationNet, self).__init__()

        self.config = config

        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
        for sensor_ in config.DATA.input:
            self.encoder[sensor_] = TranslationNetEncoder(config)
            self.decoder[sensor_] = TranslationNetDecoder(config)

        if len(config.DATA.input) > 1:
            self.n_features = 2 + config.FILTERING_MODEL.features_to_sdf_enc*config.FEATURE_MODEL.n_features
            activation = eval(config.FILTERING_MODEL.MLP_MODEL.activation)
            self.sensor_weighting = torch.nn.Sequential(torch.nn.Linear(2*8*self.n_features, 8*self.n_features),
                                                  # torch.nn.LayerNorm([8*self.n_features], elementwise_affine=False),
                                                  activation,
                                                  torch.nn.Linear(8*self.n_features, 1),
                                                  torch.nn.Sigmoid())

        self.tanh = torch.nn.Tanh()


    def forward(self, neighborhood):
        if self.config.FILTERING_MODEL.tanh_weight:
            for sensor_ in self.config.DATA.input:
                neighborhood[sensor_][:, :, 1] = self.tanh(neighborhood[sensor_][:, :, 1])

        sdf = dict()
        occ = dict()
        context_feature = dict() 
        for sensor_ in self.config.DATA.input:
            context_feature[sensor_] = self.encoder[sensor_](neighborhood[sensor_]) 
            sdf[sensor_], occ[sensor_] = self.decoder[sensor_](neighborhood[sensor_], context_feature[sensor_])

        output = dict()
        if len(self.config.DATA.input) > 1:
            input_ = None
            center_weight = dict()
            alpha_val = dict()

            for k, sensor_ in enumerate(self.config.DATA.input):
                if k == 0:
                    input_ = context_feature[sensor_]
                    alpha_val[sensor_] = torch.zeros_like(sdf[sensor_])
                else:
                    input_ = torch.cat((input_, context_feature[sensor_]), dim=1)
                    alpha_val[sensor_] = torch.ones_like(sdf[sensor_]) 
                if self.config.FILTERING_MODEL.MLP_MODEL.neighborhood == 3:
                    center_weight[sensor_] = neighborhood[sensor_][:, 13, 1]
                else: # Neighborhood is 5
                    center_weight[sensor_] = neighborhood[sensor_][:, 62, 1]

            alpha = self.sensor_weighting(input_)

            for sensor_ in self.config.DATA.input:
                alpha = torch.where(center_weight[sensor_].unsqueeze(-1) == 0, alpha_val[sensor_], alpha)

            sdf_final = None
            for k, sensor_ in enumerate(self.config.DATA.input):
                if k == 0:
                    sdf_final = alpha * sdf[sensor_]
                else:
                    sdf_final += (1 - alpha) * sdf[sensor_]
     
            # # only valid for tof and stereo now
            # input_ = torch.cat((context_feature['tof'], context_feature['stereo']), dim=1)

            # alpha = self.sensor_weighting(input_)
            # center_weight = dict()
            # for sensor_ in self.config.DATA.input:
            #     if self.config.FILTERING_MODEL.MLP_MODEL.neighborhood == 3:
            #         center_weight[sensor_] = neighborhood[sensor_][:, 13, 1]
            #     else: # Neighborhood is 5
            #         center_weight[sensor_] = neighborhood[sensor_][:, 62, 1]

            # alpha = torch.where(center_weight['stereo'].unsqueeze(-1) == 0, torch.ones_like(alpha), alpha)
            # alpha = torch.where(center_weight['tof'].unsqueeze(-1) == 0, torch.zeros_like(alpha), alpha)

            # sdf_final = alpha * sdf['tof'] + (1 - alpha) * sdf['stereo']

            for sensor_ in self.config.DATA.input:
                output[sensor_ + '_init'] = center_weight[sensor_] > 0
                output['tsdf_' + sensor_] = sdf[sensor_].squeeze()
                output['occ_' + sensor_] = occ[sensor_].squeeze()

        elif len(self.config.DATA.input) > 2:
            raise NotImplementedError
        else:
            sdf_final = sdf[list(sdf.keys())[0]]
            occ = occ[list(sdf.keys())[0]]
            output['occ'] = occ.squeeze()

        
        output['tsdf'] = sdf_final.squeeze()

        if len(self.config.DATA.input) > 1 and self.config.SETTINGS.test_mode:
            output['sensor_weighting'] = alpha.squeeze()

        return output




# class TranslationNet(torch.nn.Module):

#     def __init__(self, config):

#         super(TranslationNet, self).__init__()

#         self.config = config

#         self.n_features = 2 + config.FILTERING_MODEL.w_features*config.FEATURE_MODEL.n_features  # + 2 for weight and tsdf 2 because two sensors. 
#         activation = eval(config.FILTERING_MODEL.MLP_MODEL.activation)

#         self.layer_context_tof = torch.nn.Sequential(torch.nn.Linear((self.config.FILTERING_MODEL.MLP_MODEL.neighborhood ** 3) * self.n_features, 12*self.n_features),
#                                                   # torch.nn.LayerNorm([12*self.n_features], elementwise_affine=False),
#                                                   activation,
#                                                   torch.nn.Linear(12*self.n_features, 8*self.n_features),
#                                                   # torch.nn.LayerNorm([8*self.n_features], elementwise_affine=False),
#                                                   activation)

#         self.layer_context_stereo = torch.nn.Sequential(torch.nn.Linear((self.config.FILTERING_MODEL.MLP_MODEL.neighborhood ** 3) * self.n_features, 12*self.n_features),
#                                                   # torch.nn.LayerNorm([12*self.n_features], elementwise_affine=False),
#                                                   activation,
#                                                   torch.nn.Linear(12*self.n_features, 8*self.n_features),
#                                                   # torch.nn.LayerNorm([8*self.n_features], elementwise_affine=False),
#                                                   activation)

#         self.alpha_weighting = torch.nn.Sequential(torch.nn.Linear(2*8*self.n_features, 8*self.n_features),
#                                                   # torch.nn.LayerNorm([8*self.n_features], elementwise_affine=False),
#                                                   activation,
#                                                   torch.nn.Linear(8*self.n_features, 1),
#                                                   torch.nn.Sigmoid())

#         self.layer1_sdf_tof = torch.nn.Sequential(
#             #torch.nn.Linear(3 + self.n_features + self.n_features, 64),
#             torch.nn.Linear(8*self.n_features + self.n_features, 32),
#             activation)

#         self.layer2_sdf_tof = torch.nn.Sequential(
#             # torch.nn.Linear(3 + self.n_features + 64, 32),
#             torch.nn.Linear(self.n_features + 32 , 16),
#             activation)

#         self.layer3_sdf_tof = torch.nn.Sequential(
#             #torch.nn.Linear(3 + self.n_features + 32, 16),
#             torch.nn.Linear(self.n_features + 16, 8),
#             activation)

#         self.sdf_head_tof = torch.nn.Sequential(torch.nn.Linear(8, 1),
#                                                 torch.nn.Tanh())

#         if self.config.FILTERING_MODEL.MLP_MODEL.occ_head:
#             self.occ_head_tof = torch.nn.Sequential(torch.nn.Linear(8, 1),
#                                                     torch.nn.Sigmoid())

#         self.layer1_sdf_stereo = torch.nn.Sequential(
#             #torch.nn.Linear(3 + self.n_features + self.n_features, 64),
#             torch.nn.Linear(8*self.n_features + self.n_features, 32),
#             activation)

#         self.layer2_sdf_stereo = torch.nn.Sequential(
#             # torch.nn.Linear(3 + self.n_features + 64, 32),
#             torch.nn.Linear(self.n_features + 32 , 16),
#             activation)

#         self.layer3_sdf_stereo = torch.nn.Sequential(
#             #torch.nn.Linear(3 + self.n_features + 32, 16),
#             torch.nn.Linear(self.n_features + 16, 8),
#             activation)

#         self.sdf_head_stereo = torch.nn.Sequential(torch.nn.Linear(8, 1),
#                                                 torch.nn.Tanh())

#         if self.config.FILTERING_MODEL.MLP_MODEL.occ_head:
#             self.occ_head_stereo = torch.nn.Sequential(torch.nn.Linear(8, 1),
#                                                     torch.nn.Sigmoid())



#         self.tanh = torch.nn.Tanh()


#     def forward(self, neighborhood):
#         print(neighborhood['tof'].shape)
#         print('tof: ', neighborhood['tof'][234, 13, :])
#         print('stereo: ', neighborhood['stereo'][234, 13, :])
#         tof_features = neighborhood['tof']
#         stereo_features = neighborhood['stereo']

#         # print(tof_features[0, 13, :])
#         # print(stereo_features[0, 13, :])

#         if self.config.FILTERING_MODEL.tanh_weight:
#             tof_features[:, :, 1] = self.tanh(tof_features[:, :, 1])
#             stereo_features[:, :, 1] = self.tanh(stereo_features[:, :, 1])

#         if self.config.FILTERING_MODEL.MLP_MODEL.neighborhood == 3:
#             center_features_tof = tof_features[:, 13, :]
#             center_features_stereo = stereo_features[:, 13, :]
#         else: # Neighborhood is 5
#             center_features_tof = tof_features[:, 62, :]
#             center_features_stereo = stereo_features[:, 62, :]

#         tof_features = tof_features.contiguous().view(tof_features.shape[0], self.config.FILTERING_MODEL.MLP_MODEL.neighborhood**3 * self.n_features)
#         stereo_features = stereo_features.contiguous().view(stereo_features.shape[0], self.config.FILTERING_MODEL.MLP_MODEL.neighborhood**3 * self.n_features)

#         tof_context_features = self.layer_context_tof(tof_features)
#         stereo_context_features = self.layer_context_stereo(stereo_features)

#         features = torch.cat([center_features_tof, tof_context_features], dim=1)
#         features = self.layer1_sdf_tof(features)
#         features = torch.cat([center_features_tof, features], dim=1)
#         features = self.layer2_sdf_tof(features)
#         features = torch.cat([center_features_tof, features], dim=1)
#         features = self.layer3_sdf_tof(features)
#         sdf_tof = self.sdf_head_tof(features)
#         if self.config.FILTERING_MODEL.MLP_MODEL.occ_head:
#             occ_tof = self.occ_head_tof(features)

#         features = torch.cat([center_features_stereo, stereo_context_features], dim=1)
#         features = self.layer1_sdf_stereo(features)
#         features = torch.cat([center_features_stereo, features], dim=1)
#         features = self.layer2_sdf_stereo(features)
#         features = torch.cat([center_features_stereo, features], dim=1)
#         features = self.layer3_sdf_stereo(features)
#         sdf_stereo = self.sdf_head_stereo(features)
#         if self.config.FILTERING_MODEL.MLP_MODEL.occ_head:
#             occ_stereo = self.occ_head_stereo(features)

#         features = torch.cat([stereo_context_features, tof_context_features], dim=1)

#         alpha = self.alpha_weighting(features)

#         sdf_tof = torch.clamp(self.config.FILTERING_MODEL.output_scale * sdf_tof,
#                                -self.config.DATA.trunc_value,
#                                self.config.DATA.trunc_value)

#         sdf_stereo = torch.clamp(self.config.FILTERING_MODEL.output_scale * sdf_stereo,
#                                -self.config.DATA.trunc_value,
#                                self.config.DATA.trunc_value)


#         alpha = torch.where(center_features_stereo[:, 1].unsqueeze(-1) == 0, torch.ones_like(alpha), alpha)
#         alpha = torch.where(center_features_tof[:, 1].unsqueeze(-1) == 0, torch.zeros_like(alpha), alpha)

#         sdf = alpha * sdf_tof + (1 - alpha) * sdf_stereo

#         # print((sdf_tof < 0).sum())
#         # print((sdf_stereo < 0).sum())
#         # print((sdf < 0).sum())
   
#         del features, tof_context_features, stereo_context_features

#         output = dict()
#         output['tsdf'] = sdf.squeeze()
#         output['stereo_init'] = center_features_stereo[:, 1] > 0
#         output['tof_init'] = center_features_tof[:, 1] > 0
#         output['tsdf_tof'] = sdf_tof.squeeze()
#         output['tsdf_stereo'] = sdf_stereo.squeeze()
#         output['occ_tof'] = occ_tof.squeeze()
#         output['occ_stereo'] = occ_stereo.squeeze()

#         if len(self.config.DATA.input) > 1 and self.config.SETTINGS.test_mode:
#             output['sensor_weighting'] = alpha.squeeze()

#         del center_features_tof, center_features_stereo
#         return output






