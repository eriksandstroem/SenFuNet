import torch

import numpy as np

class TranslationNet(torch.nn.Module):

    def __init__(self, config):

        super(TranslationNet, self).__init__()

        self.config = config

        self.n_features = 2 * (config.FEATURE_MODEL.n_features + 1) # + 1 for weight 2 because two sensors. This is when only feeding 
        # features to translation network. No tsdf values are fed to the translation network

        self.output_scale = config.TRANSLATION_MODEL.output_scale

        activation = eval(config.TRANSLATION_MODEL.activation)

        self.layer_context = torch.nn.Sequential(torch.nn.Linear((self.config.TRANSLATION_MODEL.neighborhood ** 3) * self.n_features, self.n_features),
                                                  torch.nn.LayerNorm([self.n_features], elementwise_affine=False),
                                                  activation)

        self.layer1 = torch.nn.Sequential(
            #torch.nn.Linear(3 + self.n_features + self.n_features, 64),
            torch.nn.Linear(self.n_features + self.n_features, 64),
            activation)

        self.layer2 = torch.nn.Sequential(
            # torch.nn.Linear(3 + self.n_features + 64, 32),
            torch.nn.Linear(self.n_features + 64 , 32),
            activation)

        self.layer3 = torch.nn.Sequential(
            #torch.nn.Linear(3 + self.n_features + 32, 16),
            torch.nn.Linear(self.n_features + 32, 16),
            activation)

        self.layer4 = torch.nn.Sequential(
            #torch.nn.Linear(3 + self.n_features + 16, self.n_features),
            torch.nn.Linear(self.n_features + 16, self.n_features),
            activation)

        self.sdf_head = torch.nn.Sequential(torch.nn.Linear(self.n_features, 1),
                                                torch.nn.Tanh())

        if self.config.TRANSLATION_MODEL.occ_head:
            self.occ_head = torch.nn.Sequential(torch.nn.Linear(self.n_features, 1),
                                                    torch.nn.Sigmoid())

        self.feature_dropout = torch.nn.Dropout2d(p=0.2)

        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.hardtanh = torch.nn.Hardtanh(min_val=-0.06, max_val=0.06)
        self.softsign = torch.nn.Softsign()


    def forward(self, neighborhood):
        tof_features = neighborhood['tof']
        stereo_features = neighborhood['stereo']

        features = torch.cat((tof_features, stereo_features), dim=2)

        if self.config.TRANSLATION_MODEL.neighborhood == 3:
            center_features = torch.cat((tof_features[:, 13, :], stereo_features[:, 13, :]), dim=1)
        else: # Neighborhood is 5
            center_features = torch.cat((tof_features[:, 62, :], stereo_features[:, 62, :]), dim=1)

        # if self.config.TRANSLATION_MODEL.count == 'absolute':
        #     pass
        # elif self.config.TRANSLATION_MODEL.count == 'normalized':
        #     neighbourhood_count = features[:, 0, :].unsqueeze_(1)
        #     center_count = center_features[:, 0, :].unsqueeze_(1)
        #     print(neighbourhood_count.shape)
        #     print(center_count.shape)

        #     max_count_neighbourhood = torch.max(neighbourhood_count)
        #     max_count_center = torch.max(center_count)
        #     max_count = torch.max(max_count_neighbourhood, max_count_center) + 1.e-09

        #     features = torch.cat([features[:, :, 1:],
        #                           neighbourhood_count/max_count], dim=1)
        #     center_features = torch.cat([center_features[:, :, 1:],
        #                                  center_count/max_count], dim=1)


        features = features.contiguous().view(features.shape[0], self.config.TRANSLATION_MODEL.neighborhood**3 * self.n_features)
        # center_features = center_features.squeeze_(-1)


        center_features = center_features.unsqueeze_(-1).unsqueeze_(-1)
        center_features = self.feature_dropout(center_features)
        center_features = center_features.squeeze_(-1).squeeze_(-1)

        context_features = self.layer_context(features)

        input_features = torch.cat([center_features, context_features], dim=1)

        features = self.layer1(input_features)
        features = torch.cat([center_features, features], dim=1)

        features = self.layer2(features)
        features = torch.cat([center_features, features], dim=1)

        features = self.layer3(features)
        features = torch.cat([center_features, features], dim=1)

        features = self.layer4(features)

        sdf = self.output_scale * self.sdf_head(features)

        occ = self.occ_head(features)
   
        del features, context_features, center_features

        return sdf, occ






