import torch
import datetime
import time

from modules.routing import ConfidenceRouting
from modules.extractor import Extractor
from modules.model import FusionNet
from modules.model_features import FeatureNet
from modules.integrator import Integrator
from modules.filtering_net import *
import math
import numpy as np
from scipy import ndimage

class Pipeline(torch.nn.Module):

    def __init__(self, config):

        super(Pipeline, self).__init__()

        self.config = config

        if config.ROUTING.do:
            # define model
            Cin = 0
            if config.DATA.input == 'multidepth':
                if config.DATA.fusion_strategy == 'routingNet':
                    Cin += 2
                else:
                    Cin += 1
            else:
                Cin += 1
            if config.DATA.intensity_grad:
                Cin += 2

            if config.DATA.fusion_strategy == 'routingNet':
                self._routing_network = ConfidenceRouting(Cin=Cin,
                                  F=config.ROUTING_MODEL.contraction,
                                  Cout=config.ROUTING_MODEL.n_output_channels,
                                  depth=config.ROUTING_MODEL.depth,
                                  batchnorms=config.ROUTING_MODEL.normalization)
            else:
                self._routing_network_mono = ConfidenceRouting(Cin=Cin,
                                  F=config.ROUTING_MODEL.contraction,
                                  Cout=config.ROUTING_MODEL.n_output_channels,
                                  depth=config.ROUTING_MODEL.depth,
                                  batchnorms=config.ROUTING_MODEL.normalization)
                self._routing_network_stereo = ConfidenceRouting(Cin=Cin,
                                  F=config.ROUTING_MODEL.contraction,
                                  Cout=config.ROUTING_MODEL.n_output_channels,
                                  depth=config.ROUTING_MODEL.depth,
                                  batchnorms=config.ROUTING_MODEL.normalization)
                self._routing_network_tof = ConfidenceRouting(Cin=Cin,
                                  F=config.ROUTING_MODEL.contraction,
                                  Cout=config.ROUTING_MODEL.n_output_channels,
                                  depth=config.ROUTING_MODEL.depth,
                                  batchnorms=config.ROUTING_MODEL.normalization)

        else:
            self._routing_network = None

        config.FUSION_MODEL.fusion_strategy = config.DATA.fusion_strategy
        if config.DATA.truncation_strategy == 'standard':
            config.FUSION_MODEL.init_value = -config.DATA.init_value
        elif config.DATA.truncation_strategy == 'artificial':
            config.FUSION_MODEL.init_value = config.DATA.init_value           

        self._extractor_stereo = Extractor(config.FUSION_MODEL, 'stereo')
        self._extractor_tof = Extractor(config.FUSION_MODEL, 'tof')
        config.FEATURE_MODEL.n_points_tof = config.FUSION_MODEL.n_points_tof
        config.FEATURE_MODEL.n_points_stereo = config.FUSION_MODEL.n_points_stereo
        config.FEATURE_MODEL.n_tail_points_tof = config.FUSION_MODEL.n_tail_points_tof
        config.FEATURE_MODEL.n_tail_points_stereo = config.FUSION_MODEL.n_tail_points_stereo
        if config.DATA.fusion_strategy == 'two_fusionNet':
            # self._fusion_network_mono = FusionNet(config.FUSION_MODEL)
            self._fusion_network_stereo = FusionNet(config.FUSION_MODEL, 'stereo')
            self._fusion_network_tof = FusionNet(config.FUSION_MODEL, 'tof')
            self._feature_network_stereo = FeatureNet(config.FEATURE_MODEL, 'stereo')
            self._feature_network_tof = FeatureNet(config.FEATURE_MODEL, 'tof')
        else:
            self._fusion_network = FusionNet(config.FUSION_MODEL)
            self._feature_network = FusionNet(config.FEATURE_MODEL)
        config.FUSION_MODEL.train_on_border_voxels = config.FILTERING_MODEL.train_on_border_voxels
        self._integrator = Integrator(config.FUSION_MODEL)

        if config.FILTERING_MODEL.fuse_sensors:
            self._filtering_network = FilteringNetFuseSensors(config)
        else:
            self._filtering_network = FilteringNet(config)

    def _routing(self, data):
        if self.config.DATA.input == 'multidepth' and self.config.DATA.fusion_strategy == 'routingNet':
            depth_tof = data['tof_depth'].unsqueeze_(1)
            # depth_mono = data['mono_depth'].unsqueeze_(1)
            depth_stereo = data['stereo_depth'].unsqueeze_(1)
            # inputs = torch.cat((depth_tof, depth_mono, depth_stereo), 1)
            inputs = torch.cat((depth_tof, depth_stereo), 1)
            inputs = inputs.to(self.device)
            est = self._routing_network.forward(inputs)
            frame = est[:, 0, :, :]
            confidence = torch.exp(-1. * est[:, 1, :, :])
        elif self.config.DATA.input == 'multidepth' and (self.config.DATA.fusion_strategy == 'fusionNet' or self.config.DATA.fusion_strategy == 'two_fusionNet' or self.config.DATA.fusion_strategy == 'fusionNet_conditioned'):
            inputs = data['depth']
            inputs = inputs.to(self.device)
            inputs = inputs.unsqueeze_(1) # add number of channels
            est = eval(data['routing_net']).forward(inputs)
            frame = est[:, 0, :, :]
            confidence = torch.exp(-1. * est[:, 1, :, :])
        else:
            inputs = data[self.config.DATA.input]
            inputs = inputs.to(self.device)
            inputs = inputs.unsqueeze_(1) # add number of channels

            est = self._routing_network.forward(inputs)
            frame = est[:, 0, :, :]
            confidence = torch.exp(-1. * est[:, 1, :, :])

        return frame, confidence

    def _fusion(self, input_, input_features, values, fusionNet=None, featureNet=None):

        b, c, h, w = input_.shape
        if self.config.FUSION_MODEL.fixed:
            with torch.no_grad():
                tsdf_pred = eval(fusionNet).forward(input_)
        else:
            tsdf_pred = eval(fusionNet).forward(input_)

        feat_pred =  eval(featureNet).forward(input_features)

        tsdf_pred = tsdf_pred.permute(0, 2, 3, 1)

        feat_pred = feat_pred.permute(0, 2, 3, 1) # (1, 256, 256, n_points*n_features)


        output = dict()


        n_points = eval('self.config.FUSION_MODEL.n_points_' + fusionNet.split('_')[-1])
        tsdf_est = tsdf_pred.view(b, h * w, n_points)
        feature_est = feat_pred.view(b, h * w, 1, self.config.FEATURE_MODEL.n_features)
        feature_est = feature_est.repeat(1, 1, n_points, 1)



        # computing weighted updates for loss calculation
        tsdf_old = values['fusion_values'] # values that were used as input to the fusion net
        weights_old = values['fusion_weights'] # weights that were used as input to the fusion net

        tsdf_new = torch.clamp(tsdf_est,
                               -self.config.DATA.trunc_value,
                               self.config.DATA.trunc_value)

        tsdf_fused = (weights_old * tsdf_old + tsdf_new) / (
                    weights_old + torch.ones_like(weights_old))

        output['tsdf_est'] = tsdf_est # output from fusion net (used for integration into voxel grid)
        output['tsdf_fused'] = tsdf_fused # fused version at the floating point location (we supervise on this) i.e. we supervise in floating coordinate voxel space on the fused values
        output['feature_est'] = feature_est # output from fusion net (used for integration into voxel grid)
        
        return output

    def _filtering(self, neighborhood): 
        if self.config.FILTERING_MODEL.fixed:
            with torch.no_grad():
                output = self._filtering_network.forward(neighborhood)
        else:
            output = self._filtering_network.forward(neighborhood)

        return output

    def _translation(self, neighborhood): 
        if self.config.TRANSLATION_MODEL.fixed:
            with torch.no_grad():
                tsdf, occ = self._translation_network.forward(neighborhood)
        else:
            tsdf, occ = self._translation_network.forward(neighborhood)

        return tsdf, occ

    def _prepare_fusion_input(self, frame, values_sensor, values_sensor_op, confidence=None, n_points=None, rgb=None):

        # get frame shape
        b, h, w = frame.shape

        # extracting data
        # reshaping data
        tsdf_input_sensor = values_sensor['fusion_values'].view(b, h, w, n_points)
        tsdf_weights_sensor = values_sensor['fusion_weights'].view(b, h, w, n_points)

        tsdf_input_sensor_op = values_sensor_op['fusion_values'].view(b, h, w, n_points)
        tsdf_weights_sensor_op = values_sensor['fusion_weights'].view(b, h, w, n_points)

        tsdf_frame = torch.unsqueeze(frame, -1)


        feature_input = tsdf_frame # torch.cat([tsdf_frame, feature_weights_sensor, features], dim=3)
        if rgb is not None:
            rgb = rgb.unsqueeze(-1)
            rgb = rgb.view(1, h, w, -1)
            feature_input = torch.cat((feature_input, rgb), dim=3)
        del rgb
        # del features, feature_weights_sensor
        # permuting input
        feature_input = feature_input.permute(0, -1, 1, 2)

        # stacking input data
        if self.config.FUSION_MODEL.with_peek:
            if self.config.FUSION_MODEL.confidence:
                assert confidence is not None
                tsdf_confidence = torch.unsqueeze(confidence, -1)
                tsdf_input = torch.cat([tsdf_frame, tsdf_confidence, tsdf_weights_sensor, tsdf_input_sensor,
                   tsdf_weights_sensor_op, tsdf_input_sensor_op], dim=3)
                del tsdf_confidence
            else:
                tsdf_input = torch.cat([tsdf_frame, tsdf_weights_sensor, tsdf_input_sensor,
                   tsdf_weights_sensor_op, tsdf_input_sensor_op], dim=3)

        else:
            if self.config.FUSION_MODEL.confidence:
                assert confidence is not None
                tsdf_confidence = torch.unsqueeze(confidence, -1)
                tsdf_input = torch.cat([tsdf_frame, tsdf_confidence, tsdf_weights_sensor, tsdf_input_sensor], dim=3)
                del tsdf_confidence
            else:
                tsdf_input = torch.cat([tsdf_frame, tsdf_weights_sensor, tsdf_input_sensor], dim=3)

        # permuting input
        tsdf_input = tsdf_input.permute(0, -1, 1, 2)

        del tsdf_frame

        return tsdf_input, feature_input

    def _prepare_volume_update(self, values, est, features, inputs, sensor) -> dict:

        tail_points = eval('self.config.FUSION_MODEL.n_tail_points_' + sensor)

        b, h, w = inputs.shape
        # print(b, h, w)
        depth = inputs.view(b, h * w, 1)

        valid = (depth != 0.)
        valid = valid.nonzero()[:, 1]

        # remove border voxels belonging to rays that are at the edge of the image in order to
        # better train the filtering network on the real outliers. This step assumes that we do
        # erosion on the weight grid to get rid of the 2nd surface.
        # nope this did nothing since we are anyway not using the 10 pixels around the boundary. I need to erode here instead!
        valid_filter = inputs[0, :, :].cpu().numpy()
        valid_filter = (valid_filter != 0.)
        # print(valid_filter.sum())
        # print('valid bef erode: ', (valid_filter > 0).sum())

        valid_filter = ndimage.binary_erosion(valid_filter, structure=np.ones((3,3)), iterations=1)
        # print('valid aft erode: ', (valid_filter > 0).sum())

        valid_filter = torch.tensor(valid_filter).unsqueeze(0)

        valid_filter = valid_filter.view(b, h * w, 1)

        
        valid_filter = valid_filter.nonzero()[:, 1]
        # print(valid_filter.shape)
        # print(valid_filter.shape)
        # it appears that idx is always 0, it should not be. It appears that the 8 weights do not sum to one! This is wrong!
        # print(values['indices'].shape)
        # max_w, idxs = torch.max(values['weights'], dim=-1)
        # print(idxs.shape)
        # print('idx', idxs[0, 0, 0])
        # print('sum idx', idxs.sum())
        # print('weigt', values['weights'][0, 0, 0, :])
        # print('ind', values['indices'][0, 0, 0, idxs[0, 0, 0], :])
        # print('featind', values['feature_indices'][0, 0, 0, 0, :])
        update_indices = values['indices'][:, valid, :tail_points, :, :]
        # filter_weights = values['weights'][:, valid_filter, :(tail_points - math.floor(self.config.FILTERING_MODEL.neighborhood/2)), :] # this is only to do what?
        update_weights = values['weights'][:, valid, :tail_points, :]
        update_indices_empty = values['indices_empty'][:, valid, :, :, :] # indices of the voxel vertices we want to update
        update_weights_empty = values['weights_empty'][:, valid, :, :] # wei
        # update_indices_empty_behind = values['indices_empty_behind'][:, valid, :, :, :] # indices of the voxel vertices we want to update
        # update_weights_empty_behind = values['weights_empty_behind'][:, valid, :, :] # wei
        # print(update_weights_empty.shape)

        update_values = est[:, valid, :tail_points]

        update_values = torch.clamp(update_values,
                                    -self.config.DATA.trunc_value,
                                    self.config.DATA.trunc_value)

        # we do not need to compute an update_feature_weights variable for the features since they are all one at the indices in question.
        # we need to compute an update_feature_indices however
        update_features = features[:, valid, :tail_points, :]
        update_feature_indices = values['feature_indices'][:, valid, :tail_points, :, :]
        filter_indices = values['feature_indices'][:, valid_filter, :(tail_points - 2), :, :] # quite randomly chosen number 2 here. I might not need it even.

        del valid

        # packing
        output = dict(update_values=update_values,
                      update_features=update_features,
                      update_weights=update_weights,
                      update_indices=update_indices,
                      update_feature_indices=update_feature_indices,
                      filter_indices=filter_indices,
                      update_indices_empty=update_indices_empty,
                      update_weights_empty=update_weights_empty)

        return output

    def _prepare_input_training(self, tsdf_volume, weights_volume, feat_vol, bbox) -> dict:

        output = dict()

        # print(bbox)
        # Now we have a bounding box which we know have valid indices. Now it is time to extract this volume from the input grid,
        # which is already on the gpu

        # tsdf
        tsdf_neighborhood = tsdf_volume[bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]
        # features
        feat_neighborhood = feat_vol[bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1], :]
        feat_neighborhood = feat_neighborhood.contiguous().view(-1, feat_neighborhood.shape[0], feat_neighborhood.shape[1], feat_neighborhood.shape[2])

        # print(neighborhood.shape)
        del tsdf_volume, feat_vol
        
        neighborhood_weights = weights_volume[bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]
        del weights_volume


        neighborhood = torch.cat([tsdf_neighborhood.unsqueeze_(0), neighborhood_weights.unsqueeze_(0), feat_neighborhood], dim=0) # shape (C, D, H ,W)

        output['neighborhood'] = neighborhood.unsqueeze_(0).float() # add batch dimension
        del neighborhood

        return output

    def _prepare_sensor_fusion_input(self, database, bbox) -> dict:
        # TODO: do reflection padding if the central voxel is at the grid boundary. For now I do constant padding
        output = dict()
        neighborhood_filter = dict()

        volume = database[scene] # this yields tensors for the database. I suppose I need to use this

        neighborhood_tsdf = self.config.FUSION_MODEL.init_value*torch.ones(tuple(bbox[:, 1] - bbox[:, 0]))

        # even if we exceed the volume dimensions, we will only retrieve the largest possible box
        extract_tsdf = volume['tsdf_tof'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]

        # if extract_tsdf is smaller than neighborhoo_tsdf we need to position it at the correct indices
        neighborhood_tsdf[0:extract_tsdf.shape[0],
                        0:extract_tsdf.shape[1],
                        0:extract_tsdf.shape[2]] = extract_tsdf

        neighborhood_weights = torch.zeros(tuple(bbox[:, 1] - bbox[:, 0]))

        extract_weights = volume['weights_tof'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]

        neighborhood_weights[0:extract_tsdf.shape[0],
                        0:extract_tsdf.shape[1],
                        0:extract_tsdf.shape[2]] = extract_weights

        feat_neighborhood = torch.zeros(tuple(bbox[:, 1] - bbox[:, 0]))
        feat_neighborhood = feat_neighborhood.unsqueeze_(-1)

        extract_feat = volume['features_tof'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1], :]

        feat_neighborhood[0:extract_feat.shape[0],
                        0:extract_tsdf.shape[1],
                        0:extract_tsdf.shape[2]] = extract_feat

        feat_neighborhood = feat_neighborhood.view(-1, feat_neighborhood.shape[0], feat_neighborhood.shape[1], feat_neighborhood.shape[2])

        neighborhood_filter['tof'] = torch.cat((neighborhood_tsdf.unsqueeze(0), neighborhood_weights.unsqueeze(0), feat_neighborhood), dim=0).unsqueeze(0).to(self.device) # add batch dimension

        neighborhood_tsdf = self.config.FUSION_MODEL.init_value*torch.ones(tuple(bbox[:, 1] - bbox[:, 0]))

        # even if we exceed the volume dimensions, we will only retrieve the largest possible box
        extract_tsdf = volume['tsdf_stereo'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]

        # if extract_tsdf is smaller than neighborhoo_tsdf we need to position it at the correct indices
        neighborhood_tsdf[0:extract_tsdf.shape[0],
                        0:extract_tsdf.shape[1],
                        0:extract_tsdf.shape[2]] = extract_tsdf

        neighborhood_weights = torch.zeros(tuple(bbox[:, 1] - bbox[:, 0]))

        extract_weights = volume['weights_stereo'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]

        neighborhood_weights[0:extract_tsdf.shape[0],
                        0:extract_tsdf.shape[1],
                        0:extract_tsdf.shape[2]] = extract_weights

        feat_neighborhood = torch.zeros(tuple(bbox[:, 1] - bbox[:, 0]))
        feat_neighborhood = feat_neighborhood.unsqueeze_(-1)

        extract_feat = volume['features_stereo'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1], :]

        feat_neighborhood[0:extract_feat.shape[0],
                        0:extract_tsdf.shape[1],
                        0:extract_tsdf.shape[2]] = extract_feat

        feat_neighborhood = feat_neighborhood.view(-1, feat_neighborhood.shape[0], feat_neighborhood.shape[1], feat_neighborhood.shape[2])

        neighborhood_filter['stereo'] = torch.cat((neighborhood_tsdf.unsqueeze(0), neighborhood_weights.unsqueeze(0), feat_neighborhood), dim=0).unsqueeze(0).to(self.device) # add batch dimension


        output['neighborhood_filter'] = neighborhood_filter


        del neighborhood_filter

        return output

    def _prepare_local_grids(self, bbox, database, scene):
        # extract bbox from global grid
        tsdf_tof = database[scene]['tsdf_tof'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]

        weights_tof = database[scene]['weights_tof'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]

        feat_tof = database[scene]['features_tof'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1], :]

        feat_tof = feat_tof.contiguous().view(-1, feat_tof.shape[0], feat_tof.shape[1], feat_tof.shape[2])

        tsdf_stereo = database[scene]['tsdf_stereo'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]

        weights_stereo = database[scene]['weights_stereo'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1]]

        feat_stereo = database[scene]['features_stereo'][bbox[0, 0]:bbox[0, 1],
                bbox[1, 0]:bbox[1, 1],
                bbox[2, 0]:bbox[2, 1], :]

        feat_stereo = feat_stereo.contiguous().view(-1, feat_stereo.shape[0], feat_stereo.shape[1], feat_stereo.shape[2])
        # print(feat_stereo.shape)

        # pad the local grids so that the dimension is divisible by half the chunk size and then ad 
        # the chunk size divided by 4 so that we can update only the central region
        divisble_by = self.config.FILTERING_MODEL.chunk_size/2
        if (bbox[0, 1] - bbox[0, 0]) % divisble_by != 0: # I use 8 here because we need at least 4 due to the box_shift variable regardless
            # of network_depth
            pad_x = divisble_by - \
            (bbox[0, 1] - bbox[0, 0]) % divisble_by
        else:
            pad_x = 0

        if (bbox[1, 1] - bbox[1, 0]) % divisble_by != 0:
            pad_y = divisble_by - \
            (bbox[1, 1] - bbox[1, 0]) % divisble_by
        else:
            pad_y = 0 

        if (bbox[2, 1] - bbox[2, 0]) % divisble_by != 0:
            pad_z = divisble_by - \
            (bbox[2, 1] - bbox[2, 0]) % divisble_by
        else:
            pad_z = 0

        # concatenate the two grids and make them on the form N, C, D, H, W
        local_grid_tof = torch.cat((tsdf_tof.unsqueeze(0), weights_tof.unsqueeze(0), feat_tof), dim=0).unsqueeze(0) # add batch dimension
        local_grid_stereo = torch.cat((tsdf_stereo.unsqueeze(0), weights_stereo.unsqueeze(0), feat_stereo), dim=0).unsqueeze(0) # add batch dimension

        # pad the local grid
        pad = torch.nn.ReplicationPad3d((0, int(pad_z), 0, int(pad_y), 0, int(pad_x)))
        local_grid_tof = pad(local_grid_tof.float())
        local_grid_stereo = pad(local_grid_stereo.float())


        # pad the grid with the chunk size divided by 4 along each dimension
        extra_pad = int(self.config.FILTERING_MODEL.chunk_size/4)
        pad = torch.nn.ReplicationPad3d(extra_pad)
        local_grid_tof = pad(local_grid_tof.float())
        local_grid_stereo = pad(local_grid_stereo.float())

        output = dict()
        output['tof'] = local_grid_tof
        output['stereo'] = local_grid_stereo
        return output, int(pad_x), int(pad_y), int(pad_z)

    def fuse(self,
             batch,
             database,
             device):

        self.device = device

        # routing
        if self.config.ROUTING.do:
            frame, confidence = self._routing(batch)

            if self.config.DATA.input == 'multidepth' and (self.config.DATA.fusion_strategy == 'fusionNet' or self.config.DATA.fusion_strategy == 'two_fusionNet' or self.config.DATA.fusion_strategy == 'fusionNet_conditioned'):
                filtered_frame = frame.detach().clone()
                filtered_frame[confidence < batch['confidence_threshold']] = 0
            else:
                filtered_frame = frame.detach().clone()
                filtered_frame[confidence < self.config.ROUTING.threshold] = 0


        else:
            frame = batch[batch['sensor'] + '_depth'].squeeze_(1)
            frame = frame.to(device)
            confidence = None

        if self.config.FEATURE_MODEL.w_rgb:
            rgb = batch['image'].squeeze().to(device)
        elif self.config.FEATURE_MODEL.w_intensity_gradient:
            i = batch['intensity'].squeeze()
            g = batch['gradient'].squeeze()
            rgb = torch.cat((i, g), dim=0).to(device)
        else:
            rgb = None
        

        mask = batch['mask'].to(device)
        if self.config.ROUTING.do:
            filtered_frame = torch.where(mask == 0, torch.zeros_like(frame),
                                    filtered_frame)
        else:
            filtered_frame = torch.where(mask == 0, torch.zeros_like(frame),
                                    frame)

        # get current tsdf values
        scene_id = batch['frame_id'][0].split('/')[0]

        values_sensor = eval('self._extractor_' + batch['sensor']).forward(frame,
                                         batch['extrinsics'],
                                         batch['intrinsics' + '_' + batch['sensor']],
                                         database[scene_id]['tsdf' + '_' + batch['sensor']],
                                         database[scene_id]['features_' + batch['sensor']],
                                         database[scene_id]['origin'],
                                         database[scene_id]['resolution'],
                                         self.config.SETTINGS.gpu,
                                         database[scene_id]['weights' + '_' + batch['sensor']],
                                         database[scene_id]['feature_weights' + '_' + batch['sensor']])

        values_sensor_op = eval('self._extractor_' + batch['sensor']).forward(frame,
                                         batch['extrinsics'],
                                         batch['intrinsics' + '_' + batch['sensor']],
                                         database[scene_id]['tsdf' + '_' + batch['sensor_opposite']],
                                         database[scene_id]['features_' + batch['sensor_opposite']],
                                         database[scene_id]['origin'],
                                         database[scene_id]['resolution'],
                                         self.config.SETTINGS.gpu,
                                         database[scene_id]['weights' + '_' + batch['sensor_opposite']],
                                         database[scene_id]['feature_weights' + '_' + batch['sensor_opposite']])

        n_points = eval('self.config.FUSION_MODEL.n_points_' + batch['sensor'])
        tsdf_input, feature_input = self._prepare_fusion_input(frame, values_sensor, values_sensor_op,
                                                              confidence, n_points, rgb)
        del rgb, frame


        fusion_output = self._fusion(tsdf_input, feature_input, values_sensor, batch['fusion_net'], batch['feature_net'])

        # masking invalid losses
        tsdf_est = fusion_output['tsdf_est']
        feature_est = fusion_output['feature_est']

        integrator_input = self._prepare_volume_update(values_sensor,
                                                        tsdf_est,
                                                        feature_est,
                                                        filtered_frame,
                                                        batch['sensor'])

        values, features, weights, feature_weights, indices = self._integrator.forward(integrator_input,
                                                   database[scene_id][
                                                       'tsdf_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'features_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'weights_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'feature_weights_' + batch['sensor']].to(device))


        del indices, integrator_input

        if batch['sensor'] == 'tof':
            database.tsdf_tof[
                scene_id].volume = values.cpu().detach().numpy()
            database.fusion_weights_tof[
                scene_id] = weights.cpu().detach().numpy()
            database.features_tof[
                scene_id] = features.cpu().detach().numpy()
            database.feature_weights_tof[
                scene_id] = feature_weights.cpu().detach().numpy()
        else: # stereo
            database.tsdf_stereo[
                scene_id].volume = values.cpu().detach().numpy()
            database.fusion_weights_stereo[
                scene_id] = weights.cpu().detach().numpy()
            database.features_stereo[
                scene_id] = features.cpu().detach().numpy()
            database.feature_weights_stereo[
                scene_id] = feature_weights.cpu().detach().numpy()

        del values, weights, features, feature_weights

        return

    def fuse_training(self, batch, database, device):

        """
            Learned real-time depth map fusion pipeline

            :param batch:
            :param extractor:
            :param routing_model:
            :param tsdf_model:
            :param database:
            :param device:
            :param config:
            :param routing_config:
            :param mode:
            :return:
            """
        output = dict()

        self.device = device

        # routing
        if self.config.ROUTING.do:
            with torch.no_grad():
                frame, confidence = self._routing(batch)
            if self.config.DATA.input == 'multidepth' and (self.config.DATA.fusion_strategy == 'fusionNet' or self.config.DATA.fusion_strategy == 'two_fusionNet' or self.config.DATA.fusion_strategy == 'fusionNet_conditioned'):
                filtered_frame = frame.detach().clone()
                filtered_frame[confidence < batch['confidence_threshold']] = 0
            else:
                filtered_frame = frame.detach().clone()
                filtered_frame[confidence < self.config.ROUTING.threshold] = 0
            # frame = frame.to(device)
        else:
            frame = batch[batch['sensor'] + '_depth'].squeeze_(1)
            frame = frame.to(device) # putting extractor on gpu
            confidence = None

        if self.config.FEATURE_MODEL.w_rgb:
            rgb = batch['image'].squeeze().to(device)
        elif self.config.FEATURE_MODEL.w_intensity_gradient:
            i = batch['intensity'].squeeze()
            g = batch['gradient'].squeeze()
            rgb = torch.cat((i, g), dim=0).to(device)
        else:
            rgb = None

        mask = batch['mask'].to(device) # putting extractor on gpu
        # mask = batch['mask']
        if self.config.ROUTING.do:
            filtered_frame = torch.where(mask == 0, torch.zeros_like(frame),
                                    filtered_frame)
        else:
            filtered_frame = torch.where(mask == 0, torch.zeros_like(frame),
                                    frame)
        del mask

        b, h, w = frame.shape

        # get current tsdf values
        scene_id = batch['frame_id'][0].split('/')[0]


        values_sensor = eval('self._extractor_' + batch['sensor']).forward(frame,
                                         batch['extrinsics'],
                                         batch['intrinsics' + '_' + batch['sensor']],
                                         database[scene_id]['tsdf' + '_' + batch['sensor']],
                                         database[scene_id]['features_' + batch['sensor']],
                                         database[scene_id]['origin'],
                                         database[scene_id]['resolution'],
                                         self.config.SETTINGS.gpu,
                                         database[scene_id]['weights' + '_' + batch['sensor']],
                                         database[scene_id]['feature_weights' + '_' + batch['sensor']])

        values_sensor_op = eval('self._extractor_' + batch['sensor']).forward(frame,
                                         batch['extrinsics'],
                                         batch['intrinsics' + '_' + batch['sensor']],
                                         database[scene_id]['tsdf' + '_' + batch['sensor_opposite']],
                                         database[scene_id]['features_' + batch['sensor_opposite']],
                                         database[scene_id]['origin'],
                                         database[scene_id]['resolution'],
                                         self.config.SETTINGS.gpu,
                                         database[scene_id]['weights' + '_' + batch['sensor_opposite']],
                                         database[scene_id]['feature_weights' + '_' + batch['sensor_opposite']])


        # TODO: make function that extracts only the gt values for speed up during training
        values_gt = eval('self._extractor_' + batch['sensor']).forward(frame,
                                         batch['extrinsics'],
                                         batch['intrinsics' + '_' + batch['sensor']],
                                         database[scene_id]['gt'],
                                         database[scene_id]['features_' + batch['sensor']],
                                         database[scene_id]['origin'],
                                         database[scene_id]['resolution'],
                                         self.config.SETTINGS.gpu,
                                         database[scene_id]['weights_' + batch['sensor']],
                                         database[scene_id]['feature_weights_' + batch['sensor']])

        tsdf_target = values_gt['fusion_values']
        del values_gt

        n_points = eval('self.config.FUSION_MODEL.n_points_' + batch['sensor'])
        tsdf_input, feature_input = self._prepare_fusion_input(frame, values_sensor, values_sensor_op, confidence, n_points, rgb)
        del rgb, frame
        
        # tsdf_target = tsdf_target.view(b, h, w, eval('self.config.FUSION_MODEL.n_points_' + batch['sensor']))

        if self.config.DATA.fusion_strategy == 'two_fusionNet':
            fusion_output = self._fusion(tsdf_input, feature_input, values_sensor, batch['fusion_net'], batch['feature_net'])
        else:
            fusion_output = self._fusion(tsdf_input, feature_input, values_sensor)

        del tsdf_input, feature_input

        # reshaping target
        tsdf_target = tsdf_target.view(b, h * w, n_points)

        # masking invalid losses
        tsdf_est = fusion_output['tsdf_est']
        feature_est = fusion_output['feature_est']
        tsdf_fused = fusion_output['tsdf_fused']
        del fusion_output


        tsdf_fused = masking(tsdf_fused, filtered_frame.view(b, h * w, 1))
        tsdf_target = masking(tsdf_target, filtered_frame.view(b, h * w, 1))

        output['tsdf_fused'] = tsdf_fused
        output['tsdf_target'] = tsdf_target
        del tsdf_fused, tsdf_target

        integrator_input = self._prepare_volume_update(values_sensor,
                                                        tsdf_est,
                                                        feature_est,
                                                        filtered_frame,
                                                        batch['sensor'])

        del values_sensor, values_sensor_op, tsdf_est, feature_est, filtered_frame

        values, features, weights, feature_weights, indices = self._integrator.forward(integrator_input,
                                                   database[scene_id][
                                                       'tsdf_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'features_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'weights_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'feature_weights_' + batch['sensor']].to(device))


        del integrator_input

        if batch['sensor'] == 'tof':
            database.tsdf_tof[
                scene_id].volume = values.cpu().detach().numpy()
            database.fusion_weights_tof[
                scene_id] = weights.cpu().detach().numpy()
            database.features_tof[
                scene_id] = features.cpu().detach().numpy()
            database.feature_weights_tof[
                scene_id] = feature_weights.cpu().detach().numpy()
        else: # stereo
            database.tsdf_stereo[
                scene_id].volume = values.cpu().detach().numpy()
            database.fusion_weights_stereo[
                scene_id] = weights.cpu().detach().numpy()
            database.features_stereo[
                scene_id] = features.cpu().detach().numpy()
            database.feature_weights_stereo[
                scene_id] = feature_weights.cpu().detach().numpy()


        # I found that sampling fewer indices did not matter for the overall performance so to speed up training
        # I use fewer indides than I can - I can use 50000

        # write here a new filter_training function which takes as input the grids and the indices in question. Inside
        # the function I loop over the window chunks and feed them to the filter network. No, I can only feed
        # one chunk because to free the gpu I need to do the backprop on that chunk first. 
        # The chunk which is selected I suppose is random within the minimum bounding box of the indices. 
        # What happens if the minimum bounding box is smaller than the chunk size in any dimension? Then the box will be
        # the size of the chunk so I need to check the validity of the indices afterall and then I would have to shrink the 
        # chunk size if there are invalid indices... Tricky. Perhaps easiest to just output None for tsdf filtered
        # and skip the backward pass by doing a continue statement.

        filtered_output = self._filter_training(indices, 
            database[scene_id]['gt'], values, 
            database[scene_id]['tsdf_' + batch['sensor_opposite']].to(device), 
            weights, 
            database[scene_id]['weights_' + batch['sensor_opposite']].to(device),
            features, 
            database[scene_id]['features_' + batch['sensor_opposite']].to(device),
            batch['sensor'],
            batch['sensor_opposite'],
            device)

        if filtered_output is not None:
            output['filtered_output'] = filtered_output
        else:
            return None

        del values, weights 

        return output

    def sensor_fusion_old(self, scene, database, device):
        self.device = device
        # why do I have astype in16? Memory saveing? Well yeah I suppose I can save a bool as int16
        indices = (database.feature_weights_tof[scene] > 0).astype(np.int16)
        # indices_and = np.logical_and(indices, database.fusion_weights_stereo[scene] > 0).astype(np.int16)
        # indices = np.logical_or(indices, database.feature_weights_stereo[scene] > 0).astype(np.int16) # - indices_and

        indices = np.transpose(indices.nonzero()).astype(np.int16)

        # for testing my idea of zero-initial volume to remove outliers and keep geometry
        # an attempt to not attend to uninitialized voxel values (though this should be handled by the 
        # weights which form a natural mask
        # idx  = (database.feature_weights_tof[scene] == 0).astype(np.int16) # changed here
        # idx = np.transpose(idx.nonzero()).astype(np.int16) # changed here
        # database.scenes_est_tof[scene].volume[idx[:, 0], idx[:, 1], idx[:, 2]] = 0 # changed here

        # print(indices.shape)
        # get minimum box size
        x_size = indices[:, 0].max() - indices[:, 0].min() 
        y_size = indices[:, 1].max() - indices[:, 1].min() 
        z_size = indices[:, 2].max() - indices[:, 2].min()
        
        bbox = np.array([[indices[:, 0].min(), indices[:, 0].min() + x_size], # + 1 here because we want to include the max index in the bbox because we later do min:max during extraction
                        [indices[:, 1].min(), indices[:, 1].min() + y_size], # + 1 here because we want to include the max index in the bbox because we later do min:max during extraction
                        [indices[:, 2].min(), indices[:, 2].min() + z_size]]) # + 1 here because we want to include the max index in the bbox because we later do min:max during extraction

        # traverse the global grid, extracting chunks from the global grid to feed to the 
        # filtering network one at a time.
        moving_bbox = np.zeros_like(bbox)
        # print(bbox)
        # database.filtered[scene].volume = database.scenes_est_tof[scene].volume

        # database.filtered[scene].volume[bbox[0, 0]:bbox[0, 1],
        #                                     bbox[1, 0]:bbox[1, 1],
        #                                     bbox[2, 0]:bbox[2, 1]] = database.scenes_est_tof[scene].volume[bbox[0, 0]:bbox[0, 1],
        #                                                                                         bbox[1, 0]:bbox[1, 1],
        #                                                                                         bbox[2, 0]:bbox[2, 1]] 
        for i in range(math.ceil(x_size/self.config.FILTERING_MODEL.chunk_size)): # multiply by 2 since we have a stride that is half the chunk size
            moving_bbox[0, 0] = bbox[0, 0] + i  * self.config.FILTERING_MODEL.chunk_size # multiply by 0.5 because the stride is half the chunk size
            moving_bbox[0, 1] = moving_bbox[0, 0] + self.config.FILTERING_MODEL.chunk_size 
            if moving_bbox[0, 1] > bbox[0, 1]:
                moving_bbox[0, 1] = bbox[0, 1]
            for j in range(math.ceil(y_size/self.config.FILTERING_MODEL.chunk_size)):
                moving_bbox[1, 0] = bbox[1, 0] + j * self.config.FILTERING_MODEL.chunk_size
                moving_bbox[1, 1] = moving_bbox[1, 0] + self.config.FILTERING_MODEL.chunk_size
                if moving_bbox[1, 1] > bbox[1, 1]:
                    moving_bbox[1, 1] = bbox[1, 1]
                for k in range(math.ceil(z_size/self.config.FILTERING_MODEL.chunk_size)):
                    moving_bbox[2, 0] = bbox[2, 0] + k * self.config.FILTERING_MODEL.chunk_size
                    moving_bbox[2, 1] = moving_bbox[2, 0] + self.config.FILTERING_MODEL.chunk_size
                    # print('mb:',moving_bbox)
                    if moving_bbox[2, 1] > bbox[2, 1]:
                        moving_bbox[2, 1] = bbox[2, 1]

                    if (moving_bbox[0, 1] - moving_bbox[0, 0]) % 2**self.config.FILTERING_MODEL.network_depth != 0:
                        moving_bbox[0, 1] += 2**self.config.FILTERING_MODEL.network_depth - \
                        (moving_bbox[0, 1] - moving_bbox[0, 0]) % 2**self.config.FILTERING_MODEL.network_depth
                    if (moving_bbox[1, 1] - moving_bbox[1, 0]) % 2**self.config.FILTERING_MODEL.network_depth != 0:
                        moving_bbox[1, 1] += 2**self.config.FILTERING_MODEL.network_depth - \
                        (moving_bbox[1, 1] - moving_bbox[1, 0]) % 2**self.config.FILTERING_MODEL.network_depth
                    if (moving_bbox[2, 1] - moving_bbox[2, 0]) % 2**self.config.FILTERING_MODEL.network_depth != 0:
                        moving_bbox[2, 1] += 2**self.config.FILTERING_MODEL.network_depth - \
                        (moving_bbox[2, 1] - moving_bbox[2, 0]) % 2**self.config.FILTERING_MODEL.network_depth

                    input_ = self._prepare_sensor_fusion_input(scene, database, moving_bbox)

                    # print(input_['neighborhood_filter']['tof'][0, 0, :, :, :].shape)
                    # print(())
                    with torch.no_grad():
                        # input_['neighborhood_filter']['tof'][0, 0, :, :, :] = -0.0*torch.ones_like(input_['neighborhood_filter']['tof'][0, 0, :, :, :])
                        # input_['neighborhood_filter']['tof'][0, 1, :, :, :] = 0.0*torch.ones_like(input_['neighborhood_filter']['tof'][0, 1, :, :, :])
                        sub_tsdf_filter = self._filtering(input_['neighborhood_filter'])
                        # sub_tsdf_filter = input_['neighborhood_filter']['tof'][0, 0, :, :, :] # debug
                        # print(sub_tsdf_filter)
                        # sub_tsdf_filter = 
                        # print(sub_tsdf_filter.shape)
                        # print(sub_tsdf_filter)
                    del input_

                    sub_tsdf = sub_tsdf_filter.cpu().detach().numpy().squeeze() 
                    del sub_tsdf_filter
                    sub_tsdf = np.clip(sub_tsdf,
                                    -self.config.DATA.trunc_value,
                                    self.config.DATA.trunc_value)

                    # print('asda', (sub_tsdf - input_['neighborhood_filter']['tof'][:, :, :, 0].cpu().detach().numpy().squeeze()).sum())
                    # even though we replace all values within the bounding box, thanks to the 
                    # weight mask, when we later compute the voxel grid metrics and run marching cubes
                    # on the grid, the uninitialized voxels will not be taken into account, but they will be taken 
                    # into account during trainign for the skimage marching cubes. Could be interesting to see.
                    # Here I clip the end of the moving_bbox since it could be outside the grid due to the 
                    # adaption that the shape is divisible by 2, 4 or 8 depending on network depth.
                    # print('mb0',moving_bbox)
                    # update only the central region of the moving bbox - when using stride half the chunk size
                    # box_shift = (moving_bbox[:, 1] - moving_bbox[:, 0])/4 # requires that shape is divisible by 4! I.e. may not work with network depth 2 - need to change code above
                    update_bbox = moving_bbox.copy()
                    # update_bbox[:, 0] = update_bbox[:, 0] + box_shift # this will always be inside grid because when we use chunk size 64 and network depth 8, we ill
                    # at most add 7 voxels that are invalid. On the other hand, the moving_box can take dimensions as low as 8x8x8 meaning that box_shift could be
                    # as low as 2 in the worst case. I.e. the clipping is still needed.
                    # update_bbox[:, 1] = update_bbox[:, 1] - box_shift
                    update_bbox[0, 1] = np.clip(update_bbox[0, 1], None, database.filtered[scene].volume.shape[0])
                    update_bbox[1, 1] = np.clip(update_bbox[1, 1], None, database.filtered[scene].volume.shape[1])
                    update_bbox[2, 1] = np.clip(update_bbox[2, 1], None, database.filtered[scene].volume.shape[2])
                    # only replace those values that belong to the grid. Without the line below, sub_tsdf could be too large
                    sub_tsdf_box = update_bbox[:, 1] - update_bbox[:, 0]
                    # print('sub_tsdf shape', sub_tsdf.shape)
                    # print('sub_tsdf valid shape', sub_tsdf[:sub_tsdf_box[0],
                    #                         :sub_tsdf_box[1],
                    #                         :sub_tsdf_box[2]].shape)
                    # print('mb1', moving_bbox)
                    # print('filtered shape valid idx', database.filtered[scene].volume[moving_bbox[0, 0]:moving_bbox[0, 1],
                                            # moving_bbox[1, 0]:moving_bbox[1, 1],
                                            # moving_bbox[2, 0]:moving_bbox[2, 1]].shape)
                    # when we do the central update, we could end up in a situation where the update region is outside the 
                    # voxel grid. This happens when the end of the box is smaller than the beginning of the box. In this case,
                    # do not update anything.
                    if not (update_bbox[0, 0] > update_bbox[0, 1] or update_bbox[1, 0] > update_bbox[1, 1] or update_bbox[2, 0] > update_bbox[2, 1]):
                        database.filtered[scene].volume[update_bbox[0, 0]:update_bbox[0, 1],
                                                update_bbox[1, 0]:update_bbox[1, 1],
                                                update_bbox[2, 0]:update_bbox[2, 1]] = sub_tsdf[:sub_tsdf_box[0],
                                                :sub_tsdf_box[1],
                                                :sub_tsdf_box[2]]
                    # old update - when using stride equals chunk size
                    # moving_bbox[0, 1] = np.clip(moving_bbox[0, 1], None, database.filtered[scene].volume.shape[0])
                    # moving_bbox[1, 1] = np.clip(moving_bbox[1, 1], None, database.filtered[scene].volume.shape[1])
                    # moving_bbox[2, 1] = np.clip(moving_bbox[2, 1], None, database.filtered[scene].volume.shape[2])
                    # # only replace those values that belong to the grid. Without the line below, sub_tsdf could be too large
                    # sub_tsdf_box = moving_bbox[:, 1] - moving_bbox[:, 0]
                    # database.filtered[scene].volume[moving_bbox[0, 0]:moving_bbox[0, 1],
                    #                         moving_bbox[1, 0]:moving_bbox[1, 1],
                    #                         moving_bbox[2, 0]:moving_bbox[2, 1]] = sub_tsdf[:sub_tsdf_box[0],
                    #                         :sub_tsdf_box[1],
                    #                         :sub_tsdf_box[2]]
                    del sub_tsdf
                    # break # debugging

    def sensor_fusion(self, scene, database, device): # here we use a stride which is half the chunk size
        self.device = device
        # why do I have astype in16? Memory saveing? Well yeah I suppose I can save a bool as int16
        indices = (database.feature_weights_tof[scene] > 0).astype(np.int16)
        # indices_and = np.logical_and(indices, database.fusion_weights_stereo[scene] > 0).astype(np.int16)
        indices = np.logical_or(indices, database.feature_weights_stereo[scene] > 0).astype(np.int16) # - indices_and

        indices = np.transpose(indices.nonzero()).astype(np.int16)
        chunk_size = self.config.FILTERING_MODEL.chunk_size

        # for testing my idea of zero-initial volume to remove outliers and keep geometry
        # an attempt to not attend to uninitialized voxel values (though this should be handled by the 
        # weights which form a natural mask
        # idx  = (database.feature_weights_tof[scene] == 0).astype(np.int16) # changed here
        # idx = np.transpose(idx.nonzero()).astype(np.int16) # changed here
        # database.scenes_est_tof[scene].volume[idx[:, 0], idx[:, 1], idx[:, 2]] = 0 # changed here

        # print(indices.shape)
        # get minimum box size
        x_size = indices[:, 0].max() - indices[:, 0].min() 
        y_size = indices[:, 1].max() - indices[:, 1].min() 
        z_size = indices[:, 2].max() - indices[:, 2].min()
        
        bbox = np.array([[indices[:, 0].min(), indices[:, 0].min() + x_size + 1], # + 1 here because we want to include the max index in the bbox because we later do min:max during extraction
                        [indices[:, 1].min(), indices[:, 1].min() + y_size + 1], # + 1 here because we want to include the max index in the bbox because we later do min:max during extraction
                        [indices[:, 2].min(), indices[:, 2].min() + z_size + 1]]) # + 1 here because we want to include the max index in the bbox because we later do min:max during extraction

        # prepare local grids
        local_grids, pad_x, pad_y, pad_z = self._prepare_local_grids(bbox, database, scene)

        filtered_local_grid = torch.zeros(tuple(local_grids['tof'][0, 0, :, :, :].shape))
        # print(filtered_local_grid.shape)
        # traverse the local grid, extracting chunks from the local grid to feed to the 
        # filtering network one at a time.
        moving_bbox = np.zeros_like(bbox)

        x_size, y_size, z_size = local_grids['tof'][0, 0, :, :, :].shape
        for i in range(math.ceil(2*x_size/chunk_size) - 1): # multiply by 2 since we have a stride that is half the chunk size
            moving_bbox[0, 0] = i * 0.5 * chunk_size # multiply by 0.5 because the stride is half the chunk size
            moving_bbox[0, 1] = moving_bbox[0, 0] + chunk_size 
            for j in range(math.ceil(2*y_size/chunk_size) - 1):
                moving_bbox[1, 0] = j * 0.5 * chunk_size
                moving_bbox[1, 1] = moving_bbox[1, 0] + chunk_size
                for k in range(math.ceil(2*z_size/chunk_size) - 1):
                    moving_bbox[2, 0] = k * 0.5 * chunk_size
                    moving_bbox[2, 1] = moving_bbox[2, 0] + chunk_size

                    input_ = dict()
                    input_['tof'] = local_grids['tof'][:, :, moving_bbox[0, 0]:moving_bbox[0, 1],
                                        moving_bbox[1, 0]:moving_bbox[1, 1],
                                        moving_bbox[2, 0]:moving_bbox[2, 1]].to(self.device)
                    input_['stereo'] = local_grids['stereo'][:, :, moving_bbox[0, 0]:moving_bbox[0, 1],
                                        moving_bbox[1, 0]:moving_bbox[1, 1],
                                        moving_bbox[2, 0]:moving_bbox[2, 1]].to(self.device)
                    # print(local_grids['tof'].shape)
                    # print(input_['tof'].shape)
                    with torch.no_grad():
                        # input_['neighborhood_filter']['tof'][0, 0, :, :, :] = -0.0*torch.ones_like(input_['neighborhood_filter']['tof'][0, 0, :, :, :])
                        # input_['neighborhood_filter']['tof'][0, 1, :, :, :] = 0.0*torch.ones_like(input_['neighborhood_filter']['tof'][0, 1, :, :, :])
                        sub_tsdf_filter = self._filtering(input_)['tsdf']
 
                    del input_

                    sub_tsdf = sub_tsdf_filter.cpu().detach()
                    del sub_tsdf_filter

                    # insert sub_tsdf into the local filtered grid
                    # here I nee to do the box shift so that I don't overwrite stuff!!!
                    # print(sub_tsdf.shape)
                    # print(moving_bbox)
                    filtered_local_grid[moving_bbox[0, 0] + int(chunk_size/4):moving_bbox[0, 1] - int(chunk_size/4),
                                        moving_bbox[1, 0] + int(chunk_size/4):moving_bbox[1, 1] - int(chunk_size/4),
                                        moving_bbox[2, 0] + int(chunk_size/4):moving_bbox[2, 1] - int(chunk_size/4)] = sub_tsdf[int(chunk_size/4):-int(chunk_size/4),
                                                                            int(chunk_size/4):-int(chunk_size/4),
                                                                            int(chunk_size/4):-int(chunk_size/4)]
                    del sub_tsdf

        # transfer the local_filtered_grid to the global grid
        # first remove the padding
        filtered_local_grid = filtered_local_grid[int(chunk_size/4):-int(chunk_size/4)-pad_x, 
                        int(chunk_size/4):-int(chunk_size/4)-pad_y, int(chunk_size/4):-int(chunk_size/4)-pad_z]

        database.filtered[scene].volume[bbox[0, 0]:bbox[0, 1],
                                                bbox[1, 0]:bbox[1, 1],
                                                bbox[2, 0]:bbox[2, 1]] = filtered_local_grid.numpy().squeeze()


    def _filter_training(self, indices, gt_vol, tsdf_vol, tsdf_vol_op, weights_vol, weights_vol_op, feat_vol, feat_vol_op, sensor, sensor_op, device):
        # this function computes 

        self.device = device

        indices = indices.cpu()

        # get minimum box size
        x_size = indices[:, 0].max() - indices[:, 0].min() # + 64 - (indices[:, 0].max() - indices[:, 0].min()) % 64
        y_size = indices[:, 1].max() - indices[:, 1].min() # + 64 - (indices[:, 1].max() - indices[:, 1].min()) % 64
        z_size = indices[:, 2].max() - indices[:, 2].min() # + 64 - (indices[:, 2].max() - indices[:, 2].min()) % 64
        
        bbox = np.array([[indices[:, 0].min(), indices[:, 0].min() + x_size],
                        [indices[:, 1].min(), indices[:, 1].min() + y_size],
                        [indices[:, 2].min(), indices[:, 2].min() + z_size]])

        # print(bbox)
        # extract a random location chunk inside the minimum bound. If the minimum bound is smaller
        # than the chunk size, extract the minimum bound
        if x_size / self.config.FILTERING_MODEL.chunk_size > 1:
            bbox[0, 0] =  np.random.random_integers(bbox[0, 0], bbox[0, 1] - self.config.FILTERING_MODEL.chunk_size)
            bbox[0, 1] =  bbox[0, 0] + self.config.FILTERING_MODEL.chunk_size
        if y_size / self.config.FILTERING_MODEL.chunk_size > 1:
            bbox[1, 0] =  np.random.random_integers(bbox[1, 0], bbox[1, 1] - self.config.FILTERING_MODEL.chunk_size)
            bbox[1, 1] =  bbox[1, 0] + self.config.FILTERING_MODEL.chunk_size
        if z_size / self.config.FILTERING_MODEL.chunk_size > 1:
            bbox[2, 0] =  np.random.random_integers(bbox[2, 0], bbox[2, 1] - self.config.FILTERING_MODEL.chunk_size)
            bbox[2, 1] =  bbox[2, 0] + self.config.FILTERING_MODEL.chunk_size
        # print('1',bbox)

        # make sure that each dimension of the bounding box is divisible by 8 (requires to do 3 max pooling layers, otherwise subtract
        # appropriate dimensions. When doing the extraction we will extract the 'correct' indices with this technique. No plus 1 needed since
        # the difference will be correct and the difference is what is extracted when doing n:n plus m
        # Here I do minus instead of plus as at test time because it does not matter during training
        # that not all indices are filtered. Also, I don't have to do more computations since I know that 
        # all indices are within the volume.
        if (bbox[0, 1] - bbox[0, 0]) % 2**self.config.FILTERING_MODEL.network_depth != 0:
            bbox[0, 1] -= (bbox[0, 1] - bbox[0, 0]) % 2**self.config.FILTERING_MODEL.network_depth
        if (bbox[1, 1] - bbox[1, 0]) % 2**self.config.FILTERING_MODEL.network_depth != 0:
            bbox[1, 1] -= (bbox[1, 1] - bbox[1, 0]) % 2**self.config.FILTERING_MODEL.network_depth
        if (bbox[2, 1] - bbox[2, 0]) % 2**self.config.FILTERING_MODEL.network_depth != 0:
            bbox[2, 1] -= (bbox[2, 1] - bbox[2, 0]) % 2**self.config.FILTERING_MODEL.network_depth

        # if the dimension is 0 along any of the bbox dimensions, then we need to return None here
        if np.prod(bbox[:, 1] - bbox[:, 0]) == 0:
            return None

        # print('2',bbox)
        # bbox[:, 1] += 1 # in order to make sure that we extract the limits of the bounding box. Without the plus 1, the bounding box
        # will be one to small due to the fact that extraction with n:n+1 only extracts n.
        # print(bbox)
        # Ideally I need to check that the extended box if uneven, still is valid. For the purposes
        # of training, I could also subtract so that I don't need to filter all indices

        input_1 = self._prepare_input_training(tsdf_vol, weights_vol, feat_vol, bbox) 

        input_2 = self._prepare_input_training(tsdf_vol_op, weights_vol_op, feat_vol_op, bbox) 

        neighborhood = dict()
        neighborhood[sensor] = input_1['neighborhood']
        neighborhood[sensor_op] = input_2['neighborhood']
 
        tsdf_filtered = self._filtering(neighborhood)
        # tsdf_filtered = tsdf_vol[bbox[0, 0]:bbox[0, 1], # for now I just feed the intermediate grid as the filtered one
        #                     bbox[1, 0]:bbox[1, 1],
        #                     bbox[2, 0]:bbox[2, 1]]

        if self.config.LOSS.gt_loss:
            # Here I use feat_vol just for now, but ideally, I should use a different feat_vol for when feeding gt depth frames so
            # that the feature network also can be trained on gt images because it is not clear what feature the gt depth images correspond to
            # for now I make sure that I don't use gt loss when using features
            input_1 = self._prepare_input_training(gt_vol.to(device), self.config.FUSION_MODEL.max_weight*torch.ones_like(weights_vol), feat_vol, bbox) 

            input_2 = self._prepare_input_training(gt_vol.to(device), self.config.FUSION_MODEL.max_weight*torch.ones_like(weights_vol), feat_vol_op, bbox) 

            neighborhood = dict()
            neighborhood[sensor] = input_1['neighborhood']
            neighborhood[sensor_op] = input_2['neighborhood']
     
            tsdf_gt_filtered = self._filtering(neighborhood)
        else:
            tsdf_gt_filtered = None

        del neighborhood

        # compute a mask determining what indices should be used in the loss out of all indices in the bbox. Note
        # that the bbox is not the full min bounding volume of the indices, but only a random extraction
        # according to the chunk size
        valid_indices = ((indices[:, 0] >= bbox[0, 0]) &
                (indices[:, 0] < bbox[0, 1]) &
                (indices[:, 1] >= bbox[1, 0]) &
                (indices[:, 1] < bbox[1, 1]) &
                (indices[:, 2] >= bbox[2, 0]) &
                (indices[:, 2] < bbox[2, 1]))


        valid_indices = torch.nonzero(valid_indices)[:, 0] # gives valid indices in indices variable but not in global grid
        valid_indices = indices[valid_indices, :] # extract the indices in the global grid
        # print('final valid_indices.shape: ', valid_indices.shape)

        # if valid_indices does not contain any indices, I need to return None here
        # print(valid_indices.shape[0])
        if not valid_indices.shape[0]:
            # print('return none')
            return None

        # mask target for loss
        tsdf_target = gt_vol[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
        # print('tsdf target shape: ', tsdf_target.shape)
        tsdf_target = tsdf_target.float() #.unsqueeze_(-1)

        # print('tsdf target sq shape> ', tsdf_target.squeeze().shape)
        # if tsdf_target.squeeze().shape[0] == 0:
        #     print('return none target shaep')
        #     return None

        del gt_vol

        # compute a mask determining the valid indices in the tsdf_filtered variable
        # There is only a translation shift between the origin of the gt_vol and the 
        # tsdf_filtered variable. Thus, I should be able to take the valid_indices minus
        # bbox[:, 0] in order to get the correct valid indices for tsdf_filtered
        valid_indices  = valid_indices - bbox[:, 0]

        # mask filtered loss
        tsdf_filtered['tsdf'] = tsdf_filtered['tsdf'][valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
        if self.config.LOSS.gt_loss:
            tsdf_gt_filtered['tsdf'] = tsdf_gt_filtered['tsdf'][valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]

        if self.config.FILTERING_MODEL.fuse_sensors:
            tsdf_filtered['tsdf_tof'] = tsdf_filtered['tsdf_tof'][valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
            tsdf_filtered['tsdf_stereo'] = tsdf_filtered['tsdf_stereo'][valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
            tsdf_filtered['tof_init'] = tsdf_filtered['tof_init'][valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
            tsdf_filtered['stereo_init'] = tsdf_filtered['stereo_init'][valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]

        tsdf_target = tsdf_target.to(device)

        output = dict()
        output['tsdf_filtered_grid'] = tsdf_filtered
        output['tsdf_gt_filtered_grid'] = tsdf_gt_filtered
        output['tsdf_target_grid'] = tsdf_target #.squeeze()
 
        return output

    def _translation_training(self, indices, feat_vol, feat_vol_op, weights_vol, weights_vol_op, sensor, sensor_op, device): #CHANGED
        self.device = device
 
        indices = indices.cpu() 

        input_1 = self._prepare_input_training(feat_vol, weights_vol, indices, sensor) #CHANGED

        input_2 = self._prepare_input_training(feat_vol_op, weights_vol_op, indices, sensor_op) #CHANGED

        neighborhood = dict()
        neighborhood[sensor] = input_1['neighborhood']
        neighborhood[sensor_op] = input_2['neighborhood']
 
        tsdf_translated, occupancy_value = self._translation(neighborhood)
        del neighborhood

        return tsdf_translated.squeeze(), occupancy_value.squeeze()

def masking(x, values, threshold=0., option='ueq'):

    if option == 'leq':

        if x.dim() == 2:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == 'geq':

        if x.dim() == 2:
            valid = (values >= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values >= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == 'eq':

        if x.dim() == 2:
            valid = (values == threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values == threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == 'ueq':

        if x.dim() == 2:
            valid = (values != threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values != threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]


    return xmasked
