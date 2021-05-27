import torch
import datetime
import time

import matplotlib.pyplot as plt

from modules.routing import ConfidenceRouting
from modules.extractor import Extractor
from modules.model import FusionNet
from modules.model_features import FeatureNet
from modules.model_features import FeatureResNet
from modules.integrator import Integrator
import math
import numpy as np
from scipy import ndimage

class Fuse_Pipeline(torch.nn.Module):

    def __init__(self, config):

        super(Fuse_Pipeline, self).__init__()

        self.config = config

        if config.ROUTING.do:
            raise NotImplementedError   
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

        config.FEATURE_MODEL.n_points = config.FUSION_MODEL.n_points
        config.FEATURE_MODEL.n_points_tof = config.FUSION_MODEL.n_points_tof
        config.FEATURE_MODEL.n_points_stereo = config.FUSION_MODEL.n_points_stereo
        config.FEATURE_MODEL.n_tail_points_tof = config.FUSION_MODEL.n_tail_points_tof
        config.FEATURE_MODEL.n_tail_points_stereo = config.FUSION_MODEL.n_tail_points_stereo

        self.n_features = self.config.FEATURE_MODEL.n_features
 
        self._extractor = dict()
        self._fusion_network = torch.nn.ModuleDict()
        self._feature_network = torch.nn.ModuleDict()
        for sensor in config.DATA.input:
            self._extractor[sensor] = Extractor(config.FUSION_MODEL, sensor)
            self._fusion_network[sensor] = FusionNet(config.FUSION_MODEL, sensor)
            if config.FEATURE_MODEL.network == 'resnet':
                self._feature_network[sensor] = FeatureResNet(config.FEATURE_MODEL, sensor)# TODO: adapt to when not using features
            else:
                self._feature_network[sensor] = FeatureNet(config.FEATURE_MODEL, sensor)# TODO: adapt to when not using features

        config.FUSION_MODEL.train_on_border_voxels = config.FILTERING_MODEL.MLP_MODEL.train_on_border_voxels
        self._integrator = Integrator(config.FUSION_MODEL)

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

    def _fusion(self, input_, input_features, values, sensor, gt_depth=None): # TODO: adapt to when not using features

        b, c, h, w = input_.shape
        if self.config.FUSION_MODEL.fixed:
            with torch.no_grad():
                # print('in: ', input_.sum())
                tsdf_pred = self._fusion_network[sensor].forward(input_)
                # print('out: ', tsdf_pred.sum())
        else:
            tsdf_pred = self._fusion_network[sensor].forward(input_)

        

        if self.config.FEATURE_MODEL.learned_features:
            if self.config.FEATURE_MODEL.relative_normalization:
                norm = dict()
                feat_pred = dict()
                for sensor_ in self.config.DATA.input:
                    feat_pred[sensor_] = self._feature_network[sensor_].forward(input_features[sensor_])
                    if self.config.FEATURE_MODEL.append_depth:
                        # print(feat_pred[sensor_][:, -1, :, :])
                        norm[sensor_] = torch.linalg.norm(feat_pred[sensor_][:, :-1, :, :], dim=1)
                    else:
                        norm[sensor_] = torch.linalg.norm(feat_pred[sensor_], dim=1)

                # normalize
                normalization_factor = None
                for k, sensor_ in enumerate(self.config.DATA.input):
                    if k == 0:
                        normalization_factor = norm[sensor_]
                    else:
                        torch.min(normalization_factor, norm[sensor_])

                feat_pred = feat_pred[sensor] / normalization_factor

            else:
                feat_pred =  self._feature_network[sensor].forward(input_features[sensor])
        else:
            if self.config.FEATURE_MODEL.append_pixel_conf:
                if sensor == 'gauss_close_thresh':
                    gt = torch.unsqueeze(gt_depth, -1)
                    gt = gt.permute(0, -1, 1, 2)
                    feat_pred = gt > 1.75
                    # feat_pred = input_features[sensor] > 1.75 # one where we have no noise
                elif sensor == 'gauss_far_thresh':
                    gt = torch.unsqueeze(gt_depth, -1)
                    gt = gt.permute(0, -1, 1, 2)
                    feat_pred = gt < 1.75
                    # feat_pred = input_features[sensor] < 1.75

                feat_pred = torch.cat((feat_pred, input_features[sensor]), dim=1)
            else:
                feat_pred = input_features[sensor]


        tsdf_pred = tsdf_pred.permute(0, 2, 3, 1)

        feat_pred = feat_pred.permute(0, 2, 3, 1) # (1, 256, 256, n_features)

        # save feature maps
        # for i in range(feat_pred.shape[-1]):
        #     plt.imsave(sensor + '/' + str(i)+ '_'+ sensor+   '.jpeg', feat_pred[0, :, :, i].cpu().detach().numpy())

        output = dict()

        try:
            n_points = eval('self.config.FUSION_MODEL.n_points_' + sensor)
        except:
            n_points = self.config.FUSION_MODEL.n_points

        tsdf_est = tsdf_pred.view(b, h * w, n_points)
        feature_est = feat_pred.view(b, h * w, 1, self.n_features)
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

    def _prepare_fusion_input(self, frame, values_sensor, sensor, confidence=None, n_points=None, rgb=None): # TODO: adapt to when not using features

        # get frame shape
        b, h, w = frame[sensor].shape
        # extracting data
        # reshaping data
        tsdf_input = {}
        tsdf_weights = {}
        for sensor_ in self.config.DATA.input:
            tsdf_input[sensor_] = values_sensor[sensor_]['fusion_values'].view(b, h, w, n_points)
            tsdf_weights[sensor_] = values_sensor[sensor_]['fusion_weights'].view(b, h, w, n_points)

        tsdf_frame = torch.unsqueeze(frame[sensor], -1)

        feature_input = dict()
        for sensor_ in self.config.DATA.input:
            feature_input[sensor_] = torch.unsqueeze(frame[sensor_], -1)
            if rgb is not None:
                rgb = rgb.unsqueeze(-1)
                rgb = rgb.view(1, h, w, -1)
                feature_input[sensor_] = torch.cat((feature_input[sensor_], rgb), dim=3)
        
            # del features, feature_weights_sensor
            # permuting input
            feature_input[sensor_] = feature_input[sensor_].permute(0, -1, 1, 2)
        del rgb
        # stacking input data
        if self.config.FUSION_MODEL.with_peek:
            raise NotImplementedError
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
                tsdf_input = torch.cat([tsdf_frame, tsdf_confidence, tsdf_weights[sensor], tsdf_input[sensor]], dim=3)
                del tsdf_confidence
            else:
                tsdf_input = torch.cat([tsdf_frame, tsdf_weights[sensor], tsdf_input[sensor]], dim=3)

        # permuting input
        tsdf_input = tsdf_input.permute(0, -1, 1, 2)

        del tsdf_frame

        return tsdf_input, feature_input

    def _prepare_volume_update(self, values, est, features, inputs, sensor) -> dict:# TODO: adapt to when not using features
        try:
            tail_points = eval('self.config.FUSION_MODEL.n_tail_points_' + sensor)
        except:
            tail_points = self.config.FUSION_MODEL.n_tail_points


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

        # I do the erosion to avoid training on too many indices that are at the edge of the initialized space
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
        # the option of not training on border voxels is irrelevant for the 3dconv architecture
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

    def fuse(self, # TODO: adapt to when not using features
             batch,
             database,
             device):

        self.device = device

        # routing
        if self.config.ROUTING.do:
            raise NotImplementedError
            frame, confidence = self._routing(batch)

            if self.config.DATA.input == 'multidepth' and (self.config.DATA.fusion_strategy == 'fusionNet' or self.config.DATA.fusion_strategy == 'two_fusionNet' or self.config.DATA.fusion_strategy == 'fusionNet_conditioned'):
                filtered_frame = frame.detach().clone()
                filtered_frame[confidence < batch['confidence_threshold']] = 0
            else:
                filtered_frame = frame.detach().clone()
                filtered_frame[confidence < self.config.ROUTING.threshold] = 0


        else:
            frame = dict()
            for sensor_ in self.config.DATA.input: # we require to load all sensor frames if we do relative normalization in feature net
                frame[sensor_] = batch[batch['sensor'] + '_depth'].squeeze_(1)
                frame[sensor_] = frame[sensor_].to(device)
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
            filtered_frame = torch.where(mask == 0, torch.zeros_like(frame[batch['sensor']]),
                                    frame[batch['sensor']])

        # get current tsdf values
        scene_id = batch['frame_id'][0].split('/')[0]

        extracted_values = dict()
        for sensor in self.config.DATA.input:
            try:
                intrinsics = batch['intrinsics' + '_' + batch['sensor']]
            except:
                intrinsics = batch['intrinsics']
            extracted_values[sensor] = self._extractor[batch['sensor']].forward(frame[batch['sensor']],
                                         batch['extrinsics'],
                                         intrinsics,
                                         database[scene_id]['tsdf' + '_' + sensor],
                                         database[scene_id]['features_' + sensor],
                                         database[scene_id]['origin'],
                                         database[scene_id]['resolution'],
                                         self.config.SETTINGS.gpu,
                                         database[scene_id]['weights' + '_' + sensor],
                                         database[scene_id]['feature_weights' + '_' + sensor])

        try:
            n_points = eval('self.config.FUSION_MODEL.n_points_' + batch['sensor'])
        except:
            n_points = self.config.FUSION_MODEL.n_points
        tsdf_input, feature_input = self._prepare_fusion_input(frame, extracted_values, batch['sensor'],
                                                              confidence, n_points, rgb)
        del rgb, frame


        fusion_output = self._fusion(tsdf_input, feature_input, extracted_values[batch['sensor']], batch['sensor'], batch['gt'].to(device))

        # masking invalid losses
        tsdf_est = fusion_output['tsdf_est']
        feature_est = fusion_output['feature_est']

        integrator_input = self._prepare_volume_update(extracted_values[batch['sensor']],
                                                        tsdf_est,
                                                        feature_est,
                                                        filtered_frame,
                                                        batch['sensor'])

        tsdf, features, weights, feature_weights, indices = self._integrator.forward(integrator_input,
                                                   database[scene_id][
                                                       'tsdf_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'features_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'weights_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'feature_weights_' + batch['sensor']].to(device))


        del indices, integrator_input

        database.tsdf[batch['sensor']][
            scene_id].volume = tsdf.cpu().detach().numpy()
        database.fusion_weights[batch['sensor']] [
            scene_id] = weights.cpu().detach().numpy()
        database.features[batch['sensor']][
            scene_id] = features.cpu().detach().numpy()
        database.feature_weights[batch['sensor']][
            scene_id] = feature_weights.cpu().detach().numpy()


        del tsdf, weights, features, feature_weights

        return

    def fuse_training(self, batch, database, device): # TODO: adapt to when not using features

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
            raise NotImplementedError
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
            frame = dict()
            for sensor_ in self.config.DATA.input: # we require to load all sensor frames if we do relative normalization in feature net
                frame[sensor_] = batch[batch['sensor'] + '_depth'].squeeze_(1)
                frame[sensor_] = frame[sensor_].to(device)
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
            filtered_frame = torch.where(mask == 0, torch.zeros_like(frame[batch['sensor']]),
                                    frame[batch['sensor']])
        del mask

        b, h, w = frame[batch['sensor']].shape

        # get current tsdf values
        scene_id = batch['frame_id'][0].split('/')[0]

        extracted_values = dict()
        try:
            intrinsics = batch['intrinsics' + '_' + batch['sensor']]
        except:
            intrinsics = batch['intrinsics']
        for sensor in self.config.DATA.input:

            extracted_values[sensor] = self._extractor[batch['sensor']].forward(frame[batch['sensor']],
                                         batch['extrinsics'],
                                         intrinsics,
                                         database[scene_id]['tsdf' + '_' + sensor],
                                         database[scene_id]['features_' + sensor],
                                         database[scene_id]['origin'],
                                         database[scene_id]['resolution'],
                                         self.config.SETTINGS.gpu,
                                         database[scene_id]['weights' + '_' + sensor],
                                         database[scene_id]['feature_weights' + '_' + sensor])


        # TODO: make function that extracts only the gt values for speed up during training
        extracted_values_gt = self._extractor[batch['sensor']].forward(frame[batch['sensor']],
                                         batch['extrinsics'],
                                         intrinsics,
                                         database[scene_id]['gt'],
                                         database[scene_id]['features_' + batch['sensor']],
                                         database[scene_id]['origin'],
                                         database[scene_id]['resolution'],
                                         self.config.SETTINGS.gpu,
                                         database[scene_id]['weights_' + batch['sensor']],
                                         database[scene_id]['feature_weights_' + batch['sensor']])

        tsdf_target = extracted_values_gt['fusion_values']
        del extracted_values_gt

        try:
            n_points = eval('self.config.FUSION_MODEL.n_points_' + batch['sensor'])
        except:
            n_points = self.config.FUSION_MODEL.n_points
        tsdf_input, feature_input = self._prepare_fusion_input(frame, extracted_values, batch['sensor'], confidence, n_points, rgb)
        del rgb, frame
        
        # tsdf_target = tsdf_target.view(b, h, w, eval('self.config.FUSION_MODEL.n_points_' + batch['sensor']))
        fusion_output = self._fusion(tsdf_input, feature_input, extracted_values[batch['sensor']], batch['sensor'], batch['gt'].to(device))


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

        integrator_input = self._prepare_volume_update(extracted_values[batch['sensor']],
                                                        tsdf_est,
                                                        feature_est,
                                                        filtered_frame,
                                                        batch['sensor'])

        del extracted_values, tsdf_est, feature_est, filtered_frame

        tsdf, features, weights, feature_weights, indices = self._integrator.forward(integrator_input,
                                                   database[scene_id][
                                                       'tsdf_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'features_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'weights_' + batch['sensor']].to(device),
                                                   database[scene_id][
                                                       'feature_weights_' + batch['sensor']].to(device))


        del integrator_input

        database.tsdf[batch['sensor']][
            scene_id].volume = tsdf.cpu().detach().numpy()
        database.fusion_weights[batch['sensor']] [
            scene_id] = weights.cpu().detach().numpy()
        database.features[batch['sensor']][
            scene_id] = features.cpu().detach().numpy()
        database.feature_weights[batch['sensor']][
            scene_id] = feature_weights.cpu().detach().numpy()

        output['tsdf'] = tsdf
        output['weights'] = weights
        output['features'] = features
        output['indices'] = indices

        del tsdf, weights, features

        return output

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
