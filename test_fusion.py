import torch
import argparse
import os

import numpy as np

from utils import loading
from utils import setup
from utils import transform

from modules.extractor import Extractor
from modules.integrator import Integrator
from modules.model import FusionNet
from modules.routing import ConfidenceRouting
from modules.pipeline import Pipeline
import random
from scipy import ndimage

from tqdm import tqdm

from utils.metrics import evaluation
from utils.setup import *
from utils.visualize_sensor_weighting import visualize_sensor_weighting
from utils.visualize_features import visualize_features
from utils.compute_proxy_sensor_weighting import compute_proxy_sensor_weighting_and_mesh

import h5py
import open3d as o3d

from evaluate_3d_reconstruction import run_evaluation

def arg_parse():
    parser = argparse.ArgumentParser(description='Script for testing SenFuNet')

    parser.add_argument('--config', required=True)

    args = parser.parse_args()

    return vars(args)

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_fusion(config):
    # define output dir
    test_path = '/test_no_carving_debug'
    test_dir = config.SETTINGS.experiment_path + '/' + config.TESTING.fusion_model_path.split('/')[-3] + test_path

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if config.SETTINGS.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    config.FUSION_MODEL.device = device

    # get test dataset
    data_config = setup.get_data_config(config, mode='test')
    dataset = setup.get_data(config.DATA.dataset, data_config)
    loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=config.TESTING.test_batch_size,
                                            shuffle=config.TESTING.test_shuffle,
                                            pin_memory=True,
                                            num_workers=16)

    # specify number of features
    if config.FEATURE_MODEL.learned_features:
        config.FEATURE_MODEL.n_features = config.FEATURE_MODEL.n_features + config.FEATURE_MODEL.append_depth
    else:
        config.FEATURE_MODEL.n_features = config.FEATURE_MODEL.append_pixel_conf + config.FEATURE_MODEL.append_depth + 3*config.FEATURE_MODEL.w_rgb# 1 for label encoding of noise in gaussian threshold data


    # get test database
    database = setup.get_database(dataset, config, mode='test')

    # setup pipeline
    # pipeline = Pipeline(config)
    # pipeline = pipeline.to(device)

    # for sensor in config.DATA.input:
    #     if config.FUSION_MODEL.use_fusion_net:
    #         print('Fusion Net ', sensor, ': ', count_parameters(pipeline.fuse_pipeline._fusion_network[sensor]))
    #     print('Feature Net ', sensor, ': ', count_parameters(pipeline.fuse_pipeline._feature_network[sensor]))

    # if pipeline.filter_pipeline is not None:
    #     print('Filtering Net: ', count_parameters(pipeline.filter_pipeline._filtering_network))



    # loading.load_pipeline(config.TESTING.fusion_model_path, pipeline) # this loads all parameters it can

    # # load pretrained routing model into Imparameters
    # # if config.ROUTING.do:
    # #     if config.DATA.fusion_strategy == 'routingNet':
    # #         routing_checkpoint = torch.load(config.TESTING.routing_model_path)
    # #         pipeline._routing_network.load_state_dict(routing_checkpoint['pipeline_state_dict'])
    # #     elif config.DATA.fusion_strategy == 'fusionNet':
    # #         for sensor_ in config.DATA.input:
    # #             routing_checkpoint = torch.load(eval('config.TRAINING.routing_' + sensor_ + '_model_path'))
    # #             pipeline.fuse_pipeline._routing_network[sensor_].load_state_dict(routing_checkpoint['pipeline_state_dict'])

    # # load pipelines
    # # for sensor in config.DATA.input: # in the event that we want to load specific fusion and feature nets - overwrite the load_pipeline method
    # #     if sensor == 'tof' or sensor == 'stereo':
    # #         loading.load_net_old(eval('config.TRAINING.pretraining_fusion_' + sensor +  '_model_path'), pipeline.fuse_pipeline._fusion_network[sensor], sensor)
    # #     else:
    # #         loading.load_net(eval('config.TRAINING.pretraining_fusion_' + sensor +  '_model_path'), pipeline.fuse_pipeline._fusion_network[sensor], sensor)
    #     # to load the same fusion net for both sensors
    #     # be careful, this will load perfectly for any sensor combination, but is only suitable for the gt sensors!
    #     # loading.load_net('/cluster/work/cvl/esandstroem/src/late_fusion_3dconvnet/workspace/fusion/210507-093251/model/best.pth.tar', pipeline.fuse_pipeline._fusion_network[sensor], 'left_depth_gt_2')
        
    #     # if config.FILTERING_MODEL.w_features: # this does not work on the new models
    #         # loading.load_net(config.TESTING.fusion_model_path, pipeline.fuse_pipeline._feature_network[sensor], sensor)

    # pipeline.eval()

    # sensors = config.DATA.input # ['tof', 'stereo'] # make sure thi only used when we have input: multidepth and fusion_strategy: fusionNet and derivatives
    
    # # test model
    # pipeline.test(loader, dataset, database, sensors, device)

    # # evaluate test scenes
    # eval_results, eval_results_fused, \
    # eval_results_scene, \
    # eval_results_scene_fused = database.evaluate(mode='test')

    # # save test_eval to log file
    # logger = setup.get_logger(test_dir, 'test')
    # for sensor in eval_results.keys():
    #     logger.info('Average ' + sensor + ' test results over test scenes')
    #     for metric in eval_results[sensor]:
    #         logger.info(metric + ': ' + str(eval_results[sensor][metric]))


    # logger.info('Average test results over fused test scenes')
    # for metric in eval_results_fused:
    #     logger.info(metric + ': ' + str(eval_results_fused[metric]))

    # for sensor in eval_results_scene.keys():
    #     logger.info('Per scene ' + sensor + ' results')
    #     for scene in eval_results_scene[sensor]:
    #         logger.info('Scene: ' + scene)
    #         for metric in eval_results_scene[sensor][scene]:
    #             logger.info(metric + ': ' + str(eval_results_scene[sensor][scene][metric]))

    # logger.info('Per scene fused results')
    # for scene in eval_results_scene_fused:
    #     logger.info('Scene: ' + scene)
    #     for metric in eval_results_scene_fused[scene]:
    #         logger.info(metric + ': ' + str(eval_results_scene_fused[scene][metric]))

    # # save ply-files of test scenes
    # for scene_id in database.scenes_gt.keys():
    #     database.save(path=test_dir, scene_id=scene_id)

    # compute f-score for the test scenes and render them
    evaluate(database, config, test_path)
        # print('running scritpt: visualize_confidence_reconstruction.py ' + scene_id)
        # os.system('visualize_confidence_reconstruction.py ' + scene_id)
        # print('running script: evaluate_3d_reconstruction.py ' + scene_id + '.ply artificial_trunc confidencefusion_w_attention')
        # os.system('evaluate_3d_reconstruction.py ' + scene_id + '.ply artificial_trunc confidencefusion_w_attention')
        # print('running script: evaluate_3d_reconstruction.py ' + scene_id + '_filtered.ply artificial_trunc confidencefusion_w_attention')
        # os.system('evaluate_3d_reconstruction.py ' + scene_id + '_filtered.ply artificial_trunc confidencefusion_w_attention')
        # # add colored confidence ply here as well
        # it appears I have to write a script that inputs the confidence grid as well as the ply file. Then, for each face in the ply
        # file, find the vertices and convert each vertex into the voxel grid space and find the closest index. Take the average
        # of all vertex indices confidence to get the face confidence and map the value to a colormap.

def evaluate(database, config, test_path):
    sdf_gt_path = os.getenv(config.DATA.root_dir) # when training on local scratch
        
    # os.getenv returns none when the input does not exist. When 
    # it returns none, we want to train on the work folder
    if not sdf_gt_path:
        sdf_gt_path  = config.DATA.root_dir
    test_dir = config.TESTING.fusion_model_path.split('/')[:-2]
    test_dir = '/'.join(test_dir) + test_path
    weight_thresholds = config.TESTING.weight_thresholds
    pad = config.DATA.pad

    for scene in database.scenes_gt.keys():
        tsdf_path = test_dir

        sdf_gt = sdf_gt_path + '/' + scene + '/sdf_' + scene + '.hdf' 

        # load gt grid
        f = h5py.File(sdf_gt, 'r')
        sdf_gt = np.array(f['sdf']).astype(np.float16)
        # truncate
        truncation = config.DATA.trunc_value
        # if config.DATA.truncation_strategy == 'standard':
        sdf_gt[sdf_gt >= truncation] = truncation
        sdf_gt[sdf_gt <= -truncation] = -truncation
        # elif config.DATA.truncation_strategy == 'artificial':
        #     sdf_gt[sdf_gt >= truncation] = truncation
        #     sdf_gt[sdf_gt <= -truncation] = truncation

        # pad gt grid if necessary
        if pad > 0:
           if config.DATA.truncation_strategy == 'standard':
               sdf_gt = np.pad(sdf_gt, pad, 'constant', constant_values=-truncation)
           elif config.DATA.truncation_strategy == 'artificial':
               sdf_gt = np.pad(sdf_gt, pad, 'constant', constant_values=truncation)

        voxel_size = f.attrs['voxel_size'] 
        resolution = sdf_gt.shape
        max_resolution = np.array(resolution).max()
        length = (max_resolution)*voxel_size

        # define paths
        for weight_threshold in weight_thresholds:
            model_test = scene + '_weight_threshold_' + str(weight_threshold)
            model_test = model_test + '_filtered'
            logger = get_logger(test_dir, name=model_test)
            tsdf = tsdf_path + '/' + scene + '.tsdf_filtered.hf5'

            # read tsdfs and weight grids
            f = h5py.File(tsdf, 'r')
            tsdf = np.array(f['TSDF_filtered']).astype(np.float16)

            # load weight grids for all sensors here to get the mask used for filtering
            # Fail now I know why I never could get better recall on my conv3dmodels with stereo / becasue I only
            # used the tof mask!!
            mask = np.zeros_like(tsdf)
            and_mask = np.ones_like(tsdf)
            sensor_mask = dict()
            
            for sensor_ in config.DATA.input:
                # print(sensor_)
                weights = tsdf_path + '/' + scene + '_' + sensor_ + '.weights.hf5'
                f = h5py.File(weights, 'r')
                weights = np.array(f['weights']).astype(np.float16)
                mask = np.logical_or(mask, weights > 0)
                and_mask = np.logical_and(and_mask, weights > 0)
                sensor_mask[sensor_] = weights > 0

            sensor_weighting_mask = mask.copy() # we want to display the sensor weighting before outlier filtering
                # break
                # wrong here when we use weight_threshold > 0 since we make voxels where both
                # sensors have integrated into single-sensor observations suddenly. Instead we
                # want to use 0 as the threshold above and then to filter with a non-zero weight
                # threshold, we want to apply the weight_thresholding after the single sensor 
                # filtering

            # use the and_mask together with the sensor_weighting grid to filter the mask s.t.
            # when only one sensor has integrated and the confidence is less than 0.5 for the integrated sensor,
            # remove the entry from the mask
            # load sensor weighting grid
            if len(config.DATA.input) > 1: #alpha eq 0 means we trust gauss far 
                # load weighting sensor grid
                sensor_weighting = tsdf_path + '/' + scene + '.sensor_weighting.hf5'
                f = h5py.File(sensor_weighting, 'r')
                sensor_weighting = np.array(f['sensor_weighting']).astype(np.float16)

                if config.FILTERING_MODEL.outlier_channel:
                    sensor_weighting = sensor_weighting[1, :, :, :]

                only_one_sensor_mask = np.logical_xor(mask, and_mask)
                for sensor_ in config.DATA.input:
                    only_sensor_mask = np.logical_and(only_one_sensor_mask, sensor_mask[sensor_])
                    if sensor_ == config.DATA.input[0]: 
                        rem_indices = np.logical_and(only_sensor_mask, sensor_weighting < 0.5)
                    else:
                        # before I fixed the bug always ended up here when I had tof and stereo as sensors
                        # but this would mean that for the tof sensor I removed those indices
                        # if alpha was larger than 0.5 which it almost always is. This means that 
                        # essentially all (cannot be 100 % sure) voxels where we only integrated 
                        # tof, was removed. Since the histogram is essentially does not have 
                        # any voxels with trust less than 0.5, we also removed all alone stereo voxels
                        # so at the end we end up with a mask very similar to the and_mask
                        rem_indices = np.logical_and(only_sensor_mask, sensor_weighting > 0.5)

                    mask[rem_indices] = 0

            weight_mask = np.zeros_like(tsdf)
            for sensor_ in config.DATA.input:
                # print(sensor_)
                weights = tsdf_path + '/' + scene + '_' + sensor_ + '.weights.hf5'
                f = h5py.File(weights, 'r')
                weights = np.array(f['weights']).astype(np.float16)
                weight_mask = np.logical_or(weight_mask, weights > weight_threshold)

            # filter away outliers using the weight mask when weight_threshold > 0
            mask = np.logical_and(mask, weight_mask)


            # erode masks appropriately
            if config.FILTERING_MODEL.erosion:
                mask = ndimage.binary_erosion(mask, structure=np.ones((3,3,3)), iterations=1)

            eval_results_scene = evaluation(tsdf, sdf_gt, mask)

            logger.info('Test Scores for scene: ' + scene)
            for key in eval_results_scene:
                logger.info(key + ': ' + str(eval_results_scene[key]))
    

            # Create the mesh using the given mask
            tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
            tsdf_cube[:resolution[0], :resolution[1], :resolution[2]] = tsdf


            indices_x = mask.nonzero()[0]
            indices_y = mask.nonzero()[1]
            indices_z = mask.nonzero()[2]

            # this creates a voxelgrid with max_resolution voxels along the length length. Each 
            # voxel consists of 8 vertices in the tsdf_cube which means that when we have a tsdf_cube
            # of max_resolution 2 (8 vertices), we will make the uniform volume of size 27 vertices.
            # This is not a problem, however, since we will only initialize the valid indices. I.e. 
            # the unifor volue is always 1 vertex layer too large compared to the tsdf_cube. To correct
            # for this, the max_resolution variable should be 1 less than it is now, making length smaller
            # as well since length is max_resolution times voxel_size
            volume = o3d.integration.UniformTSDFVolume(
                    length=length,
                    resolution=max_resolution,
                    sdf_trunc=truncation,
                    color_type=o3d.integration.TSDFVolumeColorType.RGB8)
            
            for i in range(indices_x.shape[0]):
                volume.set_tsdf_at(tsdf_cube[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
                volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])               

            print("Extract a triangle mesh from the volume and visualize it.")
            mesh = volume.extract_triangle_mesh()

            del volume
            mesh.compute_vertex_normals()
            # o3d.visualization.draw_geometries([mesh])
            o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test + '.ply'), mesh)


            # # volume = o3d.integration.UniformTSDFVolume(
            # #         length=3.0,
            # #         resolution=3,
            # #         sdf_trunc=1.0,
            # #         color_type=o3d.integration.TSDFVolumeColorType.RGB8)
            
            # # volume.set_tsdf_at(0.5, 1,1,1)
            # # volume.set_tsdf_at(0.5, 2,1,1)
            # # volume.set_tsdf_at(0.5, 2,1,2)
            # # volume.set_tsdf_at(0.5, 1,1,2)
            # # volume.set_weight_at(1, 1,1,1)
            # # volume.set_weight_at(1, 2,1,1)
            # # volume.set_weight_at(1, 2,1,2)
            # # volume.set_weight_at(1, 1,1,2)
            # # volume.set_weight_at(1, 1,2,1)
            # # volume.set_weight_at(1, 1,2,2)
            # # volume.set_weight_at(1, 2,2,2)
            # # volume.set_weight_at(1, 2,2,1)
            # # volume.set_tsdf_at(-0.5, 1,2,1)
            # # volume.set_tsdf_at(-0.5, 1,2,2)
            # # volume.set_tsdf_at(-0.5, 2,2,2)
            # # volume.set_tsdf_at(-0.5, 2,2,1)


            # # volume.set_weight_at(1, 1,1,0)
            # # volume.set_weight_at(1, 2,1,0)
            # # volume.set_weight_at(1, 2,1,1)
            # # volume.set_weight_at(1, 1,1,1)
            # # volume.set_weight_at(1, 1,2,0)
            # # volume.set_weight_at(1, 1,2,1)
            # # volume.set_weight_at(1, 2,2,1)
            # # volume.set_weight_at(1, 2,2,0)
            # # volume.set_tsdf_at(0.5, 1,1,0)
            # # volume.set_tsdf_at(0.5, 2,1,0)
            # # volume.set_tsdf_at(0.5, 2,1,1)
            # # volume.set_tsdf_at(0.5, 1,1,1)
            # # volume.set_tsdf_at(-0.5, 1,2,0)
            # # volume.set_tsdf_at(-0.5, 1,2,1)
            # # volume.set_tsdf_at(-0.5, 2,2,1)
            # # volume.set_tsdf_at(-0.5, 2,2,0)

            # # a = volume.extract_voxel_grid()
            # # print(a)

            # # mesh = volume.extract_triangle_mesh()
            # # print(np.asarray(mesh.vertices))
            # # print(np.asarray(mesh.faces))
            # # o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test + 'test4.ply'), mesh)
            # # volume.set_weight_at(1, 0,1,1)
            # # mesh = volume.extract_triangle_mesh()
            # # o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test + 'test2.ply'), mesh)
            # # return
            # if len(config.DATA.input) > 1:
            #     # Generate visualization of the sensor weighting
            #     # load weighting sensor grid
            #     sensor_weighting = tsdf_path + '/' + scene + '.sensor_weighting.hf5'
            #     f = h5py.File(sensor_weighting, 'r')
            #     sensor_weighting = np.array(f['sensor_weighting']).astype(np.float16)

            #     # compute sensor weighting histogram and mesh visualization
            #     visualize_sensor_weighting(tsdf, sensor_weighting, test_dir, sensor_weighting_mask, \
            #         truncation, length, max_resolution, resolution, voxel_size, config.FILTERING_MODEL.outlier_channel)


            # Compute the F-score, precision and recall
            ply_path = model_test + '.ply'

            # run commandline command
            # legacy code: used when 3d evaluation was an executable
            # os.chdir(test_dir)
            # print('running script: evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)
            # os.system('evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)

            # evaluate F-score
            run_evaluation(ply_path, 'standard_trunc', scene, test_dir)

            # move the logs and plys to the evaluation dirs
            os.system('mv ' + test_dir + '/' + model_test + '.logs ' + test_dir + '/' + model_test + '/' + model_test + '.logs')
            os.system('mv ' + test_dir + '/' + model_test + '.ply ' + test_dir + '/' + model_test + '/' + model_test + '.ply')
            # if len(config.DATA.input) > 1:
            #     os.system('mv ' + test_dir + '/sensor_weighting_nn_no_outlier_filter.ply ' + test_dir + '/' + model_test + '/sensor_weighting_nn.ply')
            #     os.system('mv ' + test_dir + '/sensor_weighting_grid_histogram_no_outlier_filter.png ' + test_dir + '/' + model_test + '/sensor_weighting_grid_histogram.png')
            #     os.system('mv ' + test_dir + '/sensor_weighting_surface_histogram_no_outlier_filter.png ' + test_dir + '/' + model_test + '/sensor_weighting_surface_histogram.png')

        # # return
            # for sensor_ in config.DATA.input:
            #     model_test = scene + '_weight_threshold_' + str(weight_threshold)
            #     model_test = model_test + '_' + sensor_
            #     logger = get_logger(test_dir, name=model_test)

            #     tsdf = tsdf_path + '/' + scene + '_' + sensor_ + '.tsdf.hf5'
            #     weights = tsdf_path + '/' + scene + '_' + sensor_ + '.weights.hf5'

            #     f = h5py.File(weights, 'r')
            #     weights = np.array(f['weights']).astype(np.float16)
            #     # to eval routedfusion on nn mask
            #     weights = np.zeros_like(weights)
            #     for sensor_ in ['sgm_stereo', 'stereo']:
            #         weight_path = config.SETTINGS.experiment_path + '/' + \
            #             '211009-101812' + test_path[:-5] + '/' + scene + '_' + sensor_ + '.weights.hf5'
            #         f = h5py.File(weight_path, 'r')
            #         weights = np.logical_or(weights, np.array(f['weights']).astype(np.float16))
            #     # read tsdfs and weight grids
            #     f = h5py.File(tsdf, 'r')
            #     tsdf = np.array(f['TSDF']).astype(np.float16)
            #     # print(tsdf.astype(np.float32).sum())

            #     # print(weights.astype(np.float32).sum())

            #     # compute the L1, IOU and Acc
            
            #     mask = weights > weight_threshold

            #     # erode masks appropriately
            #     if config.FILTERING_MODEL.erosion:
            #         mask = ndimage.binary_erosion(mask, structure=np.ones((3,3,3)), iterations=1)

            #     eval_results_scene = evaluation(tsdf, sdf_gt, mask)

            #     logger.info('Test Scores for scene: ' + scene)
            #     for key in eval_results_scene:
            #         logger.info(key + ': ' + str(eval_results_scene[key]))
        

            #     # Create the mesh using the given mask
            #     tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
            #     tsdf_cube[:resolution[0], :resolution[1], :resolution[2]] = tsdf


            #     indices_x = mask.nonzero()[0]
            #     indices_y = mask.nonzero()[1]
            #     indices_z = mask.nonzero()[2]
            #     # print(indices_x.shape)

            #     volume = o3d.integration.UniformTSDFVolume(
            #             length=length,
            #             resolution=max_resolution,
            #             sdf_trunc=truncation,
            #             color_type=o3d.integration.TSDFVolumeColorType.RGB8)
                
            #     for i in range(indices_x.shape[0]):
            #         volume.set_tsdf_at(tsdf_cube[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
            #         volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])               

            #     print("Extract a triangle mesh from the volume and visualize it.")
            #     mesh = volume.extract_triangle_mesh()
            #     # print(np.asarray(mesh.vertices).shape)
            #     # print('isnan mesh verticies', np.isnan(np.asarray(mesh.vertices).sum()))
            #     del volume
            #     mesh.compute_vertex_normals()
            #     # o3d.visualization.draw_geometries([mesh])
            #     o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test + '.ply'), mesh)

            #     # Compute the F-score, precision and recall
            #     ply_path = model_test + '.ply'

            #     # run commandline command
            #     # legacy code: used when I had the evaluate_3d_reconstruction as an executable
            #     os.chdir(test_dir)
            #     print('running script: evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)
            #     os.system('evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)

            #     evaluate F-score
            #     run_evaluation(ply_path, 'standard_trunc', scene, test_dir)

            #     # move the logs and plys to the evaluation dirs
            #     os.system('mv ' + test_dir + '/' + model_test + '.logs ' + test_dir + '/' + model_test + '/' + model_test + '.logs')
            #     os.system('mv ' + test_dir + '/' + model_test + '.ply ' + test_dir + '/' + model_test + '/' + model_test + '.ply')
  
            #     # evalute the refined tsdf grid if available
            #     if config.FILTERING_MODEL.use_outlier_filter:
            #         model_test = scene + '_weight_threshold_' + str(weight_threshold)
            #         model_test = model_test + '_refined_' + sensor_
            #         logger = get_logger(test_dir, name=model_test)

            #         tsdf = tsdf_path + '/' + scene + '_' + sensor_ + '.tsdf_refined.hf5'
            #         weights = tsdf_path + '/' + scene + '_' + sensor_ + '.weights.hf5'

            #         # read tsdfs and weight grids
            #         f = h5py.File(tsdf, 'r')
            #         tsdf = np.array(f['TSDF']).astype(np.float16)

            #         f = h5py.File(weights, 'r')
            #         weights = np.array(f['weights']).astype(np.float16)
            #         # print(weights.astype(np.float32).sum())

            #         # compute the L1, IOU and Acc
                
            #         mask = weights > weight_threshold

            #         # erode masks appropriately
            #         if config.FILTERING_MODEL.erosion:
            #             mask = ndimage.binary_erosion(mask, structure=np.ones((3,3,3)), iterations=1)

            #         eval_results_scene = evaluation(tsdf, sdf_gt, mask)

            #         logger.info('Test Scores for scene: ' + scene)
            #         for key in eval_results_scene:
            #             logger.info(key + ': ' + str(eval_results_scene[key]))
            
            #         # Create the mesh using the given mask
            #         tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
            #         tsdf_cube[:resolution[0], :resolution[1], :resolution[2]] = tsdf

            #         indices_x = mask.nonzero()[0]
            #         indices_y = mask.nonzero()[1]
            #         indices_z = mask.nonzero()[2]

            #         volume = o3d.integration.UniformTSDFVolume(
            #                 length=length,
            #                 resolution=max_resolution,
            #                 sdf_trunc=truncation,
            #                 color_type=o3d.integration.TSDFVolumeColorType.RGB8)
                    
            #         for i in range(indices_x.shape[0]):
            #             volume.set_tsdf_at(tsdf_cube[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
            #             volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])               

            #         print("Extract a triangle mesh from the volume and visualize it.")
            #         mesh = volume.extract_triangle_mesh()
            #         # print(np.asarray(mesh.vertices).shape)
            #         # print('isnan mesh verticies', np.isnan(np.asarray(mesh.vertices).sum()))
            #         del volume
            #         mesh.compute_vertex_normals()
            #         # o3d.visualization.draw_geometries([mesh])
            #         o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test + '.ply'), mesh)

            #         # # Compute the F-score, precision and recall
            #         ply_path = model_test + '.ply'

            #         # legacy code: when evaluate_3d_reconstruction was an used as 
            #         # an executable
            #         # run commandline command
            #         os.chdir(test_dir)
            #         print('running script: evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)
            #         os.system('evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)

            #         evaluate F-score
            #         run_evaluation(ply_path, 'standard_trunc', scene, test_dir)

            #         # # move the logs and plys to the evaluation dirs
            #         os.system('mv ' + test_dir + '/' + model_test + '.logs ' + test_dir + '/' + model_test + '/' + model_test + '.logs')
            #         os.system('mv ' + test_dir + '/' + model_test + '.ply ' + test_dir + '/' + model_test + '/' + model_test + '.ply')
      


            # if config.FILTERING_MODEL.features_to_sdf_enc or config.FILTERING_MODEL.features_to_weight_head:
            #     features = dict()
            #     tsdfs = dict()
            #     weights = dict()
            #     for sensor_ in config.DATA.input:
            #         featurename = tsdf_path + '/' + scene + '_' + sensor_ + '.features.hf5'
            #         f = h5py.File(featurename, 'r')
            #         features[sensor_] = np.array(f['features']).astype(np.float16)
            #         tsdfname = tsdf_path + '/' + scene + '_' + sensor_ + '.tsdf.hf5'
            #         f = h5py.File(tsdfname, 'r')
            #         tsdfs[sensor_] = np.array(f['TSDF']).astype(np.float16)
            #         weightname = tsdf_path + '/' + scene + '_' + sensor_ + '.weights.hf5'
            #         f = h5py.File(weightname, 'r')
            #         weights[sensor_] = np.array(f['weights']).astype(np.float16)

            #     proxy_sensor_weighting = compute_proxy_sensor_weighting_and_mesh(tsdfs, sdf_gt, test_dir, weights, voxel_size, truncation, scene)
                
            #     # load fused tsdf
            #     fused_tsdf = tsdf_path + '/' + scene + '.tsdf_filtered.hf5'
            #     # read tsdf
            #     f = h5py.File(fused_tsdf, 'r')
            #     fused_tsdf = np.array(f['TSDF_filtered']).astype(np.float16)

            #     sensor_weighting = tsdf_path + '/' + scene + '.sensor_weighting.hf5'
            #     f = h5py.File(sensor_weighting, 'r')
            #     sensor_weighting = np.array(f['sensor_weighting']).astype(np.float16)

            #     if config.FILTERING_MODEL.outlier_channel:
            #        sensor_weighting = sensor_weighting[0, :, :, :]

            #     visualize_features(proxy_sensor_weighting, sensor_weighting, fused_tsdf, sdf_gt, tsdfs, weights, features, test_dir, voxel_size, truncation, scene)

                
if __name__ == '__main__':

    # parse commandline arguments
    args = arg_parse()

    # load config
    test_config = loading.load_config_from_yaml(args['config'])

    test_fusion(test_config)