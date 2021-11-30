import torch
import argparse
import os

import numpy as np
import datetime

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

import h5py
import open3d as o3d

from evaluate_3d_reconstruction import run_evaluation

def arg_parse():
    parser = argparse.ArgumentParser(description='Script for testing RoutedFusion')

    parser.add_argument('--config', required=True)

    args = parser.parse_args()

    return vars(args)

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_fusion(config):
    
    # define output dir
    test_path = '/test_no_carving'
    time = '211019-170325' #datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    print(time)
    test_dir = config.SETTINGS.experiment_path + '/' + time + test_path
 
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
        config.FEATURE_MODEL.n_features = config.FEATURE_MODEL.append_pixel_conf + config.FEATURE_MODEL.append_depth # 1 for label encoding of noise in gaussian threshold data


    # get test database
    database = setup.get_database(dataset, config, mode='test')

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device)

    # only parameters when using routing network
    if config.ROUTING.do:
        if config.DATA.fusion_strategy == 'routingNet': # early fusion
            routing_checkpoint = torch.load(config.TESTING.routing_model_path)
            # print(routing_checkpoint)
            # load_model(config.TESTING.routing_model_path, pipeline._routing_network)
            # Keep line below until I see that the new loading function works.
            pipeline.fuse_pipeline._routing_network.load_state_dict(routing_checkpoint['pipeline_state_dict'])
        elif config.DATA.fusion_strategy == 'fusionNet':
            for sensor_ in config.DATA.input:
                checkpoint = torch.load(eval('config.TRAINING.routing_' + sensor_ + '_model_path'))
                pipeline.fuse_pipeline._routing_network[sensor_].load_state_dict(checkpoint['pipeline_state_dict'])


    pipeline.eval()

    sensors = config.DATA.input # ['tof', 'stereo'] # make sure thi only used when we have input: multidepth and fusion_strategy: fusionNet and derivatives
    
    # test model
    pipeline.test_tsdf_baseline(loader, dataset, database, sensors, device)

    # evaluate test scenes
    eval_results, eval_results_fused, \
    eval_results_scene, \
    eval_results_scene_fused = database.evaluate(mode='test')

    # save test_eval to log file
    logger = setup.get_logger(test_dir, 'test')
    for sensor in eval_results.keys():
        logger.info('Average ' + sensor + ' test results over test scenes')
        for metric in eval_results[sensor]:
            logger.info(metric + ': ' + str(eval_results[sensor][metric]))


    logger.info('Average test results over fused test scenes')
    for metric in eval_results_fused:
        logger.info(metric + ': ' + str(eval_results_fused[metric]))

    for sensor in eval_results_scene.keys():
        logger.info('Per scene ' + sensor + ' results')
        for scene in eval_results_scene[sensor]:
            logger.info('Scene: ' + scene)
            for metric in eval_results_scene[sensor][scene]:
                logger.info(metric + ': ' + str(eval_results_scene[sensor][scene][metric]))

    logger.info('Per scene fused results')
    for scene in eval_results_scene_fused:
        logger.info('Scene: ' + scene)
        for metric in eval_results_scene_fused[scene]:
            logger.info(metric + ': ' + str(eval_results_scene_fused[scene][metric]))

    # save ply-files of test scenes
    for scene_id in database.scenes_gt.keys():
        database.save(path=test_dir, scene_id=scene_id)

    # compute f-score for the test scenes and render them
    evaluate(database, config, test_dir)

def evaluate(database, config, test_dir):
    sdf_gt_path = os.getenv(config.DATA.root_dir) # when training on local scratch
        
    # os.getenv returns none when the input does not exist. When 
    # it returns none, we want to train on the work folder
    if not sdf_gt_path:
        sdf_gt_path  = config.DATA.root_dir

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
            mask = np.zeros_like(tsdf)
            and_mask = np.ones_like(tsdf)
            sensor_mask = dict()
            
            for sensor_ in config.DATA.input:
                # print(sensor_)
                weights = tsdf_path + '/' + scene + '_' + sensor_ + '.weights.hf5'
                f = h5py.File(weights, 'r')
                weights = np.array(f['weights']).astype(np.float16)
                mask = np.logical_or(mask, weights > weight_threshold)
                and_mask = np.logical_and(and_mask, weights > weight_threshold)
                sensor_mask[sensor_] = weights > weight_threshold
                # break

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
            mask_filtered = mask

            # return
            if len(config.DATA.input) > 1:
                # Generate visualization of the sensor weighting
                # load weighting sensor grid
                sensor_weighting = tsdf_path + '/' + scene + '.sensor_weighting.hf5'
                f = h5py.File(sensor_weighting, 'r')
                sensor_weighting = np.array(f['sensor_weighting']).astype(np.float16)

                # compute sensor weighting histogram and mesh visualization
                visualize_sensor_weighting(mesh, sensor_weighting, test_dir, mask, voxel_size, config.FILTERING_MODEL.outlier_channel)


            # Compute the F-score, precision and recall
            ply_path = model_test + '.ply'

            # evaluate F-score
            run_evaluation(ply_path, 'standard_trunc', scene, test_dir)

            # move the logs and plys to the evaluation dirs
            os.system('mv ' + test_dir + '/' + model_test + '.logs ' + test_dir + '/' + model_test + '/' + model_test + '.logs')
            os.system('mv ' + test_dir + '/' + model_test + '.ply ' + test_dir + '/' + model_test + '/' + model_test + '.ply')
            if len(config.DATA.input) > 1:
                os.system('mv ' + test_dir + '/sensor_weighting_nn.ply ' + test_dir + '/' + model_test + '/sensor_weighting_nn.ply')
                os.system('mv ' + test_dir + '/sensor_weighting_grid_histogram.png ' + test_dir + '/' + model_test + '/sensor_weighting_grid_histogram.png')
                os.system('mv ' + test_dir + '/sensor_weighting_surface_histogram.png ' + test_dir + '/' + model_test + '/sensor_weighting_surface_histogram.png')

            # return
            for sensor_ in config.DATA.input:
                model_test = scene + '_weight_threshold_' + str(weight_threshold)
                model_test = model_test + '_' + sensor_
                logger = get_logger(test_dir, name=model_test)

                tsdf = tsdf_path + '/' + scene + '_' + sensor_ + '.tsdf.hf5'
                weights = tsdf_path + '/' + scene + '_' + sensor_ + '.weights.hf5'

                # read tsdfs and weight grids
                f = h5py.File(tsdf, 'r')
                tsdf = np.array(f['TSDF']).astype(np.float16)
                # print(tsdf.astype(np.float32).sum())
                f = h5py.File(weights, 'r')
                weights = np.array(f['weights']).astype(np.float16)
                # print(weights.astype(np.float32).sum())

                # compute the L1, IOU and Acc
            
                mask = weights > weight_threshold

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
                # print(indices_x.shape)

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
                # print(np.asarray(mesh.vertices).shape)
                # print('isnan mesh verticies', np.isnan(np.asarray(mesh.vertices).sum()))
                del volume
                mesh.compute_vertex_normals()
                # o3d.visualization.draw_geometries([mesh])
                o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test + '.ply'), mesh)

                # # Compute the F-score, precision and recall
                ply_path = model_test + '.ply'

                # evaluate F-score
                run_evaluation(ply_path, 'standard_trunc', scene, test_dir)

                # # move the logs and plys to the evaluation dirs
                os.system('mv ' + test_dir + '/' + model_test + '.logs ' + test_dir + '/' + model_test + '/' + model_test + '.logs')
                os.system('mv ' + test_dir + '/' + model_test + '.ply ' + test_dir + '/' + model_test + '/' + model_test + '.ply')
  

if __name__ == '__main__':

    # parse commandline arguments
    args = arg_parse()

    # load config
    test_config = loading.load_config_from_yaml(args['config'])

    test_fusion(test_config)