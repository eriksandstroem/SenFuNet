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

import h5py
import open3d as o3d


def arg_parse():
    parser = argparse.ArgumentParser(description='Script for testing RoutedFusion')

    parser.add_argument('--config', required=True)

    args = parser.parse_args()

    return vars(args)

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_fusion(config):
    
    # define output dir
    test_path = '/test'
    test_dir = config.SETTINGS.experiment_path + '/' + config.TESTING.fusion_model_path.split('/')[-3] + test_path

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if config.SETTINGS.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    config.FUSION_MODEL.device = device

    # get test dataset
    data_config = setup.get_data_config(test_config, mode='test')
    dataset = setup.get_data(test_config.DATA.dataset, data_config)
    loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=config.TESTING.test_batch_size,
                                            shuffle=config.TESTING.test_shuffle,
                                            pin_memory=True,
                                            num_workers=16)

    # get test database
    database = setup.get_database(dataset, test_config, mode='test')

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device)

    for sensor in config.DATA.input:
        print('Fusion Net ', sensor, ': ', count_parameters(pipeline.fuse_pipeline._fusion_network[sensor]))
        print('Feature Net ', sensor, ': ', count_parameters(pipeline.fuse_pipeline._feature_network[sensor]))

    print('Filtering Net: ', count_parameters(pipeline.filter_pipeline._filtering_network))



   # load pretrained routing model into parameters
    # if config.ROUTING.do:
    #     if config.DATA.fusion_strategy == 'routingNet':
    #         routing_checkpoint = torch.load(config.TESTING.routing_model_path)
    #         pipeline._routing_network.load_state_dict(routing_checkpoint['pipeline_state_dict'])
    #     elif config.DATA.fusion_strategy == 'fusionNet' or config.DATA.fusion_strategy == 'three_fusionNet' or config.DATA.fusion_strategy == 'fusionNet_conditioned':
    #         routing_mono_checkpoint = torch.load(config.TESTING.routing_mono_model_path)
    #         routing_stereo_checkpoint = torch.load(config.TESTING.routing_stereo_model_path)
    #         routing_tof_checkpoint = torch.load(config.TESTING.routing_tof_model_path)

    #         pipeline._routing_network_mono.load_state_dict(routing_mono_checkpoint['state_dict'])
    #         pipeline._routing_network_stereo.load_state_dict(routing_stereo_checkpoint['pipeline_state_dict'])
    #         pipeline._routing_network_tof.load_state_dict(routing_tof_checkpoint['pipeline_state_dict'])

    # load pipelines
    for sensor in config.DATA.input:
        loading.load_net(eval('config.TRAINING.pretraining_fusion_' + sensor +  '_model_path'), pipeline.fuse_pipeline._fusion_network[sensor], sensor)
        if config.FILTERING_MODEL.w_features:
            loading.load_net(config.TESTING.fusion_model_path, pipeline.fuse_pipeline._feature_network[sensor], sensor)

    loading.load_pipeline(config.TESTING.fusion_model_path, pipeline.filter_pipeline)
    pipeline.eval()

    sensors = config.DATA.input # ['tof', 'stereo'] # make sure thi only used when we have input: multidepth and fusion_strategy: fusionNet and derivatives
    
    # test model
    pipeline.test(loader, dataset, database, sensors, device)

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
        length = max_resolution*voxel_size

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
            for sensor_ in config.DATA.input:
                weights = tsdf_path + '/' + scene + '_' + sensor_ + '.weights.hf5'
                f = h5py.File(weights, 'r')
                weights = np.array(f['weights']).astype(np.float16)
                mask = np.logical_or(mask, weights > weight_threshold)

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

            if len(config.DATA.input) > 1:
                # Generate visualization of the sensor weighting
                # load weighting sensor grid
                sensor_weighting = tsdf_path + '/' + scene + '.sensor_weighting.hf5'
                f = h5py.File(sensor_weighting, 'r')
                sensor_weighting = np.array(f['sensor_weighting']).astype(np.float16)

                # compute sensor weighting histogram and mesh visualization
                visualize_sensor_weighting(mesh, sensor_weighting, test_dir, voxel_size)

            # Compute the F-score, precision and recall
            ply_path = model_test + '.ply'

            # run commandline command
            os.chdir(test_dir)

            print('running script: evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)
            os.system('evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)

            # move the logs and plys to the evaluation dirs
            os.system('mv ' + test_dir + '/' + model_test + '.logs ' + test_dir + '/' + model_test + '/' + model_test + '.logs')
            os.system('mv ' + test_dir + '/' + model_test + '.ply ' + test_dir + '/' + model_test + '/' + model_test + '.ply')
            if len(config.DATA.input) > 1:
                os.system('mv ' + test_dir + '/sensor_weighting.ply ' + test_dir + '/' + model_test + '/sensor_weighting.ply')
                os.system('mv ' + test_dir + '/sensor_weighting_histogram.png ' + test_dir + '/' + model_test + '/sensor_weighting_histogram.png')

            for sensor_ in config.DATA.input:
                model_test = scene + '_weight_threshold_' + str(weight_threshold)
                model_test = model_test + '_' + sensor_
                logger = get_logger(test_dir, name=model_test)

                tsdf = tsdf_path + '/' + scene + '_' + sensor_ + '.tsdf.hf5'
                weights = tsdf_path + '/' + scene + '_' + sensor_ + '.weights.hf5'

                # read tsdfs and weight grids
                f = h5py.File(tsdf, 'r')
                tsdf = np.array(f['TSDF']).astype(np.float16)
                f = h5py.File(weights, 'r')
                weights = np.array(f['weights']).astype(np.float16)

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

                # Compute the F-score, precision and recall
                ply_path = model_test + '.ply'

                # run commandline command
                os.chdir(test_dir)

                print('running script: evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)
                os.system('evaluate_3d_reconstruction.py ' + ply_path + ' standard_trunc ' + scene)

                # move the logs and plys to the evaluation dirs
                os.system('mv ' + test_dir + '/' + model_test + '.logs ' + test_dir + '/' + model_test + '/' + model_test + '.logs')
                os.system('mv ' + test_dir + '/' + model_test + '.ply ' + test_dir + '/' + model_test + '/' + model_test + '.ply')
  




if __name__ == '__main__':

    # parse commandline arguments
    args = arg_parse()

    # load config
    test_config = loading.load_config_from_yaml(args['config'])

    test_fusion(test_config)