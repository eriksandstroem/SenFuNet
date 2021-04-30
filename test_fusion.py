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
    test_path = '/test_fusion_net_late_fusion_210426-153135'
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

    print('Fusion Net ToF: ', count_parameters(pipeline._fusion_network_tof))
    print('Fusion Net stereo: ', count_parameters(pipeline._fusion_network_stereo))
    print('Feature Net ToF: ', count_parameters(pipeline._feature_network_tof))
    print('Feature Net stereo: ', count_parameters(pipeline._feature_network_stereo))
    print('Filtering Net: ', count_parameters(pipeline._filtering_network))
    print('All Parameters: ', count_parameters(pipeline))

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

    loading.load_filtering(config.TESTING.fusion_model_path, pipeline) # this loads all parameters available in the checkpoint that can be loaded
    # When you want to load both fusion nets and filtering net in one file, then comment out the two lines below
    # load same fusion net for all models
    loading.load_pipeline(config.TRAINING.pretraining_fusion_tof_model_path, pipeline, 'tof')
    loading.load_pipeline(config.TRAINING.pretraining_fusion_stereo_model_path, pipeline, 'stereo')
    # loading.load_pipeline_stereo(config.TRAINING.pretraining_fusion_stereo_model_path, pipeline, 'stereo') # to load from routedfusion_fusion_only


    # for name, param in pipeline.named_parameters():
    #     print(name)
    #     print(torch.norm(param))
                    # print('grad norm: ', torch.norm(param.grad))
                    # print('val norm: ' , torch.norm(param))
                # print(name, param.grad)

    # fuse frames indep tof and stereo models loading
    # stereo_path = '/home/esandstroem/scratch-second/opportunistic_3d_capture/Erik_3D_Reconstruction_Project/src/RoutedFusion/workspace/fusion/stereo/model/best.pth.tar'
    # tof_path = '/home/esandstroem/scratch-second/opportunistic_3d_capture/Erik_3D_Reconstruction_Project/src/RoutedFusion/workspace/fusion/tof/model/best.pth.tar'
    # fusion_stereo_checkpoint = torch.load(stereo_path)
    # fusion_tof_checkpoint = torch.load(tof_path)
    
    # for key in fusion_stereo_checkpoint['pipeline_state_dict'].copy():
    #     if key.startswith('_f'):
    #         fusion_stereo_checkpoint['pipeline_state_dict'][key[16:]] = fusion_stereo_checkpoint['pipeline_state_dict'].pop(key)
    #     elif key.startswith('_r'):
    #         fusion_stereo_checkpoint['pipeline_state_dict'].pop(key)

    # for key in fusion_tof_checkpoint['pipeline_state_dict'].copy():
    #     if key.startswith('_f'):
    #         fusion_tof_checkpoint['pipeline_state_dict'][key[16:]] = fusion_tof_checkpoint['pipeline_state_dict'].pop(key)
    #     elif key.startswith('_r'):
    #         fusion_tof_checkpoint['pipeline_state_dict'].pop(key)

    # pipeline._fusion_network_stereo.load_state_dict(fusion_stereo_checkpoint['pipeline_state_dict'])
    # pipeline._fusion_network_tof.load_state_dict(fusion_tof_checkpoint['pipeline_state_dict'])

    # fuse two fusionNet loading
    # fusion_checkpoint = torch.load(config.TESTING.fusion_model_path)
    # fusion_stereo_checkpoint = dict()
    # fusion_tof_checkpoint = dict()

    # for key in fusion_checkpoint['pipeline_state_dict'].copy():
    #     if key.startswith('_fusion_network_stereo'):
    #         fusion_stereo_checkpoint[key[23:]] = fusion_checkpoint['pipeline_state_dict'][key]
    #     elif key.startswith('_fusion_network_tof'):
    #         fusion_tof_checkpoint[key[20:]] = fusion_checkpoint['pipeline_state_dict'][key]

    # print(fusion_tof_checkpoint)
    # pipeline._fusion_network_stereo.load_state_dict(fusion_stereo_checkpoint)
    # pipeline._fusion_network_tof.load_state_dict(fusion_tof_checkpoint)


    # fusionNet_conditioned Loading
    # fusion_checkpoint = torch.load(config.TESTING.fusion_model_path)

    # for key in fusion_checkpoint['pipeline_state_dict'].copy():
    #     if key.startswith('_f'):
    #         fusion_checkpoint['pipeline_state_dict'][key[16:]] = fusion_checkpoint['pipeline_state_dict'].pop(key)
    #     elif key.startswith('_r'):
    #         fusion_checkpoint['pipeline_state_dict'].pop(key)

    # pipeline._fusion_network.load_state_dict(fusion_checkpoint['pipeline_state_dict'])

    pipeline.eval()

    if config.FILTERING_MODEL.fuse_sensors:
        sensors = ['tof', 'stereo'] # make sure thi only used when we have input: multidepth and fusion_strategy: fusionNet and derivatives
    else:
        sensors = ['tof']
    sensor_opposite = {'tof': 'stereo', 'stereo': 'tof'}

    print(len(dataset))
    for i, batch in tqdm(enumerate(loader), total=len(dataset)):

        if config.DATA.input == 'multidepth' and (config.DATA.fusion_strategy == 'fusionNet' or config.DATA.fusion_strategy == 'two_fusionNet' or config.DATA.fusion_strategy == 'fusionNet_conditioned'):
            # randomly integrate the three sensors
            random.shuffle(sensors)
            for sensor in sensors:
                if sensor == 'tof' and i % config.DATA.sampling_density_tof != 0:
                    continue
                if sensor == 'stereo' and i % config.DATA.sampling_density_stereo != 0:
                    continue
                # print(sensor) # for debugging
                # print(batch['frame_id']) # for debugging
                batch['depth'] = batch[sensor + '_depth']
                batch['confidence_threshold'] = eval('config.ROUTING.threshold_' + sensor) 
                if config.DATA.fusion_strategy == 'two_fusionNet':
                    batch['fusion_net'] = 'self._fusion_network_' + sensor
                    batch['feature_net'] = 'self._feature_network_' + sensor
                batch['routing_net'] = 'self._routing_network_' + sensor
                batch['mask'] = batch[sensor + '_mask']
                batch['sensor'] = sensor
                # print(sensor)
                batch['sensor_opposite'] = sensor_opposite[sensor]
                # batch = transform.to_device(batch, device) # should not be needed

                pipeline.fuse(batch, database, device)

        else:
            # print(batch['frame_id']) # for debugging
            # put all data on GPU
            batch = transform.to_device(batch, device)
            # fusion pipeline
            pipeline.fuse(batch, database, device)

        # if i == 5: # for debugging
        #     break

    # filter outliers
    # database.filter(value=config.TRAINING.outlier_filter_val)

    for scene in database.filtered.keys():   
        pipeline.sensor_fusion(scene, database, device)


    # evaluate test scenes
    eval_results_tof, eval_results_stereo, eval_results_fused, \
    eval_results_scene_tof, eval_results_scene_stereo, \
    eval_results_scene_fused = database.evaluate(mode='test')

    # save test_eval to log file
    logger = setup.get_logger(test_dir, 'test')
    logger.info('Average tof test results over test scenes')
    for metric in eval_results_tof:
        logger.info(metric + ': ' + str(eval_results_tof[metric]))

    # logger.info('Average stereo test results over test scenes')
    # for metric in eval_results_stereo:
    #     logger.info(metric + ': ' + str(eval_results_stereo[metric]))

    logger.info('Average test results over fused test scenes')
    for metric in eval_results_fused:
        logger.info(metric + ': ' + str(eval_results_fused[metric]))


    logger.info('Per scene tof results')
    for scene in eval_results_scene_tof:
        logger.info('Scene: ' + scene)
        for metric in eval_results_scene_tof[scene]:
            logger.info(metric + ': ' + str(eval_results_scene_tof[scene][metric]))

    # logger.info('Per scene stereo results')
    # for scene in eval_results_scene_stereo:
    #     logger.info('Scene: ' + scene)
    #     for metric in eval_results_scene_stereo[scene]:
    #         logger.info(metric + ': ' + str(eval_results_scene_stereo[scene][metric]))

    logger.info('Per scene fused results')
    for scene in eval_results_scene_fused:
        logger.info('Scene: ' + scene)
        for metric in eval_results_scene_fused[scene]:
            logger.info(metric + ': ' + str(eval_results_scene_fused[scene][metric]))

    # save ply-files of test scenes
    for scene_id in database.scenes_gt.keys():
        database.save(path=test_dir, save_mode=config.SETTINGS.save_mode, scene_id=scene_id)

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
        for weight_threshold in weight_thresholds:
            model_test = scene + '_weight_threshold_' + str(weight_threshold)
            model_test_filtered = model_test + '_filtered'
            model_test_tof = model_test + '_tof'
            # model_test_stereo = model_test + '_stereo'


            logger = get_logger(test_dir, name=model_test_filtered)
            logger_tof = get_logger(test_dir, name=model_test_tof)
            # logger_stereo = get_logger(test_dir, name=model_test_stereo)

            tsdf_path = test_dir

            sdf_gt = sdf_gt_path + '/' + scene + '/sdf_' + scene + '.hdf' 

            tsdf = tsdf_path + '/' + scene + '.tsdf_filtered.hf5'
            tsdf_tof = tsdf_path + '/' + scene + '_tof.tsdf.hf5'
            weights_tof = tsdf_path + '/' + scene + '_tof.weights.hf5'
            # tsdf_stereo = tsdf_path + '/' + scene + '_stereo.tsdf.hf5'
            # weights_stereo = tsdf_path + '/' + scene + '_stereo.weights.hf5'

            # read tsdfs and weight grids
            f = h5py.File(tsdf, 'r')
            tsdf = np.array(f['TSDF_filtered']).astype(np.float16)
            f = h5py.File(tsdf_tof, 'r')
            tsdf_tof = np.array(f['TSDF']).astype(np.float16)
            f = h5py.File(weights_tof, 'r')
            weights_tof = np.array(f['weights']).astype(np.float16)
            # f = h5py.File(tsdf_stereo, 'r')
            # tsdf_stereo = np.array(f['TSDF']).astype(np.float16)
            # f = h5py.File(weights_stereo, 'r')
            # weights_stereo = np.array(f['weights']).astype(np.float16)

            # compute the L1, IOU and Acc on the fused_grid
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

            mask = weights_tof > weight_threshold #np.logical_or(weights_tof > weight_threshold, weights_stereo > weight_threshold)
            mask_tof = weights_tof > weight_threshold
            # mask_stereo = weights_stereo > weight_threshold

            # erode masks appropriately
            if config.FILTERING_MODEL.erosion:
                mask = ndimage.binary_erosion(mask, structure=np.ones((3,3,3)), iterations=1)
                mask_tof = ndimage.binary_erosion(mask_tof, structure=np.ones((3,3,3)), iterations=1)
                # mask_stereo = ndimage.binary_erosion(mask_stereo, structure=np.ones((3,3,3)), iterations=1)

            eval_results_scene = evaluation(tsdf, sdf_gt, mask)
            eval_results_scene_tof = evaluation(tsdf_tof, sdf_gt, mask_tof)
            # eval_results_scene_stereo = evaluation(tsdf_stereo, sdf_gt, mask_stereo)

            logger.info('Test Scores for scene: ' + scene)
            logger_tof.info('Test Scores for scene: ' + scene)
            # logger_stereo.info('Test Scores for scene: ' + scene)
            for key in eval_results_scene:
                logger.info(key + ': ' + str(eval_results_scene[key]))
                logger_tof.info(key + ': ' + str(eval_results_scene_tof[key]))
                # logger_stereo.info(key + ': ' + str(eval_results_scene_stereo[key]))

            # Create the mesh using the given mask
            resolution = sdf_gt.shape
            max_resolution = np.array(resolution).max()
            tsdf_cube = np.zeros((max_resolution, max_resolution, max_resolution))
            tsdf_cube[:resolution[0], :resolution[1], :resolution[2]] = tsdf

            tsdf_cube_tof = np.zeros((max_resolution, max_resolution, max_resolution))
            tsdf_cube_tof[:resolution[0], :resolution[1], :resolution[2]] = tsdf_tof

            # tsdf_cube_stereo = np.zeros((max_resolution, max_resolution, max_resolution))
            # tsdf_cube_stereo[:resolution[0], :resolution[1], :resolution[2]] = tsdf_stereo

            indices_x = mask.nonzero()[0]
            indices_y = mask.nonzero()[1]
            indices_z = mask.nonzero()[2]
            # compare with standard marching cubes result
            voxel_size = f.attrs['voxel_size'] 

            length = max_resolution*f.attrs['voxel_size'] 

            volume = o3d.integration.UniformTSDFVolume(
                    length=length,
                    resolution=max_resolution,
                    sdf_trunc=truncation,
                    color_type=o3d.integration.TSDFVolumeColorType.RGB8)
            
            for i in range(indices_x.shape[0]):
                volume.set_tsdf_at(tsdf_cube[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
                volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])               

            print("Extract a filtered triangle mesh from the volume and visualize it.")
            mesh = volume.extract_triangle_mesh()
            del volume
            mesh.compute_vertex_normals()
            # o3d.visualization.draw_geometries([mesh])
            o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test_filtered + '.ply'), mesh)

            indices_x = mask_tof.nonzero()[0]
            indices_y = mask_tof.nonzero()[1]
            indices_z = mask_tof.nonzero()[2]

            volume_tof = o3d.integration.UniformTSDFVolume(
                    length=length,
                    resolution=max_resolution,
                    sdf_trunc=truncation,
                    color_type=o3d.integration.TSDFVolumeColorType.RGB8)

            tof = False
            if indices_x.shape[0] > 0:
                print('Rendering tof')
                tof = True
                for i in range(indices_x.shape[0]):
                    volume_tof.set_tsdf_at(tsdf_cube_tof[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
                    volume_tof.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])   

                print("Extract a noisy triangle mesh from the volume and visualize it.")
                mesh = volume_tof.extract_triangle_mesh()
                del volume_tof
                mesh.compute_vertex_normals()
                # o3d.visualization.draw_geometries([mesh])
                o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test_tof + '.ply'), mesh)

            # indices_x = mask_stereo.nonzero()[0]
            # indices_y = mask_stereo.nonzero()[1]
            # indices_z = mask_stereo.nonzero()[2]

            # volume_stereo = o3d.integration.UniformTSDFVolume(
            #         length=length,
            #         resolution=max_resolution,
            #         sdf_trunc=0.1,
            #         color_type=o3d.integration.TSDFVolumeColorType.RGB8)

            # stereo = False
            # if indices_x.shape[0] > 0:
            #     print('Rendering stereo')
            #     stereo = True
            #     for i in range(indices_x.shape[0]):
            #         volume_stereo.set_tsdf_at(tsdf_cube_stereo[indices_x[i], indices_y[i], indices_z[i]], indices_x[i] , indices_y[i], indices_z[i])
            #         volume_stereo.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i])   

            #     print("Extract a noisy triangle mesh from the volume and visualize it.")
            #     mesh = volume_stereo.extract_triangle_mesh()
            #     del volume_stereo
            #     mesh.compute_vertex_normals()
            #     # o3d.visualization.draw_geometries([mesh])
            #     o3d.io.write_triangle_mesh(os.path.join(test_dir, model_test_stereo + '.ply'), mesh)

            # Compute the F-score, precision and recall
            filtered_ply_path = model_test_filtered + '.ply'
            tof_ply_path = model_test_tof + '.ply'
            # stereo_ply_path = model_test_stereo + '.ply'

            # run commandline command
            os.chdir(test_dir)

            print('running script: evaluate_3d_reconstruction.py ' + filtered_ply_path + ' standard_trunc ' + scene)
            os.system('evaluate_3d_reconstruction.py ' + filtered_ply_path + ' standard_trunc ' + scene)
            if tof:
                print('running script: evaluate_3d_reconstruction.py ' + tof_ply_path + ' standard_trunc ' + scene)
                os.system('evaluate_3d_reconstruction.py ' + tof_ply_path + ' standard_trunc ' + scene)
            # if stereo:
            #     print('running script: evaluate_3d_reconstruction.py ' + stereo_ply_path + ' standard_trunc ' + scene)
            #     os.system('evaluate_3d_reconstruction.py ' + stereo_ply_path + ' standard_trunc ' + scene)

            # move the logs and plys to the evaluation dirs
            os.system('mv ' + test_dir + '/' + model_test_filtered + '.logs ' + test_dir + '/' + model_test_filtered + '/' + model_test_filtered + '.logs')
            os.system('mv ' + test_dir + '/' + model_test_filtered + '.ply ' + test_dir + '/' + model_test_filtered + '/' + model_test_filtered + '.ply')
            if tof:
                os.system('mv ' + test_dir + '/' + model_test_tof + '.logs ' + test_dir + '/' + model_test_tof + '/' + model_test_tof + '.logs')
                os.system('mv ' + test_dir + '/' + model_test_tof + '.ply ' + test_dir + '/' + model_test_tof + '/' + model_test_tof + '.ply')
            # if stereo:
            #     os.system('mv ' + test_dir + '/' + model_test_stereo + '.logs ' + test_dir + '/' + model_test_stereo + '/' + model_test_stereo + '.logs')
            #     os.system('mv ' + test_dir + '/' + model_test_stereo + '.ply ' + test_dir + '/' + model_test_stereo + '/' + model_test_stereo + '.ply')
          
            




if __name__ == '__main__':

    # parse commandline arguments
    args = arg_parse()

    # load config
    test_config = loading.load_config_from_yaml(args['config'])

    test_fusion(test_config)