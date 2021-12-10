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
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','sans-serif':['Times New Roman']})
rc('text', usetex=True)

def arg_parse():
    parser = argparse.ArgumentParser(description='Script for testing RoutedFusion')

    parser.add_argument('--config', required=True)

    args = parser.parse_args()

    return vars(args)

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_fusion(config):
    # define output dir
    test_path = '/test_plot_trajectory_office_0'
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
    if config.FEATURE_MODEL.use_feature_net:
        config.FEATURE_MODEL.n_features = config.FEATURE_MODEL.n_features + config.FEATURE_MODEL.append_depth
    else:
        config.FEATURE_MODEL.n_features = config.FEATURE_MODEL.append_depth # 1 for label encoding of noise in gaussian threshold data


    # get test database
    database = setup.get_database(dataset, config, mode='test')

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device)

    for sensor in config.DATA.input:
        if config.FUSION_MODEL.use_fusion_net:
            print('Fusion Net ', sensor, ': ', count_parameters(pipeline.fuse_pipeline._fusion_network[sensor]))
        print('Feature Net ', sensor, ': ', count_parameters(pipeline.fuse_pipeline._feature_network[sensor]))

    print('Filtering Net: ', count_parameters(pipeline.filter_pipeline._filtering_network))



    loading.load_pipeline(config.TESTING.fusion_model_path, pipeline) # this loads all parameters it can

    pipeline.eval()

    sensors = config.DATA.input # ['tof', 'stereo'] # make sure thi only used when we have input: multidepth and fusion_strategy: fusionNet and derivatives
    
    l1 = {}
    l2 = {}
    iou = {}
    acc = {}

 
    for scene in database.scenes_gt.keys():
        l1[scene] = {}
        l2[scene] = {}
        iou[scene] = {}
        acc[scene] = {}

        for sensor_ in sensors:
            l1[scene][sensor_] = []
            l2[scene][sensor_] = []
            iou[scene][sensor_] = []
            acc[scene][sensor_] = []

        l1[scene]['fused'] = []
        l2[scene]['fused'] = []
        iou[scene]['fused'] = []
        acc[scene]['fused'] = []

    # test model
    for i, batch in tqdm(enumerate(loader), total=len(dataset)):
        pipeline.test_step(batch, database, sensors, device)

        # evaluate test scenes
        eval_results, eval_results_fused, \
        eval_results_scene, \
        eval_results_scene_fused = database.evaluate(mode='test')

        for scene in database.scenes_gt.keys():
            for sensor_ in sensors:
                l1[scene][sensor_].append(eval_results_scene[sensor_][scene]['mad'])
                l2[scene][sensor_].append(eval_results_scene[sensor_][scene]['mse'])
                iou[scene][sensor_].append(eval_results_scene[sensor_][scene]['iou'])
                acc[scene][sensor_].append(eval_results_scene[sensor_][scene]['acc'])

            l1[scene]['fused'].append(eval_results_scene_fused[scene]['mad'])
            l2[scene]['fused'].append(eval_results_scene_fused[scene]['mse'])
            iou[scene]['fused'].append(eval_results_scene_fused[scene]['iou'])
            acc[scene]['fused'].append(eval_results_scene_fused[scene]['acc'])

        # convert lists to numpy arrays and save per frame
        for scene in database.scenes_gt.keys():
            for sensor_ in l1[scene].keys():
                l1[scene][sensor_] = np.asarray(l1[scene][sensor_])
                l2[scene][sensor_] = np.asarray(l2[scene][sensor_])
                iou[scene][sensor_] = np.asarray(iou[scene][sensor_])
                acc[scene][sensor_] = np.asarray(acc[scene][sensor_])

                # write to txt file
                np.savetxt(test_dir + '/' + 'mad_' + sensor_ + '.txt', l1[scene][sensor_])
                np.savetxt(test_dir + '/' + 'mse_' + sensor_ + '.txt', l2[scene][sensor_])
                np.savetxt(test_dir + '/' + 'iou_' + sensor_ + '.txt', iou[scene][sensor_])
                np.savetxt(test_dir + '/' + 'acc_' + sensor_ + '.txt', acc[scene][sensor_])

                l1[scene][sensor_] = l1[scene][sensor_].tolist()
                l2[scene][sensor_] = l2[scene][sensor_].tolist()
                iou[scene][sensor_] = iou[scene][sensor_].tolist()
                acc[scene][sensor_] = acc[scene][sensor_].tolist()

        # if i == 2:
        #     break

    x = np.linspace(1, len(dataset), len(dataset))
    # plot result over the trajectory
    for scene in database.scenes_gt.keys():
        plot(x, l1[scene]['tof'], l1[scene]['stereo'], l1[scene]['fused'],'MAD', test_dir)
        plot(x, l2[scene]['tof'], l2[scene]['stereo'], l2[scene]['fused'],'MSE', test_dir)
        plot(x, iou[scene]['tof'], iou[scene]['stereo'], iou[scene]['fused'],'IoU', test_dir)
        plot(x, acc[scene]['tof'], acc[scene]['stereo'], acc[scene]['fused'],'Accuracy', test_dir)


def plot(x, tof, stereo, fused, metric, test_dir):
    f = plt.figure()
    ax = plt.subplot(111)
    label_str = "Fused"
    ax.plot(
        x,
        fused,
        c="red",
        label=label_str,
        linewidth=2.0,
    )

    label_str = "ToF"
    ax.plot(
        x,
        tof,
        c="blue",
        label=label_str,
        linewidth=2.0,
    )

    label_str = "Stereo"
    ax.plot(
        x,
        stereo,
        c="green",
        label=label_str,
        linewidth=2.0,
    )

    ax.grid(True)

    plt.ylabel(metric, fontsize=20)
    plt.xlabel("Frames", fontsize=20)

    box = ax.get_position()
    ax.set_position([box.x0* 1.15, box.y0* 1.15, box.width * 1.05, box.height * 0.9])

    # Put a legend to the right of the current axis
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', ncol=3, prop={'size': 20})
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    png_name = test_dir + '/' + metric + '_plot.png'
    pdf_name = test_dir + '/' + metric + '_plot.pdf'

    # save figure and display
    f.savefig(png_name, format="png", bbox_inches="tight")
    f.savefig(pdf_name, format="pdf", bbox_inches="tight")

    # plt.show()


if __name__ == '__main__':

    # parse commandline arguments
    args = arg_parse()

    # load config
    test_config = loading.load_config_from_yaml(args['config'])

    test_fusion(test_config)