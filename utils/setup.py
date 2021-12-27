import os
import logging

from dataset import Replica

# the above import works because the __init__.py file
# in the dataset folder imports the Replica class from the module replica.
# we can import dataset in the package utils because the setup.py
# module is only called from the train or test scripts i.e. from a higher
# level.
from dataset import CoRBS
from dataset import Scene3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from torch.utils.tensorboard import SummaryWriter
import trimesh
import skimage.measure

from modules.database import Database

from utils import transform

from copy import copy

from utils.saving import *


def get_data_config(config, mode):
    data_config = copy(config.DATA)
    data_config.filtering_model = config.FILTERING_MODEL.model

    if mode == "train":
        data_config.mode = "train"
        data_config.scene_list = data_config.train_scene_list
    elif mode == "val":
        data_config.mode = "val"
        data_config.scene_list = data_config.val_scene_list
    elif mode == "test":
        data_config.mode = "test"
        data_config.scene_list = data_config.test_scene_list

    data_config.transform = transform.ToTensor()

    return data_config


def get_data(dataset, config):
    try:
        return eval(dataset)(config.DATA)
    except AttributeError:
        return eval(dataset)(config)


def get_database(dataset, config, mode="train"):

    # TODO: make this better
    database_config = copy(config.DATA)
    database_config.transform = transform.ToTensor()
    database_config.n_features = config.FEATURE_MODEL.n_features

    database_config.refinement = config.FILTERING_MODEL.CONV3D_MODEL.use_refinement
    database_config.test_mode = mode == "val" or mode == "test"
    database_config.alpha_supervision = config.LOSS.alpha_supervision
    database_config.visualize_features_and_proxy = (
        config.TESTING.visualize_features_and_proxy
    )
    database_config.outlier_channel = (
        config.FILTERING_MODEL.CONV3D_MODEL.outlier_channel
    )
    database_config.scene_list = eval("config.DATA.{}_scene_list".format(mode))

    return Database(dataset, database_config)


def get_workspace(config):
    workspace_path = os.path.join(config.SETTINGS.experiment_path, config.TIMESTAMP)
    workspace = Workspace(workspace_path)
    workspace.save_config(config)
    return workspace


def get_logger(path, name="training"):

    filehandler = logging.FileHandler(os.path.join(path, "{}.logs".format(name)), "a")
    consolehandler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    filehandler.setFormatter(formatter)
    consolehandler.setFormatter(formatter)

    logger = logging.getLogger(name)

    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)

    logger.addHandler(filehandler)  # set the new handler
    logger.addHandler(consolehandler)

    logger.setLevel(logging.DEBUG)

    return logger


class Workspace(object):
    def __init__(self, path):

        self.workspace_path = path
        self.model_path = os.path.join(path, "model")
        self.log_path = os.path.join(path, "logs")
        self.output_path = os.path.join(path, "output")

        os.makedirs(self.workspace_path)
        os.makedirs(self.model_path)
        os.makedirs(self.log_path)
        os.makedirs(self.output_path)

        self.writer = SummaryWriter(self.log_path)

        self._init_logger()

    def _init_logger(self):
        self.train_logger = get_logger(self.log_path, "training")
        self.val_logger = get_logger(self.log_path, "validation")

    def save_config(self, config):
        print("Saving config to ", self.workspace_path)
        save_config_to_json(self.workspace_path, config)

    def save_model_state(self, state, is_best, is_best_filt=None):
        save_checkpoint(state, is_best, self.model_path, is_best_filt)

    def save_alpha_histogram(self, database, sensors, epoch):

        for scene in database.scenes_gt.keys():
            mask = np.zeros_like(database.sensor_weighting[scene], dtype=bool)
            for sensor_ in sensors:
                mask = np.logical_or(
                    mask, (database.fusion_weights[sensor_][scene] > 0)
                )

            hist = database.sensor_weighting[scene][mask].flatten().astype(np.float32)
            plt.hist(hist, bins=100)
            plt.savefig(
                self.output_path
                + "/sensor_weighting_grid_histogram_"
                + scene
                + "_epoch_"
                + str(epoch)
                + ".png"
            )
            plt.clf()

    def log(self, message, mode="train"):
        if mode == "train":
            self.train_logger.info(message)
        elif mode == "val":
            self.val_logger.info(message)
