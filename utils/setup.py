import torch
import os
import logging
from dataset import ShapeNet
from dataset import Replica # core dumped
from dataset import CoRBS
 
from torch.utils.tensorboard import SummaryWriter
import trimesh
import skimage.measure
# 
from modules.database import Database # core dumped

from utils import transform

from easydict import EasyDict
from copy import copy

from utils.saving import *

def get_data_config(config, mode):

    data_config = copy(config.DATA)

    if mode == 'train':
        data_config.mode = 'train'
        data_config.scene_list = data_config.train_scene_list
    elif mode == 'val':
        data_config.mode = 'val'
        data_config.scene_list = data_config.val_scene_list
    elif mode == 'test':
        data_config.mode = 'test'
        data_config.scene_list = data_config.test_scene_list

    data_config.transform = transform.ToTensor()

    return data_config


def get_data(dataset, config):

    try:
        return eval(dataset)(config.DATA)
    except AttributeError:
        return eval(dataset)(config)


def get_database(dataset, config, mode='train'):

    #TODO: make this better
    database_config = copy(config.DATA)
    database_config.transform = transform.ToTensor()
    database_config.erosion = config.FILTERING_MODEL.erosion
    database_config.n_features = config.FEATURE_MODEL.n_features
    database_config.features_to_sdf_enc = config.FILTERING_MODEL.features_to_sdf_enc
    database_config.features_to_weight_head = config.FILTERING_MODEL.features_to_weight_head
    database_config.test_mode = config.SETTINGS.test_mode
    database_config.scene_list = eval('config.DATA.{}_scene_list'.format(mode))

    return Database(dataset, database_config)


def get_workspace(config):
    workspace_path = os.path.join(config.SETTINGS.experiment_path,
                                  config.TIMESTAMP)
    workspace = Workspace(workspace_path)
    workspace.save_config(config)
    return workspace


def get_logger(path, name='training'):

    filehandler = logging.FileHandler(os.path.join(path, '{}.logs'.format(name)), 'a')
    consolehandler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    filehandler.setFormatter(formatter)
    consolehandler.setFormatter(formatter)

    logger = logging.getLogger(name)

    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)

    logger.addHandler(filehandler)  # set the new handler
    logger.addHandler(consolehandler)

    logger.setLevel(logging.DEBUG)

    return logger


def save_tsdf(filename, data):
    with h5py.File(filename, 'w') as file:
        file.create_dataset('TSDF',
                            shape=data.shape,
                            data=data,
                            compression='gzip',
                            compression_opts=9)

def save_weights(filename, data):
    with h5py.File(filename, 'w') as file:
        file.create_dataset('weights',
                            shape=data.shape,
                            data=data,
                            compression='gzip', 
                            compression_opts=9)

def save_ply(filename, data):
    voxel_size = 0.01
    vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(data, 
        level=0, spacing=(voxel_size, voxel_size, voxel_size))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.export(filename)


class Workspace(object):

    def __init__(self, path):

        self.workspace_path = path
        self.model_path = os.path.join(path, 'model')
        self.log_path = os.path.join(path, 'logs')
        self.output_path = os.path.join(path, 'output')

        os.makedirs(self.workspace_path)
        os.makedirs(self.model_path)
        os.makedirs(self.log_path)
        os.makedirs(self.output_path)

        self.writer = SummaryWriter(self.log_path)

        self._init_logger()

    def _init_logger(self):
        self.train_logger = get_logger(self.log_path, 'training')
        self.val_logger = get_logger(self.log_path, 'validation')

    def save_config(self, config):
        print('Saving config to ', self.workspace_path)
        save_config_to_json(self.workspace_path, config)

    def save_model_state(self, state, is_best_filt, is_best):
        save_checkpoint(state, is_best_filt, is_best, self.model_path)

    def save_tsdf_data(self, file, data):
        tsdf_file = os.path.join(self.output_path, file)
        save_tsdf(tsdf_file, data)

    def save_weights_data(self, file, data):
        weight_files = os.path.join(self.output_path, file)
        save_weights(weight_files, data)

    def save_ply_data(self, file, data):
        ply_files = os.path.join(self.output_path, file)
        save_ply(ply_files, data)

    def log(self, message, mode='train'):
        if mode == 'train':
            self.train_logger.info(message)
        elif mode == 'val':
            self.val_logger.info(message)

