import os
import json
import shutil
import torch
import h5py


def save_config_to_json(path, config):
    """Saves config to json file
    """
    with open(os.path.join(path, 'config.json'), 'w') as file:
        json.dump(config, file)

def save_checkpoint(state, is_best, is_best_tof, is_best_stereo, checkpoint):
    """Saves model and training parameters
    at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
       state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
       is_best: (bool) True if it is the best model seen till now
       checkpoint: (string) folder where parameters are to be saved
    """
    if not os.path.exists(checkpoint):
       print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
       os.mkdir(checkpoint)


    filepath = os.path.join(checkpoint, 'last.pth.tar')
    torch.save(state, filepath)
    if is_best:
      shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
    if is_best_tof:
      shutil.copyfile(filepath, os.path.join(checkpoint, 'best_tof.pth.tar'))
    if is_best_stereo:
      shutil.copyfile(filepath, os.path.join(checkpoint, 'best_stereo.pth.tar'))

