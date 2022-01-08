import os
import json
import shutil
import torch


def save_config_to_json(path, config):
    """Saves config to json file"""
    with open(os.path.join(path, "config.json"), "w") as file:
        json.dump(config, file)


def save_checkpoint(state, is_best, checkpoint, is_best_filt=None):
    """Saves model and training parameters
    at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
       state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
       is_best: (dict) Dict of bools for each sensor. True at one sensor if it is the best model seen untill now
       checkpoint: (string) folder where parameters are to be saved
       is_best_filt: (bool) True if it is the best filtered model seen until now
    """
    if not os.path.exists(checkpoint):
        print(
            "Checkpoint Directory does not exist! Making directory {}".format(
                checkpoint
            )
        )
        os.mkdir(checkpoint)

    filepath = os.path.join(checkpoint, "last.pth.tar")
    torch.save(state, filepath)
    if is_best_filt:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))

    if isinstance(is_best, dict):
        for sensor in is_best.keys():
            if is_best[sensor]:
                shutil.copyfile(
                    filepath, os.path.join(checkpoint, "best_" + sensor + ".pth.tar")
                )
    else:
        if is_best:
            shutil.copyfile(
                filepath, os.path.join(checkpoint, "best.pth.tar")
            )  # train routing network with multiple sensor inputs
