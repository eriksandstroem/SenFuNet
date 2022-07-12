import yaml
import json
import os
import torch

from easydict import EasyDict


def load_config_from_yaml(path):
    """
    Method to load the config file for
    neural network training
    :param path: yaml-filepath with configs stored
    :return: easydict containing config
    """
    c = yaml.safe_load(open(path))
    config = EasyDict(c)

    return config


def load_config_from_json(path):
    """
    Method to load the config file
    from json files.
    :param path: path to json file
    :return: easydict containing config
    """
    with open(path, "r") as file:
        data = json.load(file)
    config = EasyDict(data)
    return config


def load_config(path):
    """
    Wrapper method around different methods
    loading config file based on file ending.
    """

    if path[-4:] == "yaml":
        return load_config_from_yaml(path)
    elif path[-4:] == "json":
        return load_config_from_json(path)
    else:
        raise ValueError("Unsupported file format for config")


def load_pipeline(
    file, model
):  # loads all paramters that can be loaded in the checkpoint!

    checkpoint = file

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))

        model.load_state_dict(checkpoint["pipeline_state_dict"])
        print("loading full model")
    except:
        print("loading model partly")

        print(
            "nbr of entries in checkpoint model: ",
            len(checkpoint["pipeline_state_dict"].keys()),
        )
        pretrained_dict = {
            k: v
            for k, v in checkpoint["pipeline_state_dict"].items()
            if k in model.state_dict()
        }
        print("nbr of entries found in created model: ", len(model.state_dict().keys()))
        print(
            "nbr of entries found in created model and checkpoint model: ",
            len(pretrained_dict.keys()),
        )
        print("Keys in created model but not in checkpoint:")
        for key in model.state_dict().keys():
            if key not in checkpoint["pipeline_state_dict"].keys():
                print(key)
        print("...")
        print("Keys in checkpoint but not in created model")
        for key in checkpoint["pipeline_state_dict"].keys():
            if key not in model.state_dict().keys():
                print(key)

        model.load_state_dict(pretrained_dict, False)
