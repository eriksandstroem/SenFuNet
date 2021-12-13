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


def load_experiment(path):
    """
    Method to load experiment from path
    :param path: path to experiment folder
    :return: easydict containing config
    """
    path = os.path.join(path, "config.json")
    config = load_config_from_json(path)
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


def load_model(file, model):

    checkpoint = file

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
    except:
        print("loading model partly")
        pretrained_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if k in model.state_dict()
        }
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())


# def load_pipeline(file, model):

#     checkpoint = file

#     if not os.path.exists(checkpoint):
#         raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
#     try:
#         if torch.cuda.is_available():
#             checkpoint = torch.load(checkpoint)
#         else:
#             checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
#         model.load_state_dict(checkpoint['pipeline_state_dict'])
#     except:
#         print('loading model partly')
#         pretrained_dict = {k: v for k, v in checkpoint['pipeline_state_dict'].items() if k in model.state_dict()}
#         model.state_dict().update(pretrained_dict)
#         model.load_state_dict(model.state_dict())


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
        # print(model.state_dict().keys())
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

        # USE THESE TWO LINES
        # model.state_dict().update(pretrained_dict)
        # model.load_state_dict(model.state_dict())
        # OR THESE TWO LINES TO LOAD (UNCLEAR WHICH IS BEST, sometimes one works, but the other does not)
        model.load_state_dict(pretrained_dict, False)


def load_net_old(
    file, model, sensor
):  # to load fusion net weights from model that was trained with the old setup i.e. pipeline._fusion_net_tof.etc...

    checkpoint = file

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))

    print("loading model partly")
    # print(model.state_dict().keys())
    # print(checkpoint['pipeline_state_dict'].keys())
    print(
        "nbr of entries in checkpoint model: ",
        len(checkpoint["pipeline_state_dict"].keys()),
    )
    # only load the model parameters in checkpoint related to the sensor
    sensor_specific_checkpoint = {
        ".".join(k.split(".")[1:]): v
        for k, v in checkpoint["pipeline_state_dict"].items()
        if k.split(".")[0].endswith(sensor)
    }  # and k.split('.')[0].startswith('_fusion')}
    print(
        "nbr of entries in sensor specific checkpoint model: ",
        len(sensor_specific_checkpoint.keys()),
    )
    # check so that the sensor specific checkpoint weight names are in the model
    pretrained_dict = {
        k: v for k, v in sensor_specific_checkpoint.items() if k in model.state_dict()
    }
    print("nbr of entries found in created model: ", len(model.state_dict().keys()))
    print(
        "nbr of entries found in created model and sensor specific checkpoint model: ",
        len(pretrained_dict.keys()),
    )
    # for key in model.state_dict().keys():
    #     if key not in pretrained_dict.keys():
    #         print(key)

    # print(pretrained_dict.keys())
    # print(checkpoint['pipeline_state_dict'].keys())
    # print(model.state_dict().keys())
    # model.state_dict().update(pretrained_dict)
    # model.load_state_dict(model.state_dict())
    model.load_state_dict(pretrained_dict, False)


def load_net(file, model, sensor):

    checkpoint = file

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))

    print("loading model partly")
    # print(checkpoint['pipeline_state_dict'].keys())
    # print(model.state_dict().keys())
    print(
        "nbr of entries in checkpoint model: ",
        len(checkpoint["pipeline_state_dict"].keys()),
    )
    # only load the model parameters in checkpoint related to the sensor
    sensor_specific_checkpoint = {
        ".".join(k.split(".")[3:]): v
        for k, v in checkpoint["pipeline_state_dict"].items()
        if k.split(".")[2].endswith(sensor)
    }  # and k.split('.')[0].startswith('_fusion')}
    # print(sensor_specific_checkpoint.keys())
    print(
        "nbr of entries in sensor specific checkpoint model: ",
        len(sensor_specific_checkpoint.keys()),
    )
    # check so that the sensor specific checkpoint weight names are in the model
    pretrained_dict = {
        k: v for k, v in sensor_specific_checkpoint.items() if k in model.state_dict()
    }
    print("nbr of entries found in created model: ", len(model.state_dict().keys()))
    print(
        "nbr of entries found in created model and sensor specific checkpoint model: ",
        len(pretrained_dict.keys()),
    )
    # for key in model.state_dict().keys():
    #     if key not in pretrained_dict.keys():
    #         print(key)
    # print(pretrained_dict.keys())
    # print(checkpoint['pipeline_state_dict'].keys())
    # print(model.state_dict().keys())
    # model.state_dict().update(pretrained_dict)
    # model.load_state_dict(model.state_dict())
    model.load_state_dict(pretrained_dict, False)


def load_pipeline_stereo(file, model, sensor):
    def insert(k, sensor):
        if sensor == "tof":
            if k.startswith("_r"):
                k = k[:16] + "_tof" + k[16:]
            else:
                k = k[:15] + "_tof" + k[15:]
        elif sensor == "stereo":
            if k.startswith("_r"):
                k = k[:16] + "_stereo" + k[16:]
            else:
                k = k[:15] + "_stereo" + k[15:]
        return k

    checkpoint = file

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
        # print(checkpoint['pipeline_state_dict'].keys())
        model.load_state_dict(checkpoint["pipeline_state_dict"])
        print("loading full")
    except:
        print("loading model partly")
        # print(checkpoint['pipeline_state_dict'].keys())
        print(
            "nbr of entries in checkpoint model: ",
            len(checkpoint["pipeline_state_dict"].keys()),
        )
        # print()
        pretrained_dict = {
            insert(k, sensor): v
            for k, v in checkpoint["pipeline_state_dict"].items()
            if insert(k, sensor) in model.state_dict()
        }
        print("nbr of entries found in created model: ", len(model.state_dict().keys()))
        print(
            "nbr of entries found in created model and checkpoint model: ",
            len(pretrained_dict.keys()),
        )
        # for key in model.state_dict().keys():
        #     if key not in pretrained_dict.keys():
        #         print(key)
        # print(pretrained_dict.keys())
        # print(checkpoint['pipeline_state_dict'].keys())
        # print(model.state_dict().keys())
        # model.state_dict().update(pretrained_dict)
        # model.load_state_dict(model.state_dict())
        model.load_state_dict(pretrained_dict, False)


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
    except:
        print("loading model partly")
        pretrained_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if k in model.state_dict()
        }
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint
