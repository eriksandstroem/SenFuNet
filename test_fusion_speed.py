import torch
import argparse
import os

from utils import loading
from utils import setup

from modules.pipeline import Pipeline


def arg_parse():
    parser = argparse.ArgumentParser(description="Script for testing RoutedFusion")

    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    return vars(args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_fusion(config):
    # define output dir
    test_path = "/test_no_carving"
    test_dir = (
        config.SETTINGS.experiment_path
        + "/"
        + config.TESTING.fusion_model_path.split("/")[-3]
        + test_path
    )

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if config.SETTINGS.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    config.FUSION_MODEL.device = device

    # get test dataset
    data_config = setup.get_data_config(config, mode="test")
    dataset = setup.get_data(config.DATA.dataset, data_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.TESTING.test_batch_size,
        shuffle=config.TESTING.test_shuffle,
        pin_memory=True,
        num_workers=16,
    )

    # specify number of features
    if config.FEATURE_MODEL.use_feature_net:
        config.FEATURE_MODEL.n_features = (
            config.FEATURE_MODEL.n_features + config.FEATURE_MODEL.append_depth
        )
    else:
        config.FEATURE_MODEL.n_features = (
            config.FEATURE_MODEL.append_depth + 3 * config.FEATURE_MODEL.w_rgb
        )  # 1 for label encoding of noise in gaussian threshold data

    # get test database
    database = setup.get_database(dataset, config, mode="test")

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device)

    for sensor in config.DATA.input:
        if config.FUSION_MODEL.use_fusion_net:
            print(
                "Fusion Net ",
                sensor,
                ": ",
                count_parameters(pipeline.fuse_pipeline._fusion_network[sensor]),
            )
        print(
            "Feature Net ",
            sensor,
            ": ",
            count_parameters(pipeline.fuse_pipeline._feature_network[sensor]),
        )

    if pipeline.filter_pipeline is not None:
        print(
            "Filtering Net: ",
            count_parameters(pipeline.filter_pipeline._filtering_network),
        )

    loading.load_pipeline(
        config.TESTING.fusion_model_path, pipeline
    )  # this loads all parameters it can

    pipeline.eval()

    sensors = config.DATA.input

    # test model
    pipeline.test_speed(loader, dataset, database, sensors, device)


if __name__ == "__main__":

    # parse commandline arguments
    args = arg_parse()

    # load config
    test_config = loading.load_config_from_yaml(args["config"])

    test_fusion(test_config)
