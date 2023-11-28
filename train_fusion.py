import torch
import argparse
import datetime
import numpy as np
import random
import wandb

from tqdm import tqdm
import math

from utils.setup import *
from utils.loading import *
from utils.loss import *

from modules.pipeline import Pipeline


def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config")

    args = parser.parse_args()
    return vars(args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_fusion(args):

    config = load_config_from_yaml(args["config"])

    config.TIMESTAMP = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    # initialize weights and biases logging
    wandb.init(
        config=config,
        entity="esandstroem",
        project="senfunet-fusion",
        name=config.TIMESTAMP,
        notes="put comment here",
    )
    # change run name of wandb
    wandb.run.name = config.TIMESTAMP
    wandb.run.save()

    # set seed for reproducibility
    if config.SETTINGS.seed:
        random.seed(config.SETTINGS.seed)
        np.random.seed(config.SETTINGS.seed)
        torch.manual_seed(config.SETTINGS.seed)
        torch.cuda.manual_seed_all(config.SETTINGS.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # get workspace
    workspace = get_workspace(config)

    # save config before training
    workspace.save_config(config)

    if config.SETTINGS.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    config.FUSION_MODEL.device = device

    # get datasets
    # get train dataset
    train_data_config = get_data_config(config, mode="train")
    train_dataset = get_data(config.DATA.dataset, train_data_config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, config.TRAINING.train_batch_size, config.TRAINING.train_shuffle
    )

    # get val dataset
    val_data_config = get_data_config(config, mode="val")
    val_dataset = get_data(config.DATA.dataset, val_data_config)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, config.TRAINING.val_batch_size, config.TRAINING.val_shuffle
    )

    # specify number of features
    if config.FEATURE_MODEL.use_feature_net:
        config.FEATURE_MODEL.n_features = (
            config.FEATURE_MODEL.n_features + config.FEATURE_MODEL.append_depth
        )
    else:
        config.FEATURE_MODEL.n_features = (
            config.FEATURE_MODEL.append_depth + 3 * config.FEATURE_MODEL.w_rgb
        )

    # get database
    # get train database
    train_database = get_database(train_dataset, config, mode="train")
    val_database = get_database(val_dataset, config, mode="val")

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device)  # put the networks on the gpu

    for sensor in config.DATA.input:
        if config.FUSION_MODEL.use_fusion_net:
            print(
                "Fusion Net ",
                sensor,
                ": ",
                count_parameters(pipeline.fuse_pipeline._fusion_network[sensor]),
            )
        if config.FEATURE_MODEL.use_feature_net:
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
    print("Fusion and Filtering: ", count_parameters(pipeline))

    # optimization
    criterion = Fusion_TranslationLoss(config)

    # load pretrained routing model into parameters
    if config.ROUTING.do:
        if config.FILTERING_MODEL.model == "tsdf_early_fusion":
            routing_checkpoint = torch.load(config.TESTING.routing_model_path)

            pipeline.fuse_pipeline._routing_network.load_state_dict(
                routing_checkpoint["pipeline_state_dict"]
            )
        else:
            for sensor_ in config.DATA.input:
                checkpoint = torch.load(
                    eval("config.TRAINING.routing_" + sensor_ + "_model_path")
                )
                pipeline.fuse_pipeline._routing_network[sensor_].load_state_dict(
                    checkpoint["pipeline_state_dict"]
                )

    if config.TRAINING.pretrain_filtering_net:
        load_pipeline(config.TESTING.fusion_model_path, pipeline)

    if config.TRAINING.pretrain_fusion_net and config.FUSION_MODEL.use_fusion_net:
        raise NotImplementedError

    if config.FEATURE_MODEL.use_feature_net and config.FILTERING_MODEL.do:
        feature_params = []
        for sensor in config.DATA.input:
            feature_params += list(
                pipeline.fuse_pipeline._feature_network[sensor].parameters()
            )

        optimizer_feature = torch.optim.Adam(
            feature_params,
            config.OPTIMIZATION.lr_fusion,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )

        scheduler_feature = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer_feature,
            step_size=config.OPTIMIZATION.scheduler.step_size_fusion,
            gamma=config.OPTIMIZATION.scheduler.gamma_fusion,
        )

    if (
        not config.FILTERING_MODEL.CONV3D_MODEL.fixed
        and pipeline.filter_pipeline is not None
    ):

        optimizer_filt = torch.optim.Adam(
            pipeline.filter_pipeline._filtering_network.parameters(),
            config.OPTIMIZATION.lr_filtering,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )

        scheduler_filt = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer_filt,
            step_size=config.OPTIMIZATION.scheduler.step_size_filtering,
            gamma=config.OPTIMIZATION.scheduler.gamma_filtering,
        )

    if not config.FUSION_MODEL.fixed and config.FUSION_MODEL.use_fusion_net:
        fusion_params = []
        for sensor in config.DATA.input:
            fusion_params += list(
                pipeline.fuse_pipeline._fusion_network[sensor].parameters()
            )
        optimizer_fusion = torch.optim.RMSprop(
            fusion_params,
            config.OPTIMIZATION.lr_fusion,
            config.OPTIMIZATION.rho,
            config.OPTIMIZATION.eps,
            momentum=config.OPTIMIZATION.momentum,
            weight_decay=config.OPTIMIZATION.weight_decay,
        )

        scheduler_fusion = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer_fusion,
            step_size=config.OPTIMIZATION.scheduler.step_size_fusion,
            gamma=config.OPTIMIZATION.scheduler.gamma_fusion,
        )

    # add weight and gradient tracking in wandb
    wandb.watch(pipeline, criterion, log="all", log_freq=500)

    # define some parameters
    n_batches = float(len(train_dataset) / config.TRAINING.train_batch_size)

    # evaluation metrics
    best_iou_filt = 0.0  # best filtered
    is_best_filt = False

    best_iou = dict()
    is_best = dict()
    for sensor in config.DATA.input:
        best_iou[sensor] = 0.0
        is_best[sensor] = False

    # copy sensor list so that we can shuffle the sensors but still have the same
    # sensor at index 0 and index 1 as originally in the config file input
    sensors = config.DATA.input.copy()

    for epoch in range(0, config.TRAINING.n_epochs):

        workspace.log(
            "Training epoch {}/{}".format(epoch, config.TRAINING.n_epochs), mode="train"
        )

        pipeline.train()

        if config.ROUTING.do:
            pipeline.fuse_pipeline._routing_network.eval()
        if config.FUSION_MODEL.fixed and config.FUSION_MODEL.use_fusion_net:
            pipeline.fuse_pipeline._fusion_network.eval()
        if (
            config.FILTERING_MODEL.CONV3D_MODEL.fixed
            and pipeline.filter_pipeline is not None
        ):
            pipeline.filter_pipeline._filtering_network.eval()

        # resetting databases before each epoch starts
        train_database.reset()
        val_database.reset()

        divide = 0
        train_loss = 0
        grad_norm_alpha_net = 0
        grad_norm_feature = dict()
        val_norm = 0
        l1_interm = 0
        l1_grid = 0
        for sensor_ in config.DATA.input:
            grad_norm_feature[sensor_] = 0

        for i, batch in tqdm(enumerate(train_loader), total=len(train_dataset)):

            if config.TRAINING.reset_strategy:
                if np.random.random_sample() <= config.TRAINING.reset_prob:
                    workspace.log(
                        "Resetting randomly trajectory {} at step {}".format(
                            batch["frame_id"][0][:-2], i
                        ),
                        mode="train",
                    )
                    workspace.log(
                        "Resetting grid for scene {} at step {}".format(
                            batch["frame_id"][0].split("/")[0], i
                        ),
                        mode="train",
                    )
                    train_database.reset(batch["frame_id"][0].split("/")[0])

            if config.DATA.collaborative_reconstruction:
                if (
                    math.ceil(
                        int(batch["frame_id"][0].split("/")[-1])
                        / config.DATA.frames_per_chunk
                    )
                    % 2
                    == 0
                ):
                    sensor = config.DATA.input[0]
                else:
                    sensor = config.DATA.input[1]

                batch["depth"] = batch[sensor + "_depth"]
                batch["mask"] = batch[sensor + "_mask"]
                if config.FILTERING_MODEL.model == "routedfusion":
                    batch["sensor"] = config.DATA.input[0]
                else:
                    batch["sensor"] = sensor

                batch["routingNet"] = sensor  # used to be able to train routedfusion
                batch["fusionNet"] = sensor  # used to be able to train routedfusion
                output = pipeline(batch, train_database, epoch, device)

                # optimization
                if (
                    output is None
                ):  # output is None when no valid indices were found for the filtering net within the random bbox within the bounding volume of the integrated indices
                    print("output None from pipeline")
                    continue

                if output == "save_and_exit":
                    print("Found alpha nan. Save and exit")
                    workspace.save_model_state(
                        {"pipeline_state_dict": pipeline.state_dict(), "epoch": epoch},
                        is_best=is_best,
                        is_best_filt=is_best_filt,
                    )
                    return

                output = criterion(output)

                if output["loss"] is not None:
                    divide += 1
                    train_loss += output[
                        "loss"
                    ].item()  # note that this loss is a moving average over the training window of log_freq steps
                if output["l1_interm"] is not None:
                    l1_interm += output[
                        "l1_interm"
                    ].item()  # note that this loss is a moving average over the training window of log_freq steps
                if output["l1_grid"] is not None:
                    l1_grid += output["l1_grid"].item()

                if output["loss"] is not None:
                    output["loss"].backward()
                # break
            else:
                # fusion pipeline
                # randomly integrate the selected sensors
                random.shuffle(sensors)

                for sensor in sensors:
                    batch["depth"] = batch[sensor + "_depth"]
                    batch["mask"] = batch[sensor + "_mask"]

                    if config.FILTERING_MODEL.model == "routedfusion":
                        batch["sensor"] = config.DATA.input[0]
                    else:
                        batch["sensor"] = sensor
                    batch[
                        "routingNet"
                    ] = sensor  # used to be able to train routedfusion
                    batch["fusionNet"] = sensor  # used to be able to train routedfusion
                    output = pipeline(batch, train_database, epoch, device)

                    # optimization
                    if (
                        output is None
                    ):  # output is None when no valid indices were found for the filtering net within the random bbox within the bounding volume of the integrated indices
                        print("output None from pipeline")
                        # break
                        continue

                    if output == "save_and_exit":
                        print("Found alpha nan. Save and exit")
                        workspace.save_model_state(
                            {
                                "pipeline_state_dict": pipeline.state_dict(),
                                "epoch": epoch,
                            },
                            is_best=is_best,
                            is_best_filt=is_best_filt,
                        )
                        return

                    output = criterion(output)

                    if output["loss"] is not None:
                        divide += 1
                        train_loss += output[
                            "loss"
                        ].item()  # note that this loss is a moving average over the training window of log_freq steps
                    if output["l1_interm"] is not None:
                        l1_interm += output[
                            "l1_interm"
                        ].item()  # note that this loss is a moving average over the training window of log_freq steps
                    if output["l1_grid"] is not None:
                        l1_grid += output["l1_grid"].item()

                    if output["loss"] is not None:
                        output["loss"].backward()
                    # break

            del batch

            for name, param in pipeline.named_parameters():
                if param.grad is not None:
                    # accumulate gradient norms from feature and weighting net
                    if (
                        (i + 1) % config.OPTIMIZATION.accumulation_steps == 0
                        or i == n_batches - 1
                    ):
                        if name.startswith("fuse_pipeline._feature"):
                            grad_norm_feature[name.split(".")[2]] += torch.norm(
                                param.grad
                            )
                        else:
                            grad_norm_alpha_net += torch.norm(param.grad)
                    val_norm += torch.norm(param)

            if (i + 1) % config.SETTINGS.log_freq == 0 and divide > 0:

                train_loss /= divide
                grad_norm_alpha_net /= divide

                val_norm /= divide

                l1_interm /= divide
                l1_grid /= divide
                for sensor_ in config.DATA.input:
                    grad_norm_feature[sensor_] /= divide

                wandb.log({"Train/nbr frames": i + 1 + epoch * n_batches})
                wandb.log({"Train/total loss": train_loss})
                wandb.log({"Train/gradient norm alpha net": grad_norm_alpha_net})
                wandb.log({"Train/parameter value norm": val_norm})
                if config.FUSION_MODEL.use_fusion_net:
                    wandb.log(
                        {"Train/fusion network l1 loss": l1_interm}
                    )  # concatenated norm from both sensors
                wandb.log({"Train/l1 sensor fused loss": l1_grid})

                for sensor_ in config.DATA.input:
                    if (
                        config.FEATURE_MODEL.use_feature_net
                        and config.FILTERING_MODEL.do
                    ):
                        wandb.log(
                            {
                                "Train/gradident norm feature net "
                                + sensor_: grad_norm_feature[sensor_]
                            }
                        )

                divide = 0
                train_loss = 0
                grad_norm_alpha_net = 0
                grad_norm_feature = dict()
                val_norm = 0
                l1_interm = 0
                l1_grid = 0
                for sensor_ in config.DATA.input:
                    grad_norm_feature[sensor_] = 0

            if config.TRAINING.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    pipeline.parameters(), max_norm=1.0, norm_type=2
                )

            if (
                i + 1
            ) % config.OPTIMIZATION.accumulation_steps == 0 or i == n_batches - 1:
                if config.FEATURE_MODEL.use_feature_net and config.FILTERING_MODEL.do:
                    optimizer_feature.step()
                    scheduler_feature.step()
                    optimizer_feature.zero_grad(set_to_none=True)

                if (
                    not config.FILTERING_MODEL.CONV3D_MODEL.fixed
                    and pipeline.filter_pipeline is not None
                ):
                    # make the gradients belonging to layers with zero-norm gradient none instead of zero to avoid update
                    # of weights
                    optimizer_filt.step()
                    scheduler_filt.step()
                    optimizer_filt.zero_grad(set_to_none=True)
                if not config.FUSION_MODEL.fixed and config.FUSION_MODEL.use_fusion_net:
                    optimizer_fusion.step()
                    scheduler_fusion.step()
                    optimizer_fusion.zero_grad(set_to_none=True)

            # for debugging
            if (
                (i + 1) % config.SETTINGS.eval_freq == 0
                or i == n_batches - 1
                or (i == 2 and epoch == 0)
            ):
                val_database.reset()
                # zero out all grads
                if config.FEATURE_MODEL.use_feature_net and config.FILTERING_MODEL.do:
                    optimizer_feature.zero_grad(set_to_none=True)
                if (
                    not config.FILTERING_MODEL.CONV3D_MODEL.fixed
                    and pipeline.filter_pipeline is not None
                ):
                    optimizer_filt.zero_grad(set_to_none=True)
                if not config.FUSION_MODEL.fixed and config.FUSION_MODEL.use_fusion_net:
                    optimizer_fusion.zero_grad(set_to_none=True)

                pipeline.eval()

                pipeline.test(
                    val_loader,
                    val_dataset,
                    val_database,
                    config.DATA.input,
                    device,
                )

                val_eval, val_eval_fused = val_database.evaluate(
                    mode="val", workspace=workspace
                )

                for sensor_ in config.DATA.input:
                    wandb.log({"Val/mse " + sensor_: val_eval[sensor_]["mse"]})
                    wandb.log({"Val/acc " + sensor_: val_eval[sensor_]["acc"]})
                    wandb.log({"Val/iou " + sensor_: val_eval[sensor_]["iou"]})
                    wandb.log({"Val/mad " + sensor_: val_eval[sensor_]["mad"]})

                wandb.log({"Val/mse fused": val_eval_fused["mse"]})
                wandb.log({"Val/acc fused": val_eval_fused["acc"]})
                wandb.log({"Val/iou fused": val_eval_fused["iou"]})
                wandb.log({"Val/mad fused": val_eval_fused["mad"]})

                # check if current checkpoint is best
                if val_eval_fused["iou"] >= best_iou_filt:
                    is_best_filt = True
                    best_iou_filt = val_eval_fused["iou"]
                    workspace.log(
                        "found new best model overall with iou {} at epoch {}".format(
                            best_iou_filt, epoch
                        ),
                        mode="val",
                    )

                else:
                    is_best_filt = False

                for sensor in config.DATA.input:
                    if val_eval[sensor]["iou"] >= best_iou[sensor]:
                        is_best[sensor] = True
                        best_iou[sensor] = val_eval[sensor]["iou"]
                        workspace.log(
                            "found new best "
                            + sensor
                            + " model with iou {} at epoch {}".format(
                                best_iou[sensor], epoch
                            ),
                            mode="val",
                        )

                    else:
                        is_best[sensor] = False

                # save alpha histogram
                workspace.save_alpha_histogram(val_database, config.DATA.input, epoch)

                # save checkpoint
                workspace.save_model_state(
                    {"pipeline_state_dict": pipeline.state_dict(), "epoch": epoch},
                    is_best=is_best,
                    is_best_filt=is_best_filt,
                )

                pipeline.train()
                if config.ROUTING.do:
                    pipeline.fuse_pipeline._routing_network.eval()
                if config.FUSION_MODEL.fixed and config.FUSION_MODEL.use_fusion_net:
                    pipeline.fuse_pipeline._fusion_network.eval()
                if (
                    config.FILTERING_MODEL.CONV3D_MODEL.fixed
                    and pipeline.filter_pipeline is not None
                ):
                    pipeline.filter_pipeline._filtering_network.eval()


if __name__ == "__main__":

    args = arg_parser()
    train_fusion(args)
