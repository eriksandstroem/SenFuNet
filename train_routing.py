import torch
import argparse
import datetime
import random

import numpy as np

from tqdm import tqdm

from utils.loading import load_config_from_yaml
from utils.setup import *

from utils.loss import RoutingLoss
from modules.routing import ConfidenceRouting


def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    return vars(args)


def prepare_input_data(batch, config, device):

    for k, sensor_ in enumerate(config.DATA.input):
        if k == 0:
            inputs = batch[sensor_ + "_depth"].unsqueeze_(1)
        else:
            inputs = torch.cat((batch[sensor_ + "_depth"].unsqueeze_(1), inputs), 1)
    inputs = inputs.to(device)

    if config.ROUTING.intensity_grad:
        intensity = batch["intensity"].unsqueeze_(1)
        grad = batch["gradient"].unsqueeze_(1)
        inputs = torch.cat((intensity, grad, inputs), 1)
    inputs = inputs.to(device)

    target = batch[config.DATA.target]  # 2, 512, 512 (batch size, height, width)
    target = target.to(device)
    target = target.unsqueeze_(
        1
    )  # 2, 1, 512, 512 (batch size, channels, height, width)
    return inputs, target


def train(args, config):

    # set seed for reproducibility
    if config.SETTINGS.seed:
        random.seed(config.SETTINGS.seed)
        np.random.seed(config.SETTINGS.seed)
        torch.manual_seed(config.SETTINGS.seed)
        torch.cuda.manual_seed_all(config.SETTINGS.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.SETTINGS.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    config.TIMESTAMP = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    print("model time stamp: ", config.TIMESTAMP)

    workspace = get_workspace(config)
    workspace.save_config(config)

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

    # define model
    Cin = len(config.DATA.input)

    if config.ROUTING.intensity_grad:
        Cin += 2

    model = ConfidenceRouting(
        Cin=Cin, F=config.MODEL.contraction, batchnorms=config.MODEL.normalization
    )
    model = model.to(device)

    # define loss function
    criterion = RoutingLoss(config)
    criterion = criterion.to(device)

    # define optimizer
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        config.OPTIMIZATION.lr,
        config.OPTIMIZATION.rho,
        config.OPTIMIZATION.eps,
        momentum=config.OPTIMIZATION.momentum,
        weight_decay=config.OPTIMIZATION.weight_decay,
    )

    n_train_batches = int(len(train_dataset) / config.TRAINING.train_batch_size)
    n_val_batches = int(len(val_dataset) / config.TRAINING.val_batch_size)

    val_loss_best = np.infty

    # sample validation visualization frames
    val_vis_ids = np.random.choice(np.arange(0, n_val_batches), 10, replace=False)

    # # define metrics
    l1_criterion = torch.nn.L1Loss()
    l2_criterion = torch.nn.MSELoss()

    for epoch in range(0, config.TRAINING.n_epochs):
        print("epoch: ", epoch)

        val_loss_t = 0.0
        val_loss_l1 = 0.0
        val_loss_l2 = 0.0

        train_loss_t = 0.0
        train_loss_l1 = 0.0
        train_loss_l2 = 0.0

        # make ready for training and clear optimizer
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, total=n_train_batches)):
            inputs, target = prepare_input_data(batch, config, device)

            output = model.forward(inputs)

            est = output[:, 0, :, :].unsqueeze_(1)
            unc = output[:, 1, :, :].unsqueeze_(1)

            if not config.LOSS.completion:
                if len(config.DATA.input) == 1:
                    mask = (
                        batch[config.DATA.input[0] + "_mask"].to(device).unsqueeze_(1)
                    )
                else:
                    mask = batch["mask"].to(device).unsqueeze_(1)
                target = torch.where(mask == 0.0, torch.zeros_like(target), target)

            # compute training loss
            loss = criterion.forward(est, unc, target)
            loss.backward()

            # compute metrics for analysis
            loss_l1 = l1_criterion.forward(est, target)
            loss_l2 = l2_criterion.forward(est, target)

            train_loss_t += loss.item()
            train_loss_l1 += loss_l1.item()
            train_loss_l2 += loss_l2.item()

            if i % config.OPTIMIZATION.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # break

        train_loss_t /= n_train_batches
        train_loss_l1 /= n_train_batches
        train_loss_l2 /= n_train_batches

        # log training metrics
        workspace.log("Epoch {} Loss {}".format(epoch, train_loss_t))
        workspace.log("Epoch {} L1 Loss {}".format(epoch, train_loss_l1))
        workspace.log("Epoch {} L2 Loss {}".format(epoch, train_loss_l2))

        workspace.writer.add_scalar("Train/loss_t", train_loss_t, global_step=epoch)
        workspace.writer.add_scalar("Train/loss_l1", train_loss_l1, global_step=epoch)
        workspace.writer.add_scalar("Train/loss_l2", train_loss_l2, global_step=epoch)

        model.eval()

        for i, batch in enumerate(tqdm(val_loader, total=n_val_batches)):
            inputs, target = prepare_input_data(batch, config, device)

            output = model.forward(inputs)

            est = output[:, 0, :, :].unsqueeze_(1)
            unc = output[:, 1, :, :].unsqueeze_(1)

            # visualize frames
            if i in val_vis_ids:
                # parse frames
                frame_est = est[0, :, :, :].cpu().detach().numpy()
                # frame_inp = inputs[0, :, :, :].cpu().detach().numpy()
                frame_gt = target[0, :, :, :].cpu().detach().numpy()
                frame_unc = output[0, :, :, :].cpu().detach().numpy()
                frame_conf = np.exp(-1.0 * frame_unc)
                frame_l1 = np.abs(frame_est - frame_gt)
                # frame_inp_l1 = np.abs(frame_inp - frame_gt)

                # write to logger
                workspace.writer.add_image(
                    "Val/est_{}".format(i), frame_est, global_step=epoch
                )
                workspace.writer.add_image(
                    "Val/gt_{}".format(i), frame_gt, global_step=epoch
                )
                workspace.writer.add_image(
                    "Val/unc_{}".format(i), frame_unc, global_step=epoch
                )
                workspace.writer.add_image(
                    "Val/conf_{}".format(i), frame_conf, global_step=epoch
                )
                workspace.writer.add_image(
                    "Val/l1_{}".format(i), frame_l1, global_step=epoch
                )
                # workspace.writer.add_image('Val/l1_inp_{}'.format(i), frame_inp_l1, global_step=epoch)

            if not config.LOSS.completion:
                if len(config.DATA.input) == 1:
                    mask = (
                        batch[config.DATA.input[0] + "_mask"].to(device).unsqueeze_(1)
                    )
                else:
                    mask = batch["mask"].to(device).unsqueeze_(1)
                target = torch.where(mask == 0.0, torch.zeros_like(target), target)

            loss_t = criterion.forward(est, unc, target)
            loss_l1 = l1_criterion.forward(est, target)
            loss_l2 = l2_criterion.forward(est, target)

            val_loss_t += loss_t.item()
            val_loss_l1 += loss_l1.item()
            val_loss_l2 += loss_l2.item()

        val_loss_t /= n_val_batches
        val_loss_l1 /= n_val_batches
        val_loss_l2 /= n_val_batches

        # log validation metrics
        workspace.log(
            "Epoch {} Validation Loss {}".format(epoch, val_loss_t), mode="val"
        )
        workspace.log(
            "Epoch {} Validation L1 Loss {}".format(epoch, val_loss_l1), mode="val"
        )
        workspace.log(
            "Epoch {} Validation L2 Loss {}".format(epoch, val_loss_l2), mode="val"
        )

        workspace.writer.add_scalar("Val/loss_t", val_loss_t, global_step=epoch)
        workspace.writer.add_scalar("Val/loss_l1", val_loss_l1, global_step=epoch)
        workspace.writer.add_scalar("Val/loss_l2", val_loss_l2, global_step=epoch)

        # define model state for storing
        model_state = {
            "epoch": epoch,
            "pipeline_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if val_loss_t <= val_loss_best:
            val_loss_best = val_loss_t
            workspace.log(
                "Found new best model with loss {} at epoch {}".format(
                    val_loss_best, epoch
                ),
                mode="val",
            )
            workspace.save_model_state(model_state, is_best=True)
        else:
            workspace.save_model_state(model_state, is_best=False)


if __name__ == "__main__":

    # get arguments
    args = arg_parser()

    # get configs
    config = load_config_from_yaml(args["config"])

    # train
    train(args, config)
