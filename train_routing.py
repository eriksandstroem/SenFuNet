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
import wandb


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
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(config.SETTINGS.seed)
        torch.backends.cudnn.benchmark = False

    if config.SETTINGS.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    config.TIMESTAMP = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    print("model time stamp: ", config.TIMESTAMP)

    # initialize weights and biases logging
    wandb.init(
        config=config,
        entity="esandstroem",
        project="senfunet-routing",
        name=config.TIMESTAMP,
        notes="put comment here",
    )
    # change run name of wandb
    wandb.run.name = config.TIMESTAMP
    wandb.run.save()

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

    # add weight and gradient tracking in wandb
    wandb.watch(model, criterion, log="all", log_freq=1)

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
    val_vis_ids = np.random.choice(np.arange(0, n_val_batches), 5, replace=False)

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

        train_epoch_loss_t = 0.0
        train_epoch_loss_l1 = 0.0
        train_epoch_loss_l2 = 0.0

        # make ready for training and clear optimizer
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, total=n_train_batches)):
            inputs, target = prepare_input_data(batch, config, device)

            output = model(inputs)

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

            train_epoch_loss_t += loss.item()
            train_epoch_loss_l1 += loss_l1.item()
            train_epoch_loss_l2 += loss_l2.item()

            if i % config.OPTIMIZATION.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % config.SETTINGS.log_freq == 0 and i > 0:
                # compute avg. loss per frame
                train_loss_t /= (
                    config.SETTINGS.log_freq * config.TRAINING.train_batch_size
                )
                train_loss_l1 /= (
                    config.SETTINGS.log_freq * config.TRAINING.train_batch_size
                )
                train_loss_l2 /= (
                    config.SETTINGS.log_freq * config.TRAINING.train_batch_size
                )
                # logging not working properly since I log at i = 0 and then
                # the train_loss parameters are divided with a large number causing weird oscillations. Also add wandb to gitignore. Also remove the lines which do the workspace.writer calls to tensorboard, but I still want to log per epoch to file to compare the training loss to the validation loss over the epochs. So I need a new parameter for this.
                wandb.log(
                    {
                        "Train/total loss": train_loss_t,
                        "Train/l1 loss": train_loss_l1,
                        "Train/l2 loss": train_loss_l2,
                        "Train/nbr_frames": (epoch * n_train_batches + i)
                        * config.TRAINING.train_batch_size,
                    }
                )
                train_loss_t = 0
                train_loss_l1 = 0
                train_loss_l2 = 0

        train_epoch_loss_t /= n_train_batches * config.TRAINING.train_batch_size
        train_epoch_loss_l1 /= n_train_batches * config.TRAINING.train_batch_size
        train_epoch_loss_l2 /= n_train_batches * config.TRAINING.train_batch_size

        # log training metrics
        workspace.log("Epoch {} Loss {}".format(epoch, train_epoch_loss_t))
        workspace.log("Epoch {} L1 Loss {}".format(epoch, train_epoch_loss_l1))
        workspace.log("Epoch {} L2 Loss {}".format(epoch, train_epoch_loss_l2))

        model.eval()

        for i, batch in enumerate(tqdm(val_loader, total=n_val_batches)):
            inputs, target = prepare_input_data(batch, config, device)

            output = model(inputs)

            est = output[:, 0, :, :].unsqueeze_(1)
            unc = output[:, 1, :, :].unsqueeze_(1)
            # visualize frames
            if i in val_vis_ids:
                # parse frames and normalize to range 0-1
                frame_est = est[0, :, :, :].cpu().detach().numpy().reshape(512, 512, 1)
                frame_est /= np.amax(frame_est)
                frame_gt = (
                    target[0, :, :, :].cpu().detach().numpy().reshape(512, 512, 1)
                )
                frame_gt /= np.amax(frame_gt)
                frame_unc = unc[0, :, :, :].cpu().detach().numpy().reshape(512, 512, 1)
                frame_conf = np.exp(-1.0 * frame_unc)
                frame_unc /= np.amax(frame_unc)
                frame_l1 = np.abs(frame_est - frame_gt).reshape(512, 512, 1)
                frame_l1 /= np.amax(frame_l1)

                wandb.log(
                    {
                        "Val/images": [
                            wandb.Image(
                                frame_est,
                                caption="depth estimate {}".format(i),
                            ),
                            wandb.Image(frame_gt, caption="gt depth {}".format(i)),
                            wandb.Image(
                                frame_unc,
                                caption="uncertainty estimate {}".format(i),
                            ),
                            wandb.Image(
                                frame_conf,
                                caption="confidence estimate {}".format(i),
                            ),
                            wandb.Image(
                                frame_l1,
                                caption="l1 depth error {}".format(i),
                            ),
                        ]
                    }
                )

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

        val_loss_t /= n_val_batches * config.TRAINING.train_batch_size
        val_loss_l1 /= n_val_batches * config.TRAINING.train_batch_size
        val_loss_l2 /= n_val_batches * config.TRAINING.train_batch_size

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

        wandb.log(
            {
                "Val/total loss": val_loss_t,
                "Val/l1 loss": val_loss_l1,
                "Val/l2 loss": val_loss_l2,
                "Val/epoch": epoch,
            }
        )

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
