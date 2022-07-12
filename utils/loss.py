import torch

import numpy as np


class Fusion_TranslationLoss(torch.nn.Module):
    def __init__(self, config, reduction="none"):
        super(Fusion_TranslationLoss, self).__init__()

        self.sensors = config.DATA.input
        self.grid_weight = config.LOSS.grid_weight
        self.fixed_fusion_net = config.FUSION_MODEL.fixed
        self.alpha_weight = config.LOSS.alpha_weight
        self.alpha_supervision = config.LOSS.alpha_supervision
        self.alpha_single_sensor_supervision = (
            config.LOSS.alpha_single_sensor_supervision
        )
        self.fusion_weight = config.LOSS.fusion_weight
        self.add_outlier_channel = config.FILTERING_MODEL.CONV3D_MODEL.outlier_channel
        self.use_fusion_net = config.FUSION_MODEL.use_fusion_net

        self.l1 = torch.nn.L1Loss(reduction=reduction)
        self.l2 = torch.nn.MSELoss(reduction=reduction)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, output):

        l = None

        if self.refinement_loss:
            est_grid_dict = dict()
            init = dict()
            for sensor_ in self.sensors:
                est_grid_dict[sensor_] = output["filtered_output"][
                    "tsdf_filtered_grid"
                ]["tsdf_" + sensor_]
                init[sensor_] = output["filtered_output"]["tsdf_filtered_grid"][
                    sensor_ + "_init"
                ]

        if "filtered_output" in output:
            target_grid = output["filtered_output"]["tsdf_target_grid"]
            est_grid = output["filtered_output"]["tsdf_filtered_grid"]["tsdf"]
            l1_grid = self.l1.forward(est_grid, target_grid)

            # remove the indices where only one sensor integrates
            mask = torch.ones_like(l1_grid)
            for sensor_ in self.sensors:
                mask = torch.logical_and(
                    mask,
                    output["filtered_output"]["tsdf_filtered_grid"][sensor_ + "_init"],
                )

            normalization = torch.ones_like(l1_grid[mask]).sum()
            l1_grid = l1_grid[mask].sum() / normalization

            if (
                mask.sum() != 0
            ):  # checks if we have an empty l1_grid. This happens for the first frame that is integrated into the scene since
                # we have no intersection to the other sensors since they are not yet integrated
                l = self.grid_weight * l1_grid
                if l.isnan() > 0 or l.isinf():
                    print("1est_grid: ", est_grid.isnan().sum())
                    print("1target_grid: ", target_grid.isnan().sum())
                    print("1est_grid inf: ", est_grid.isinf().sum())
                    print("1target_grid inf: ", target_grid.isinf().sum())
                    print("1loss nan:", l.isnan())
                    print("1loss inf: ", l.isinf())
                    print("normalization factor: ", normalization)
                    print("l1_grid.shape:", l1_grid.shape)
            else:
                l1_grid = None
        else:
            l1_grid = None

        if self.alpha_supervision:
            pos_proxy_alpha = output["filtered_output"]["proxy_alpha_grid"] >= 0
            mask = torch.logical_and(mask, pos_proxy_alpha)

            l1_alpha = self.l1.forward(
                output["filtered_output"]["alpha_grid"][mask],
                output["filtered_output"]["proxy_alpha_grid"][mask],
            )
            normalization = torch.ones_like(l1_alpha).sum()
            l1_alpha = l1_alpha.sum() / normalization

            if l is None:
                l = self.alpha_weight * l1_alpha
            else:
                l += self.alpha_weight * l1_alpha

        if self.alpha_single_sensor_supervision and "filtered_output" in output:
            # this loss is intended to improve outlier filtering performance which is
            # specifically needed where only one sensor integrates, since we do not supervise these voxels at all otherwise. The idea is to extract all voxels where only one sensor integrates and check the tsdf error compared to the ground truth. If the error is larger than e.g. 0.04 m, we say that it is an outlier. This means setting the target alpha as "no confidence" to the sensor, otherwise 100 percent confidence.
            for k, sensor_ in enumerate(self.sensors):
                # get the mask where only sensor_ is active
                # mask is a variable containing the "and"-region of both sensors. We negate it to get rid of
                # all voxels where both sensors integrate
                single_sensor_mask = torch.logical_and(
                    ~mask,
                    output["filtered_output"]["tsdf_filtered_grid"][
                        sensor_ + "_init"
                    ].squeeze(),
                )
                # compute target alpha for the sensor in question
                if self.add_outlier_channel:
                    target_alpha = torch.zeros_like(
                        output["filtered_output"]["alpha_grid"][1, :, :, :][
                            single_sensor_mask
                        ]
                    )
                else:
                    target_alpha = torch.zeros_like(
                        output["filtered_output"]["alpha_grid"][single_sensor_mask]
                    )
                tsdf_err = self.l1.forward(
                    est_grid[single_sensor_mask], target_grid[single_sensor_mask]
                )
                sgn_err = torch.abs(
                    torch.sgn(est_grid[single_sensor_mask])
                    - torch.sgn(target_grid[single_sensor_mask])
                )

                sgn_err = (sgn_err > 0).float()
                # gt 0, stereo 1. When alpha is 1 we trust gt

                if k == 0:
                    target_alpha = ~torch.logical_and(tsdf_err > 0.04, sgn_err > 0)
                    outlier_alpha = target_alpha is False
                    inlier_alpha = target_alpha is True
                    target_alpha = target_alpha.float()
                elif k == 1:
                    outlier_alpha = target_alpha is True
                    inlier_alpha = target_alpha is False
                    target_alpha = target_alpha.float()

                # only outlier if tsdf err is larger than 0.04 and sgn err, toherwise inlier

                if single_sensor_mask.sum() > 0:
                    if self.add_outlier_channel:
                        l1_alpha = self.l1.forward(
                            output["filtered_output"]["alpha_grid"][1, :, :, :][
                                single_sensor_mask
                            ],
                            target_alpha,
                        )
                    else:
                        l1_alpha = self.l1.forward(
                            output["filtered_output"]["alpha_grid"][single_sensor_mask],
                            target_alpha,
                        )

                    l1_outlier_alpha = (
                        10
                        * l1_alpha[outlier_alpha].sum()
                        / torch.ones_like(l1_alpha[outlier_alpha]).sum()
                    )
                    l1_inlier_alpha = (
                        l1_alpha[inlier_alpha].sum()
                        / torch.ones_like(l1_alpha[inlier_alpha]).sum()
                    )

                    if l is None:
                        if l1_outlier_alpha.sum() > 0:
                            l = self.alpha_weight * l1_outlier_alpha
                            if l1_inlier_alpha.sum() > 0:
                                l += self.alpha_weight * l1_inlier_alpha
                        elif l1_inlier_alpha.sum() > 0:
                            l = self.alpha_weight * l1_inlier_alpha

                    else:
                        if l1_outlier_alpha.sum() > 0:
                            l += self.alpha_weight * l1_outlier_alpha
                        if l1_inlier_alpha.sum() > 0:
                            l += self.alpha_weight * l1_inlier_alpha

        if not self.fixed_fusion_net and self.use_fusion_net:
            est_interm = output["tsdf_fused"]
            target_interm = output["tsdf_target"]
            l1_interm = self.l1.forward(est_interm, target_interm)

            normalization = torch.ones_like(l1_interm).sum()

            l1_interm = l1_interm.sum() / normalization
            if l is None:
                l = self.fusion_weight * l1_interm
            else:
                l += self.fusion_weight * l1_interm
        else:
            l1_interm = None

        output = dict()
        output["loss"] = l  # total loss
        output["l1_interm"] = l1_interm  # this mixes all sensors in one logging graph
        output["l1_grid"] = l1_grid  # sensor fused l1 loss

        return output


class RoutingLoss(torch.nn.Module):
    def __init__(self, config):

        super(RoutingLoss, self).__init__()

        self.criterion1 = GradientWeightedDepthLoss(
            crop_fraction=config.LOSS.crop_fraction,
            vmin=config.LOSS.vmin,
            vmax=config.LOSS.vmax,
            weight_scale=config.LOSS.weight_scale,
        )

        self.criterion2 = UncertaintyDepthLoss(
            crop_fraction=config.LOSS.crop_fraction,
            vmin=config.LOSS.vmin,
            vmax=config.LOSS.vmax,
            lmbda=config.LOSS.lmbda,
        )

        self.criterion4 = GradientWeightedUncertaintyDepthLoss(
            crop_fraction=config.LOSS.crop_fraction,
            vmin=config.LOSS.vmin,
            vmax=config.LOSS.vmax,
            weight_scale=config.LOSS.weight_scale,
            lmbda=config.LOSS.lmbda,
        )

        self.combined = config.LOSS.name

    def forward(self, prediction, uncertainty, target):
        if self.combined == "gradweighted + uncertainty":
            l1 = self.criterion1.forward(prediction, target)
            l2 = self.criterion2.forward(prediction, uncertainty, target)
            return l1 + l2
        elif self.combined == "uncertainty":
            l = self.criterion2.forward(prediction, uncertainty, target)
            return l
        elif self.combined == "gradweighteduncertainty":
            l = self.criterion4.forward(prediction, uncertainty, target)
            return l


class GradientWeightedDepthLoss(torch.nn.Module):
    """
    This loss is a simple L1 loss on the depth pixels and the gradients, but adds a weight to the L1 loss on the depth which is
    proportional to the gradient loss at that pixel. In that way, pixels which have the wrong gradient are also given more
    attention when it comes to the depth loss.
    """

    def __init__(self, crop_fraction=0.0, vmin=0, vmax=1, limit=10, weight_scale=1.0):
        """
        The input should be (batch x channels x height x width).
        We L1-penalize the inner portion of the image,
        with crop_fraction cut off from all borders.
        Keyword arguments:
                crop_fraction -- fraction to cut off from all sides (defaults to 0.25)
                vmin -- minimal (GT!) value to supervise
                vmax -- maximal (GT!) value to supervise
                limit -- anything higher than this is wrong, and should be ignored
        """
        super(GradientWeightedDepthLoss, self).__init__()

        self.weight_scale = weight_scale

        self.crop_fraction = crop_fraction
        "Cut-off fraction"

        self.vmin = vmin
        "Lower bound for valid target pixels"

        self.vmax = vmax
        "Upper bound for valid target pixels"

        self.sobel_x = torch.nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.sobel_x.weight = torch.nn.Parameter(
            torch.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8.0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.sobel_y = torch.nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.sobel_y.weight = torch.nn.Parameter(
            torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8.0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        gpu = torch.device("cuda")
        self.sobel_x = self.sobel_x.to(gpu)
        self.sobel_y = self.sobel_y.to(gpu)

        self.limit = limit

    def forward(self, input_, target):
        height = input_.size(2)
        heightcrop = int(height * self.crop_fraction)
        width = input_.size(3)
        widthcrop = int(width * self.crop_fraction)

        if self.crop_fraction > 0:
            input_crop = input_[
                :, :, heightcrop : height - heightcrop, widthcrop : width - widthcrop
            ]
            target_crop = target[
                :, :, heightcrop : height - heightcrop, widthcrop : width - widthcrop
            ]
        else:
            input_crop = input_
            target_crop = target

        valid_mask = (target_crop.le(self.vmax) * target_crop.ge(self.vmin)).float()

        input_gradx = self.sobel_x(input_crop)
        input_grady = self.sobel_y(input_crop)

        target_gradx = self.sobel_x(target_crop)
        target_grady = self.sobel_y(target_crop)

        grad_maskx = self.sobel_x(valid_mask)
        grad_masky = self.sobel_y(valid_mask)
        grad_valid_mask = (grad_maskx.eq(0) * grad_masky.eq(0)).float() * valid_mask

        gradloss = torch.abs((input_gradx - target_gradx)) + torch.abs(
            (input_grady - target_grady)
        )

        # weight l1 loss with gradient
        weights = self.weight_scale * gradloss + torch.ones_like(gradloss)
        gradloss = (gradloss * grad_valid_mask).sum()
        gradloss = gradloss / grad_valid_mask.sum().clamp(min=1)

        loss = torch.abs((input_crop - target_crop) * valid_mask)
        loss = torch.mul(weights, loss).sum()
        loss = loss / valid_mask.sum().clamp(min=1)

        loss = loss + gradloss

        # if this loss value is not plausible, cap it (which will also not backprop gradients)
        if self.limit is not None and loss > self.limit:
            loss = torch.clamp(loss, max=self.limit)

        if loss.ne(loss).item():
            print("Nan loss!")

        return loss


class UncertaintyDepthLoss(torch.nn.Module):
    """
    The loss described in the paper RoutedFusion
    """

    def __init__(self, crop_fraction=0, vmin=0, vmax=1, limit=10, lmbda=0.015):
        """
        The input should be (batch x channels x height x width).
        We L1-penalize the inner portion of the image,
        with crop_fraction cut off from all borders.
        Keyword arguments:
                crop_fraction -- fraction to cut off from all sides (defaults to 0.25)
                vmin -- minimal (GT!) value to supervise
                vmax -- maximal (GT!) value to supervise
                limit -- anything higher than this is wrong, and should be ignored
        """
        super().__init__()

        self.crop_fraction = crop_fraction
        "Cut-off fraction"

        self.vmin = vmin
        "Lower bound for valid target pixels"

        self.vmax = vmax
        "Upper bound for valid target pixels"

        self.sobel_x = torch.nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.sobel_x.weight = torch.nn.Parameter(
            torch.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8.0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.sobel_y = torch.nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.sobel_y.weight = torch.nn.Parameter(
            torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8.0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        gpu = torch.device("cuda")
        self.sobel_x = self.sobel_x.to(gpu)
        self.sobel_y = self.sobel_y.to(gpu)

        self.limit = limit

        self.lmbda = lmbda

    def forward(self, input_, uncertainty, target):
        height = input_.size(2)
        heightcrop = int(height * self.crop_fraction)
        width = input_.size(3)
        widthcrop = int(width * self.crop_fraction)

        if self.crop_fraction > 0:
            input_crop = input_[
                :, :, heightcrop : height - heightcrop, widthcrop : width - widthcrop
            ]
            target_crop = target[
                :, :, heightcrop : height - heightcrop, widthcrop : width - widthcrop
            ]
        else:
            input_crop = input_
            target_crop = target

        valid_mask = (target_crop.le(self.vmax) * target_crop.ge(self.vmin)).float()
        # valid_mask[target == 0] = 0

        input_gradx = self.sobel_x(input_crop)
        input_grady = self.sobel_y(input_crop)

        target_gradx = self.sobel_x(target_crop)
        target_grady = self.sobel_y(target_crop)

        grad_maskx = self.sobel_x(valid_mask)
        grad_masky = self.sobel_y(valid_mask)
        grad_valid_mask = (grad_maskx.eq(0) * grad_masky.eq(0)).float() * valid_mask
        # grad_valid_mask[target == 0] = 0

        s_i = uncertainty
        p_i = torch.exp(-1.0 * s_i)

        gradloss = torch.abs((input_gradx - target_gradx)) + torch.abs(
            (input_grady - target_grady)
        )
        gradloss = gradloss * grad_valid_mask
        gradloss = torch.mul(p_i, gradloss).sum()
        gradloss = gradloss / grad_valid_mask.sum().clamp(min=1)

        loss = torch.abs((input_crop - target_crop) * valid_mask)
        loss = torch.mul(loss, p_i).sum()
        loss = loss / valid_mask.sum().clamp(min=1)

        # sum of loss terms with uncertainty included
        loss = (
            loss
            + gradloss
            + self.lmbda * uncertainty.sum() / valid_mask.sum().clamp(min=1)
        )

        # if this loss value is not plausible, cap it (which will also not backprop gradients)
        if self.limit is not None and loss > self.limit:
            loss = torch.clamp(loss, max=self.limit)

        if loss.ne(loss).item():
            print("Nan loss!")

        return loss


class GradientWeightedUncertaintyDepthLoss(torch.nn.Module):
    """
    This loss combines the loss presented in the RoutedFusion paper and the gradient weighted loss.
    """

    def __init__(
        self, crop_fraction=0.0, vmin=0, vmax=1, limit=10, weight_scale=1.0, lmbda=0.015
    ):
        """
        The input should be (batch x channels x height x width).
        We L1-penalize the inner portion of the image,
        with crop_fraction cut off from all borders.
        Keyword arguments:
                crop_fraction -- fraction to cut off from all sides (defaults to 0.25)
                vmin -- minimal (GT!) value to supervise
                vmax -- maximal (GT!) value to supervise
                limit -- anything higher than this is wrong, and should be ignored
        """
        super().__init__()

        self.weight_scale = weight_scale

        self.crop_fraction = crop_fraction
        "Cut-off fraction"

        self.vmin = vmin
        "Lower bound for valid target pixels"

        self.vmax = vmax
        "Upper bound for valid target pixels"

        self.sobel_x = torch.nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.sobel_x.weight = torch.nn.Parameter(
            torch.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8.0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.sobel_y = torch.nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.sobel_y.weight = torch.nn.Parameter(
            torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8.0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        gpu = torch.device("cuda")
        self.sobel_x = self.sobel_x.to(gpu)
        self.sobel_y = self.sobel_y.to(gpu)

        self.limit = limit

        self.lmbda = lmbda

    def forward(self, input_, uncertainty, target):
        height = input_.size(2)
        heightcrop = int(height * self.crop_fraction)
        width = input_.size(3)
        widthcrop = int(width * self.crop_fraction)

        if self.crop_fraction > 0:
            input_crop = input_[
                :, :, heightcrop : height - heightcrop, widthcrop : width - widthcrop
            ]
            target_crop = target[
                :, :, heightcrop : height - heightcrop, widthcrop : width - widthcrop
            ]
        else:
            input_crop = input_
            target_crop = target

        valid_mask = (target_crop.le(self.vmax) * target_crop.ge(self.vmin)).float()

        input_gradx = self.sobel_x(input_crop)
        input_grady = self.sobel_y(input_crop)

        target_gradx = self.sobel_x(target_crop)
        target_grady = self.sobel_y(target_crop)

        grad_maskx = self.sobel_x(valid_mask)
        grad_masky = self.sobel_y(valid_mask)
        grad_valid_mask = (grad_maskx.eq(0) * grad_masky.eq(0)).float() * valid_mask

        s_i = uncertainty
        p_i = torch.exp(-1.0 * s_i)

        gradloss = torch.abs((input_gradx - target_gradx)) + torch.abs(
            (input_grady - target_grady)
        )

        # weight l1 loss with gradient
        weights = self.weight_scale * gradloss + torch.ones_like(gradloss)
        gradloss = gradloss * grad_valid_mask
        gradloss = torch.mul(p_i, gradloss).sum()
        gradloss = gradloss / grad_valid_mask.sum().clamp(min=1)

        loss = torch.abs((input_crop - target_crop) * valid_mask)
        loss = torch.mul(weights, loss)
        loss = torch.mul(loss, p_i).sum()
        loss = loss / valid_mask.sum().clamp(min=1)

        loss = (
            loss
            + gradloss
            + self.lmbda * uncertainty.sum() / valid_mask.sum().clamp(min=1)
        )

        # if this loss value is not plausible, cap it (which will also not backprop gradients)
        if self.limit is not None and loss > self.limit:
            loss = torch.clamp(loss, max=self.limit)

        if loss.ne(loss).item():
            print("Nan loss!")

        return loss
