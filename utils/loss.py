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
        self.refinement_loss = config.FILTERING_MODEL.CONV3D_MODEL.use_refinement
        self.add_outlier_channel = config.FILTERING_MODEL.CONV3D_MODEL.outlier_channel
        self.use_fusion_net = config.FUSION_MODEL.use_fusion_net

        self.l1 = torch.nn.L1Loss(reduction=reduction)
        self.l2 = torch.nn.MSELoss(reduction=reduction)
        self.bce = torch.nn.BCEWithLogitsLoss(
            reduction=reduction
        )  # , pos_weight=self.focus_outliers_weight)

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

            # remove the indices where only one sensor integrates since these values will contribute to the loss, but we
            # won't compute a gradient for these so they just corrupt the loss representation
            mask = torch.ones_like(l1_grid)
            for sensor_ in self.sensors:
                mask = torch.logical_and(
                    mask,
                    output["filtered_output"]["tsdf_filtered_grid"][sensor_ + "_init"],
                )

            # filter the mask so that we also remove the voxels where the ground truth tsdf is not between
            # the sensor predictions
            # mask_between = torch.logical_or(torch.logical_and((output['filtered_output']['tsdf_filtered_grid']['tsdf_' + self.sensors[0]] < target_grid), \
            #  (output['filtered_output']['tsdf_filtered_grid']['tsdf_' + self.sensors[1]] > target_grid)), \
            # 		 torch.logical_and((output['filtered_output']['tsdf_filtered_grid']['tsdf_' + self.sensors[1]] < target_grid), \
            # 		 	(output['filtered_output']['tsdf_filtered_grid']['tsdf_' + self.sensors[0]] > target_grid)))
            # print('indices both sensors: ', mask.sum())
            # # here we make sure that we only take the voxels where both sensors integrate into account
            # mask = torch.logical_and(mask, mask_between)
            normalization = torch.ones_like(l1_grid[mask]).sum()
            l1_grid = l1_grid[mask].sum() / normalization
            # print('l1 grid: ', l1_grid)

            # print('l1_grid', l1_grid)
            # Note: if you only want to compute the loss for a subset of your full loss - for debugging etc. make sure to uncomment the criterion of the
            # other loss terms since these seem to cause memory problems otherwise - I suppose those losses somehow gets saved and not emptied.

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
            # note that if the target alpha is -1 we need to remove it since it is outside of the valid grid
            # compute the valid mask
            # mask = torch.zeros_like(mask)
            # for sensor_ in self.sensors:
            # 	mask = torch.logical_or(mask, output['filtered_output']['tsdf_filtered_grid'][sensor_ + '_init'])

            # remove indices where the mask contains negative values. This can happen if we use the median filtered
            # proxy alpha since I cannot control how the median filtering is done (did not have time). Thus,
            # in the event that I have a GT -1 (uninitialized value), remove this.
            pos_proxy_alpha = output["filtered_output"]["proxy_alpha_grid"] >= 0
            mask = torch.logical_and(mask, pos_proxy_alpha)

            l1_alpha = self.l1.forward(
                output["filtered_output"]["alpha_grid"][mask],
                output["filtered_output"]["proxy_alpha_grid"][mask],
            )
            normalization = torch.ones_like(l1_alpha).sum()
            l1_alpha = l1_alpha.sum() / normalization
            # print('l1_alpha: ', l1_alpha)
            if l is None:
                l = self.alpha_weight * l1_alpha
            else:
                l += self.alpha_weight * l1_alpha

        if self.alpha_single_sensor_supervision and "filtered_output" in output:
            # this loss is intended to improve outlier filtering performance which is
            # specifically needed where only one sensor integrates, since we do not supervise
            # these voxels at all otherwise. The idea is to extract all voxels where only one sensor
            # integrates and check the tsdf error compared to the ground truth. If the error is larger than
            # e.g. 0.04 m, we say that it is an outlier. This means setting the target alpha as "no confidence"
            # to the sensor, otherwise 100 percent confidence.
            # outlier_ratio = dict()
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
                # print('tot ind: ', tsdf_err.shape)
                # print('err larger than thresh: ', (tsdf_err > 0.04).sum())
                # gt 0, stereo 1. When alpha is 1 we trust gt

                if k == 0:
                    target_alpha = ~torch.logical_and(tsdf_err > 0.04, sgn_err > 0)
                    # target_alpha = ~(tsdf_err > 0.04)
                    outlier_alpha = target_alpha is False
                    inlier_alpha = target_alpha is True
                    target_alpha = target_alpha.float()
                    # outlier_ratio[sensor_] = ((~(target_alpha.bool())).sum()/target_alpha.shape[0])
                elif k == 1:
                    # target_alpha = torch.logical_and(tsdf_err > g0.04, sgn_err > 0)
                    # target_alpha = tsdf_err > 0.04
                    outlier_alpha = target_alpha is True
                    inlier_alpha = target_alpha is False
                    target_alpha = target_alpha.float()
                    # outlier_ratio[sensor_] = (target_alpha.sum()/target_alpha.shape[0])

                # print(sensor_)
                # print((target_alpha.sum()/target_alpha.shape[0]))
                # print(single_sensor_mask.sum())
                # only outleir if tsdf err is larger than 0.04 and sgn err, toherwise inlier

                if single_sensor_mask.sum() > 0:
                    # ind = (target_grid[single_sensor_mask] > 0.099).nonzero()[:, 0]
                    # outliers = torch.logical_and(tsdf_err[ind] > 0.01, sgn_err[ind] > 0).float()
                    # # outliers = (tsdf_err[ind] > 0.01).float()
                    # print(ind.shape)
                    # # print(est_grid[single_sensor_mask][ind])

                    # # print(tsdf_err[ind])
                    # print((outliers.sum()/tsdf_err[ind].shape[0]))
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
                        # print('l1 outlier: ', l1_outlier_alpha)

        if self.refinement_loss:
            l1_grid_dict = dict()
            for sensor_ in self.sensors:
                l1_grid_dict[sensor_] = self.l1.forward(
                    est_grid_dict[sensor_][init[sensor_]], target_grid[init[sensor_]]
                )
                if l1_grid_dict[sensor_].shape[0] == 0:
                    l1_grid_dict[sensor_] = None
                else:
                    normalization = torch.ones_like(l1_grid_dict[sensor_]).sum()
                    l1_grid_dict[sensor_] = l1_grid_dict[sensor_].sum() / normalization
                    # Note: if you only want to compute the loss for a subset of your full loss - for debugging etc. make sure to uncomment the criterion of the
                    # other loss terms since these seem to cause memory problems otherwise - I suppose those losses somehow gets saved and not emptied.
                    if l is None:
                        l = self.grid_weight / 2 * l1_grid_dict[sensor_]
                    else:
                        l += self.grid_weight / 2 * l1_grid_dict[sensor_]
                    if (
                        l1_grid_dict[sensor_].isnan() > 0
                        or l1_grid_dict[sensor_].isinf()
                    ):
                        print(sensor_ + "loss nan:", l1_grid_dict[sensor_].isnan())
                        print(sensor_ + "loss inf: ", l1_grid_dict[sensor_].isinf())
                        print("normalization factor: ", normalization)
                        print(
                            "l1_grid_dict[sensor_].shape:", l1_grid_dict[sensor_].shape
                        )

        else:
            l1_grid_dict = dict()
            for sensor_ in self.sensors:
                l1_grid_dict[sensor_] = None

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
        output["loss"] = l
        output["l1_interm"] = l1_interm  # this mixes all sensors in one logging graph
        output["l1_grid"] = l1_grid
        for sensor_ in self.sensors:
            # print(l1_grid_dict[sensor_])
            output["l1_grid_" + sensor_] = l1_grid_dict[sensor_]
        # print('l: ', l)
        # print('l1_grid: ', l1_grid)

        return output


class FusionLoss(torch.nn.Module):
    def __init__(self, config, reduction="none", l1=True, l2=True, cos=True):
        super(FusionLoss, self).__init__()

        self.criterion1 = torch.nn.L1Loss(reduction=reduction)
        self.criterion2 = torch.nn.MSELoss(reduction=reduction)
        self.criterion3 = torch.nn.CosineEmbeddingLoss(
            margin=0.0, reduction="mean"
        )  # we use mean because the cosineembedingloss otherwise gives the wrong loss
        # when we take the sum and divide by the numbe of elements - at the core, the problem is that the cosineembeddingloss gives too many copies of the same output

        self.lambda1 = 1.0 if l1 else 0.0
        self.lambda2 = 0.0 if l2 else 0.0
        self.lambda3 = 0.1 if cos else 0.0

    def forward(self, est, target):

        if est.shape[1] == 0:
            return torch.ones_like(est).sum().clamp(min=1)

        x1 = torch.sign(est)
        x2 = torch.sign(target)

        x1 = x1.reshape(
            [x1.shape[0], x1.shape[2], x1.shape[1]]
        )  # we reshape to compute the cosine loss over the rays at a spatial location
        # if no reshaping is done, the loss is computed for a constant extraction depth over some spatial location.
        x2 = x2.reshape([x2.shape[0], x2.shape[2], x2.shape[1]])

        label = torch.ones_like(x1)

        l1 = self.criterion1.forward(est, target)
        l2 = self.criterion2.forward(est, target)
        l3 = self.criterion3.forward(x1, x2, label)

        normalization = torch.ones_like(l1).sum()

        l1 = l1.sum() / normalization
        l2 = l2.sum() / normalization

        # NOTE: the l3 loss gives zero-gradients so no learning happens due to L3!
        l = self.lambda1 * l1 + self.lambda2 * l2 + self.lambda3 * l3

        return l, l1, l3


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

        self.criterion3 = VNL_Loss(
            focal_x=config.DATA.focalx,
            focal_y=config.DATA.focaly,
            input_size=[config.DATA.resx, config.DATA.resy],
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
        elif self.combined == "VNL + gradweighted + uncertainty":
            l1 = self.criterion1.forward(prediction, target)
            l2 = self.criterion2.forward(prediction, uncertainty, target)
            l3 = self.criterion3.forward(prediction, target)
            return l1 + l2 + 5 * l3
        elif self.combined == "VNL + gradweighteduncertainty":
            l1 = self.criterion3.forward(prediction, target)
            l2 = self.criterion4.forward(prediction, uncertainty, target)
            return l1 + l2
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


class VNL_Loss(
    torch.nn.Module
):  # check how this loss uses vmin and vmax for the sampling
    """
    The loss described in the paper Enforcing geometric constraints of virtual normal for depth prediction
    """

    def __init__(
        self,
        focal_x,
        focal_y,
        input_size,
        delta_cos=0.867,
        delta_diff_x=0.35,
        delta_diff_y=0.35,
        delta_diff_z=0.35,
        delta_z=0.05,
        delta_far_z=12.3,
        sample_ratio=0.20,
    ):
        super(VNL_Loss, self).__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32).cuda()
        self.fy = torch.tensor([focal_y], dtype=torch.float32).cuda()
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).cuda()
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.delta_far_z = delta_far_z
        self.sample_ratio = sample_ratio

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)  # [b, h, w, c]
        return pw

    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(np.int)

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(np.int)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(np.int)
        p123 = {
            "p1_x": p1_x,
            "p1_y": p1_y,
            "p2_x": p2_x,
            "p2_y": p2_y,
            "p3_x": p3_x,
            "p3_y": p3_y,
        }
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each group.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        p1_x = p123["p1_x"]
        p1_y = p123["p1_y"]
        p2_x = p123["p2_x"]
        p2_y = p123["p2_y"]
        p3_x = p123["p3_x"]
        p3_y = p123["p3_y"]

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat(
            [
                pw1[:, :, :, np.newaxis],
                pw2[:, :, :, np.newaxis],
                pw3[:, :, :, np.newaxis],
            ],
            3,
        )
        # print(pw_groups.shape)
        return pw_groups

    def filter_mask(
        self,
        p123,
        gt_xyz,
        delta_cos=0.867,
        delta_diff_x=0.35,
        delta_diff_y=0.35,
        delta_diff_z=0.35,
    ):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        # ignore linear
        pw_diff = torch.cat(
            [
                pw12[:, :, :, np.newaxis],
                pw13[:, :, :, np.newaxis],
                pw23[:, :, :, np.newaxis],
            ],
            3,
        )  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1)
        # print(proj_query.shape) # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(
            m_batchsize * groups, -1, index
        )  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        # print(q_norm)
        # print(q_norm.shape)
        # print(q_norm.view(m_batchsize * groups, index, 1).shape)
        # print(q_norm.view(m_batchsize * groups, 1, index).shape)
        nm = torch.bmm(
            q_norm.view(m_batchsize * groups, index, 1),
            q_norm.view(m_batchsize * groups, 1, index),
        )  # []
        # print('nm', nm.shape)
        energy = torch.bmm(proj_query, proj_key)
        # print(energy.shape)  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        # print(norm_energy.shape)
        # print(norm_energy)
        mask_cos = (
            torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3
        )  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)
        # ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3
        mask_pad_far = torch.sum(pw[:, :, 2, :] < self.delta_far_z, 2) == 3

        # ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near & mask_pad_far

        return mask, pw

    def select_points_groups(self, gt_depth, pred_depth):
        pw_gt = self.transfer_xyz(gt_depth)
        pw_pred = self.transfer_xyz(pred_depth)
        B, C, H, W = gt_depth.shape
        p123 = self.select_index()
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(
            p123,
            pw_gt,
            delta_cos=0.867,
            delta_diff_x=0.35,
            delta_diff_y=0.35,
            delta_diff_z=0.35,
        )

        # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        # print(pw_groups_pred.shape)
        # print(mask.shape)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        # print(pw_groups_pred_not_ignore.shape)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)
        print(pw_groups_gt_not_ignore)
        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, pred_depth, gt_depth, select=True):
        """
        Virtual normal loss.
        :param pred_depth: predicted depth map, [B,W,H,C]
        :param data: target label, ground truth depth, [B, W, H, C], padding region [padding_up, padding_down]
        :return:
        """
        gt_points, dt_points = self.select_points_groups(gt_depth, pred_depth)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25) :]
        loss = torch.mean(loss)
        return loss


if __name__ == "__main__":
    from skimage import io

    vnl_loss = VNL_Loss(256, 256, (512, 512))
    pred_depth = "/home/esandstroem/scratch-second/opportunistic_3d_capture/data/habitat/replica/manual/room_0/1/left_psmnet_depth/0.png"
    pred_depth = io.imread(pred_depth)
    pred_depth = np.asarray(pred_depth) / 1000
    pred_depth = np.expand_dims(pred_depth, axis=0)
    pred_depth = np.expand_dims(pred_depth, axis=0)
    # print(pred_depth.shape)
    gt_depth = "/home/esandstroem/scratch-second/opportunistic_3d_capture/data/habitat/replica/manual/room_0/1/left_depth_gt/0.png"
    gt_depth = io.imread(gt_depth)
    gt_depth = np.asarray(gt_depth) / 1000
    gt_depth = np.expand_dims(gt_depth, axis=0)
    gt_depth = np.expand_dims(gt_depth, axis=0)
    # gt_depth = np.ones([2, 1, 480, 640])
    gt_depth = torch.tensor(np.asarray(gt_depth, np.float32)).cuda()
    pred_depth = torch.tensor(np.asarray(pred_depth, np.float32)).cuda()
    loss = vnl_loss.forward(pred_depth, gt_depth)
    print(loss)
