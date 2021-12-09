import torch
import datetime
from torch import nn
from torch.nn.functional import normalize
import time


class Extractor(nn.Module):
    """
    This module extracts voxel rays or blocks around voxels that are given by the
    reconstructed 2D depth map as well as the given groundtruth volume and the
    current state of the reconstruction volume
    """

    def __init__(self, config, sensor):

        super(Extractor, self).__init__()

        self.config = config
        try:
            self.n_points = eval("self.config.n_points_" + sensor)
        except:
            self.n_points = self.config.n_points
        self.mode = "ray"
        self.n_empty_space_voting = config.n_empty_space_voting
        self.init_val = config.init_value
        self.extraction_strategy = config.extraction_strategy

    def forward(
        self,
        depth,
        extrinsics,
        intrinsics,
        tsdf_volume,
        feature_volume,
        origin,
        resolution,
        gpu,
        weights_volume,
    ):
        """
        Computes the forward pass of extracting the rays/blocks and the corresponding coordinates

        :param depth: depth map with the values that define the center voxel of the ray/block
        :param extrinsics: camera extrinsics matrix for mapping
        :param intrinsics: camera intrinsics matrix for mapping
        :param volume_gt: groundtruth voxel volume
        :param volume_current: current state of reconstruction volume
        :param origin: origin of groundtruth volume in world coordinates
        :param resolution: resolution of voxel volume
        :return: values/voxels of groundtruth and current as well as at its coordinates and indices
        """

        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()

        if (
            torch.cuda.is_available() and gpu
        ):  # putting extractor on gpu. This makes computations much faster
            intrinsics = intrinsics.cuda()
            extrinsics = extrinsics.cuda()

            tsdf_volume = tsdf_volume.cuda()
            weights_volume = weights_volume.cuda()
            origin = origin.cuda()

        b, h, w = depth.shape

        coords = self.compute_coordinates(
            depth, extrinsics, intrinsics, origin, resolution, gpu
        )

        # compute rays
        eye_w = extrinsics[:, :3, 3]

        ray_pts, empty_points = self.extract_values(
            coords,
            eye_w,
            origin,
            resolution,
            n_points=int((self.n_points - 1) / 2),
            n_empty_space_voting=self.n_empty_space_voting,
        )  # ray_pts are the extracted points in floating point voxel space

        if self.extraction_strategy == "trilinear_interpolation":
            (
                fusion_values,
                indices,
                weights,
                indices_empty,
                weights_empty,
                fusion_weights,
            ) = self.trilinear_interpolation(
                ray_pts, empty_points, tsdf_volume, weights_volume
            )

            n1, n2, n3 = fusion_values.shape

            indices = indices.view(n1, n2, n3, 8, 3)
            weights = weights.view(n1, n2, n3, 8)

            indices_empty = indices_empty.view(n1, n2, self.n_empty_space_voting, 8, 3)
            weights_empty = weights_empty.view(n1, n2, self.n_empty_space_voting, 8)
        elif self.extraction_strategy == "nearest_neighbor":
            (
                fusion_values,
                indices,
                weights,
                indices_empty,
                weights_empty,
                fusion_weights,
            ) = self.nearest_neighbor_extraction_tsdf(
                ray_pts, empty_points, tsdf_volume, weights_volume
            )

            n1, n2, n3 = fusion_values.shape

            indices = indices.view(n1, n2, n3, 1, 3)
            weights = weights.view(n1, n2, n3, 1)

            indices_empty = indices_empty.view(n1, n2, self.n_empty_space_voting, 1, 3)
            weights_empty = weights_empty.view(n1, n2, self.n_empty_space_voting, 1)

        # packing
        values = dict(
            fusion_values=fusion_values,
            fusion_weights=fusion_weights,
            indices=indices,
            weights=weights,
            indices_empty=indices_empty,
            weights_empty=weights_empty,
        )

        del extrinsics, intrinsics, origin, weights_volume, tsdf_volume

        return values

    def compute_coordinates(
        self, depth, extrinsics, intrinsics, origin, resolution, gpu
    ):

        b, h, w = depth.shape
        n_points = h * w

        # generate frame meshgrid
        xx, yy = torch.meshgrid(
            [torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float)]
        )

        if torch.cuda.is_available() and gpu:  # putting extractor on gpu
            xx = xx.cuda()
            yy = yy.cuda()

        # flatten grid coordinates and bring them to batch size
        xx = xx.contiguous().view(1, h * w, 1).repeat((b, 1, 1))
        yy = yy.contiguous().view(1, h * w, 1).repeat((b, 1, 1))
        zz = depth.contiguous().view(b, h * w, 1)

        # generate points in pixel space
        points_p = torch.cat((yy, xx, zz), dim=2).clone()

        # invert
        intrinsics_inv = intrinsics.inverse().float()

        homogenuous = torch.ones((b, 1, n_points))

        if torch.cuda.is_available() and gpu:  # putting extractor on gpu
            homogenuous = homogenuous.cuda()

        # transform points from pixel space to camera space to world space (p->c->w)
        points_p[:, :, 0] *= zz[:, :, 0]
        points_p[:, :, 1] *= zz[:, :, 0]
        points_c = torch.matmul(
            intrinsics_inv, torch.transpose(points_p, dim0=1, dim1=2)
        )
        points_c = torch.cat((points_c, homogenuous), dim=1)
        points_w = torch.matmul(extrinsics[:3], points_c)
        points_w = torch.transpose(points_w, dim0=1, dim1=2)[:, :, :3]

        del xx, yy, homogenuous, points_p, points_c, intrinsics_inv
        return points_w

    def extract_values(
        self,
        coords,
        eye,
        origin,
        resolution,
        bin_size=1.0,
        n_points=4,
        n_empty_space_voting=0,
        ellipsoid=False,
    ):

        center_v = (coords - origin) / resolution
        eye_v = (
            eye - origin
        ) / resolution  # camera center in the voxel coordinate space

        direction = center_v - eye_v
        direction = normalize(direction, p=2, dim=2)

        points = [center_v]

        # ellip = []

        empty_points = []
        # dist = torch.zeros_like(center_v)[:, :, 0]
        # dists = [dist]

        for i in range(1, n_points + 1):
            point = center_v + i * bin_size * direction
            pointN = center_v - i * bin_size * direction
            points.append(point.clone())
            points.insert(0, pointN.clone())

        for i in range(1, n_empty_space_voting + 1):
            point = center_v - (8 * i + n_points) * bin_size * direction
            empty_points.insert(0, point.clone())

            # dist = i*bin_size*torch.ones_like(point)[:, :, 0]
            # distN = -1.*dist

            # dists.append(dist)
            # dists.insert(0, distN)

        # if ellipsoid:
        #     points = points + ellip

        # dists = torch.stack(dists, dim=2)
        points = torch.stack(points, dim=2)

        empty_points = torch.stack(empty_points, dim=2)  # (1, 65536, n_empty_pints, 3)

        return points, empty_points

    def nearest_neighbor_extraction_tsdf(
        self, points, empty_points, tsdf_volume, weights_volume
    ):
        x, y, z = tsdf_volume.shape
        b, h, n, dim = points.shape

        # get indices from the points which are already in the voxel grid but floating point coordinates
        points = points.contiguous().view(b * h * n, dim)
        indices = torch.cat(
            (
                torch.round(points[:, 0].unsqueeze_(-1)),
                torch.round(points[:, 1].unsqueeze_(-1)),
                torch.round(points[:, 2].unsqueeze_(-1)),
            ),
            dim=-1,
        ).long()

        # get valid indices
        valid = get_index_mask(indices, (x, y, z))

        valid_idx = torch.nonzero(valid)[:, 0]

        tsdf = extract_values(indices, tsdf_volume, valid)
        weights = extract_values(indices, weights_volume, valid)

        tsdf_container = self.init_val * torch.ones_like(valid).float()
        weight_container = torch.zeros_like(valid).float()

        # feature_container = 0 * torch.ones((valid.shape[0], nbr_features)).float()
        tsdf_container[valid_idx] = tsdf.float()
        weight_container[valid_idx] = weights.float()

        weights = torch.ones_like(valid).float()

        fusion_values = tsdf_container.view(b, h, n)
        fusion_weights = weight_container.view(b, h, n)

        del tsdf

        # handle the empty points
        b, h, n, dim = empty_points.shape

        # get indices from the points which are already in the voxel grid but floating point coordinates
        points = empty_points.contiguous().view(b * h * n, dim)
        indices_empty = torch.cat(
            (
                torch.round(points[:, 0].unsqueeze_(-1)),
                torch.round(points[:, 1].unsqueeze_(-1)),
                torch.round(points[:, 2].unsqueeze_(-1)),
            ),
            dim=-1,
        ).long()
        weights_empty = torch.ones(points.shape[0], device=self.config.device).float()

        return (
            fusion_values,
            indices,
            weights,
            indices_empty,
            weights_empty,
            fusion_weights,
        )

    def trilinear_interpolation(
        self, points, empty_points, tsdf_volume, weights_volume
    ):

        b, h, n, dim = points.shape

        # get interpolation weights
        weights, indices = interpolation_weights(points)

        weights_empty, indices_empty = interpolation_weights(
            empty_points
        )  # check speed of this - perhaps assign weight 1 to each.

        n1, n2, n3 = indices.shape
        # nbr_features = feature_volume.shape[-1]
        indices = indices.contiguous().view(n1 * n2, n3).long()

        # TODO: change to replication padding instead of zero padding
        # TODO: double check indices

        # get valid indices
        valid = get_index_mask(indices, tsdf_volume.shape)
        valid_idx = torch.nonzero(valid)[:, 0]

        tsdf_values = extract_values(indices, tsdf_volume, valid)
        tsdf_weights = extract_values(indices, weights_volume, valid)
        # features = extract_values(indices, feature_volume, valid)

        value_container = self.init_val * torch.ones_like(valid).float()
        weight_container = torch.zeros_like(valid).float()
        # feature_container = 0 * torch.ones((valid.shape[0], nbr_features), device='cuda:0').float() # here we should extract the initial value of the confidence grid
        # feature_container = 0 * torch.ones((valid.shape[0], nbr_features)).float()

        value_container[valid_idx] = tsdf_values.float()
        weight_container[valid_idx] = tsdf_weights.float()
        # I forgot to put the variable features into the container here before. FAIL!
        # feature_container[valid_idx, :] = features.float()

        value_container = value_container.view(weights.shape)
        weight_container = weight_container.view(weights.shape)
        # feature_container = feature_container.view(weights.shape[0], weights.shape[1], nbr_features)

        # trilinear interpolation
        fusion_values = torch.sum(value_container * weights, dim=1)
        fusion_weights = torch.sum(weight_container * weights, dim=1)
        weights = weights.unsqueeze_(-1)
        # feature_weights = weights.repeat(1, 1, nbr_features)
        # fusion_features = torch.sum(feature_container * feature_weights, dim=1)

        fusion_values = fusion_values.view(b, h, n)
        fusion_weights = fusion_weights.view(b, h, n)
        # fusion_features = fusion_features.view(b, h, n, nbr_features)

        indices = indices.view(n1, n2, n3)

        # return fusion_values.float(), fusion_features.float(), indices, weights, indices_empty, weights_empty, fusion_weights.float()
        return (
            fusion_values.float(),
            indices,
            weights,
            indices_empty,
            weights_empty,
            fusion_weights.float(),
        )

    def nearest_neighbor_extraction(
        self, points, feature_volume, feature_weights_volume
    ):
        b, h, n, dim = points.shape
        x, y, z, nbr_features = feature_volume.shape

        # get indices from the points which are already in the voxel grid but floating point coordinates
        points = points.contiguous().view(b * h * n, dim)
        indices = torch.cat(
            (
                torch.round(points[:, 0].unsqueeze_(-1)),
                torch.round(points[:, 1].unsqueeze_(-1)),
                torch.round(points[:, 2].unsqueeze_(-1)),
            ),
            dim=-1,
        ).long()

        # get valid indices
        valid = get_index_mask(indices, (x, y, z))

        valid_idx = torch.nonzero(valid)[:, 0]

        features = extract_values(indices, feature_volume, valid)
        weights = extract_values(indices, feature_weights_volume, valid)

        feature_container = (
            0
            * torch.ones(
                (valid.shape[0], nbr_features), device=self.config.device
            ).float()
        )
        weight_container = torch.zeros_like(valid).float()

        # feature_container = 0 * torch.ones((valid.shape[0], nbr_features)).float()
        feature_container[valid_idx, :] = features.float()
        weight_container[valid_idx] = weights.float()

        del features, weights

        feature_container = feature_container.view(b, h, n, nbr_features)
        weight_container = weight_container.view(b, h, n)

        return feature_container, indices, weight_container


def interpolation_weights(points):

    floored = torch.floor(points)
    neighbor = torch.sign(points - floored)  # always one

    # index of center voxel
    idx = torch.floor(points)

    # reshape for pytorch compatibility
    b, h, n, dim = idx.shape
    points = points.contiguous().view(b * h * n, dim)
    floored = floored.contiguous().view(b * h * n, dim)
    idx = idx.contiguous().view(b * h * n, dim)
    neighbor = neighbor.contiguous().view(b * h * n, dim)

    # center x.0
    alpha = torch.abs(points - floored)  # always positive
    alpha_inv = 1 - alpha

    weights = []
    indices = []

    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                if i == 0:
                    w1 = alpha_inv[:, 0]
                    ix = idx[:, 0]
                else:
                    w1 = alpha[:, 0]
                    ix = idx[:, 0] + neighbor[:, 0]
                if j == 0:
                    w2 = alpha_inv[:, 1]
                    iy = idx[:, 1]
                else:
                    w2 = alpha[:, 1]
                    iy = idx[:, 1] + neighbor[:, 1]
                if k == 0:
                    w3 = alpha_inv[:, 2]
                    iz = idx[:, 2]
                else:
                    w3 = alpha[:, 2]
                    iz = idx[:, 2] + neighbor[:, 2]

                weights.append((w1 * w2 * w3).unsqueeze_(1))
                indices.append(
                    torch.cat(
                        (ix.unsqueeze_(1), iy.unsqueeze_(1), iz.unsqueeze_(1)), dim=1
                    ).unsqueeze_(1)
                )

    weights = torch.cat(weights, dim=1)
    indices = torch.cat(indices, dim=1)

    del points, floored, idx, neighbor, alpha, alpha_inv, ix, iy, iz, w1, w2, w3

    return weights, indices


def get_index_mask(indices, shape):

    xs, ys, zs = shape

    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < xs)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < ys)
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < zs)
    )

    return valid


def extract_values(indices, volume, mask=None):
    if volume.dim() == 3:
        if mask is not None:
            x = torch.masked_select(
                indices[:, 0], mask
            )  # filters away the indices that are flagged as outside the grid
            y = torch.masked_select(indices[:, 1], mask)
            z = torch.masked_select(indices[:, 2], mask)
        else:
            x = indices[:, 0]
            y = indices[:, 1]
            z = indices[:, 2]

        return volume[x, y, z]
    else:  # feature volume processing
        if mask is not None:
            x = torch.masked_select(
                indices[:, 0], mask
            )  # filters away the indices that are flagged as outside the grid
            y = torch.masked_select(indices[:, 1], mask)
            z = torch.masked_select(indices[:, 2], mask)
        else:
            x = indices[:, 0]
            y = indices[:, 1]
            z = indices[:, 2]

        return volume[x, y, z, :]
