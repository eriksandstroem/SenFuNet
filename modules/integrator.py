import torch


class Integrator(torch.nn.Module):
    def __init__(self, config):

        super(Integrator, self).__init__()

        self.device = config.device
        self.max_weight = config.max_weight
        self.extraction_strategy = config.extraction_strategy
        self.n_empty_space_voting = config.n_empty_space_voting
        self.trunc_value = config.trunc_value

    def forward(
        self,
        integrator_input,
        values_volume,
        features_volume,
        weights_volume,
    ):
        xs, ys, zs = values_volume.shape

        # unpack data
        values = integrator_input["update_values"].to(self.device)
        features = integrator_input["update_features"].to(self.device)
        indices = integrator_input["update_indices"].to(self.device)
        weights = integrator_input["update_weights"].to(
            self.device
        )  # update weights. When using nearest neighbor interpolation these are all ones.

        if self.n_empty_space_voting > 0:
            indices_empty = integrator_input["update_indices_empty"].to(self.device)
            weights_empty = integrator_input["update_weights_empty"].to(self.device)

        (
            n1,
            n2,
            n3,
            f4,
        ) = (
            features.shape
        )  # f1 = 1, f2= 65536 (when no filtering), f3= tail_points, f4 = nbr_features

        # reshape tensors
        features = features.contiguous().view(-1, f4).float()
        values = values.contiguous().view(-1, 1).float()

        if self.extraction_strategy == "trilinear_interpolation":
            features = features.repeat(8, 1)
            values = values.repeat(1, 8)
            indices = indices.contiguous().view(-1, 8, 3).long()
            weights = weights.contiguous().view(-1, 8)
            if self.n_empty_space_voting > 0:
                indices_empty = indices_empty.contiguous().view(-1, 8, 3).long()
                weights_empty = weights_empty.contiguous().view(-1, 8)
        elif self.extraction_strategy == "nearest_neighbor":
            values = values.repeat(1, 1)
            indices = indices.contiguous().view(-1, 1, 3).long()
            weights = weights.contiguous().view(-1, 1)
            if self.n_empty_space_voting > 0:
                indices_empty = indices_empty.contiguous().view(-1, 1, 3).long()
                weights_empty = weights_empty.contiguous().view(-1, 1)

        values = values.contiguous().view(-1, 1).float()
        indices = indices.contiguous().view(-1, 3).long()

        if self.n_empty_space_voting > 0:
            indices_empty = (
                indices_empty.contiguous().view(-1, 3).long()
            )  # (65536*7*8, 3)
            weights_empty = weights_empty.contiguous().view(-1, 1).float()

        weights = weights.contiguous().view(-1, 1).float()

        # get valid indices
        valid = get_index_mask(indices, values_volume.shape)
        indices = extract_indices(indices, mask=valid)
        if self.n_empty_space_voting > 0:
            valid_empty = get_index_mask(indices_empty, values_volume.shape)
            indices_empty = extract_indices(indices_empty, mask=valid_empty)

        feature_indices = indices.clone()

        # remove the invalid entries from the values, features and weights
        valid_features = valid.clone().unsqueeze_(-1)
        features = torch.masked_select(
            features, valid_features.repeat(1, f4)
        )  # (65536*7*8*6) if all indices are valid, otherwise less
        features = features.view(
            int(features.shape[0] / f4), f4
        )  # (65536*7*8, 6) if all indices are valid, otherwise less

        values = torch.masked_select(values[:, 0], valid)
        weights = torch.masked_select(weights[:, 0], valid)
        if self.n_empty_space_voting > 0:
            weights_empty = torch.masked_select(weights_empty[:, 0], valid_empty)

        update_feat = weights.repeat(f4, 1).permute(1, 0) * features
        del features

        update = weights * values
        del values

        # aggregate updates to the same index

        # tsdf
        index = ys * zs * indices[:, 0] + zs * indices[:, 1] + indices[:, 2]
        indices_insert = torch.unique_consecutive(
            indices[index.sort()[1]], dim=0
        )  # since the coalesce() operation on the sparse tensors sorts the
        vcache = torch.sparse.FloatTensor(
            index.unsqueeze_(0), update, torch.Size([xs * ys * zs])
        ).coalesce()
        update = vcache.values()

        if indices_insert.shape[0] != update.shape[0]:
            print("wrong dim!")
        del vcache

        # if using the same extraction procedure for fusion and feature updates
        update_feat_weights = weights

        # weights for tsdf
        wcache = torch.sparse.FloatTensor(
            index, weights, torch.Size([xs * ys * zs])
        ).coalesce()  # this line adds the values at the same index together
        indices = wcache.indices().squeeze()
        weights = wcache.values()

        del wcache

        if self.n_empty_space_voting > 0:
            # weights for empty indices
            index_empty = (
                ys * zs * indices_empty[:, 0]
                + zs * indices_empty[:, 1]
                + indices_empty[:, 2]
            )
            indices_empty_insert = torch.unique_consecutive(
                indices_empty[index_empty.sort()[1]], dim=0
            )  # since the coalesce() operation on the sparse tensors sorts the
            wcache_empty = torch.sparse.FloatTensor(
                index_empty.unsqueeze_(0), weights_empty, torch.Size([xs * ys * zs])
            ).coalesce()  # this line adds the values at the same index together
            indices_empty = wcache_empty.indices().squeeze()
            weights_empty = wcache_empty.values()
            del wcache_empty

        # features
        feature_index = (
            ys * zs * feature_indices[:, 0]
            + zs * feature_indices[:, 1]
            + feature_indices[:, 2]
        )
        feature_indices_insert = torch.unique_consecutive(
            feature_indices[feature_index.sort()[1]], dim=0
        )
        fcache = torch.sparse.FloatTensor(
            feature_index.unsqueeze_(0), update_feat, torch.Size([xs * ys * zs, f4])
        ).coalesce()

        feature_indices = fcache.indices().squeeze()
        update_feat = fcache.values()
        if feature_indices_insert.shape[0] != update_feat.shape[0]:
            print("wrong dim feat!")
        del fcache

        # feature weights
        wcache_feat = torch.sparse.FloatTensor(
            feature_index, update_feat_weights, torch.Size([xs * ys * zs])
        ).coalesce()
        weights_feat = wcache_feat.values().unsqueeze_(-1).repeat(1, f4).float()
        del wcache_feat

        # tsdf and weights update
        values_old = values_volume.view(xs * ys * zs)[indices]
        weights_old = weights_volume.view(xs * ys * zs)[indices]
        value_update = (weights_old * values_old + update) / (weights_old + weights)
        weight_update = weights_old + weights
        weight_update = torch.clamp(weight_update, 0, self.max_weight)

        if self.n_empty_space_voting > 0:
            # empty space update
            values_old_empty = values_volume.view(xs * ys * zs)[indices_empty]
            weights_old_empty = weights_volume.view(xs * ys * zs)[indices_empty]
            value_update_empty = torch.add(
                weights_old_empty * values_old_empty, self.trunc_value * weights_empty
            ) / (weights_old_empty + weights_empty)
            weight_update_empty = weights_old_empty + weights_empty
            weight_update_empty = torch.clamp(weight_update_empty, 0, self.max_weight)

        # feature update
        feature_weights_old = (
            weights_volume.view(xs * ys * zs)[feature_indices]
            .unsqueeze_(-1)
            .repeat(1, f4)
            .float()
        )
        features_old = features_volume.view(xs * ys * zs, f4)[feature_indices]

        # here we should not multiply the update_feat with weights_feat in the nominator since we already have that baked in
        feature_update = (feature_weights_old * features_old + update_feat) / (
            feature_weights_old + weights_feat
        )

        del update_feat, feature_weights_old, weights_feat

        # inser tsdf and tsdf weights
        insert_values(value_update, indices_insert, values_volume)
        insert_values(weight_update, indices_insert, weights_volume)

        # insert features
        insert_values(feature_update, feature_indices_insert, features_volume)

        if self.n_empty_space_voting > 0:
            # insert empty tsdf and weights
            insert_values(value_update_empty, indices_empty_insert, values_volume)
            insert_values(weight_update_empty, indices_empty_insert, weights_volume)

        return (
            values_volume,
            features_volume,
            weights_volume,
            indices_insert,
        )


def get_index_mask(indices, shape):
    """
    method to check whether indices are valid
    :param indices: indices to check
    :param shape: constraints for indices
    :return: mask
    """
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
    """
    method to extract values from volume given indices
    :param indices: positions to extract
    :param volume: volume to extract from
    :param mask: optional mask for extraction
    :return: extracted values
    """

    if mask is not None:
        x = torch.masked_select(indices[:, 0], mask)
        y = torch.masked_select(indices[:, 1], mask)
        z = torch.masked_select(indices[:, 2], mask)
    else:
        x = indices[:, 0]
        y = indices[:, 1]
        z = indices[:, 2]

    return volume[x, y, z]


def extract_indices(indices, mask):
    """
    method to extract indices according to mask
    :param indices:
    :param mask:
    :return:
    """

    x = torch.masked_select(indices[:, 0], mask)
    y = torch.masked_select(indices[:, 1], mask)
    z = torch.masked_select(indices[:, 2], mask)

    masked_indices = torch.cat(
        (x.unsqueeze_(1), y.unsqueeze_(1), z.unsqueeze_(1)), dim=1
    )
    return masked_indices


def insert_values(values, indices, volume):
    """
    method to insert values back into volume
    :param values:
    :param indices:
    :param volume:
    :return:
    """
    # print(volume.dtype)
    # print(values.dtype)
    if volume.dim() == 3:
        volume = volume.half()
        # indices = indices[~torch.isnan(values)] # remove those indices where values = nan (due to round off error when confidence is too low)
        # values = values[~torch.isnan(values)] # remove those values where values = nan (due to round off error when confidence is too low)
        volume[indices[:, 0], indices[:, 1], indices[:, 2]] = values.half()
    else:
        volume = volume.half()
        # indices = indices[~torch.isnan(values)] # remove those indices where values = nan (due to round off error when confidence is too low)
        # values = values[~torch.isnan(values)] # remove those values where values = nan (due to round off error when confidence is too low)
        volume[indices[:, 0], indices[:, 1], indices[:, 2], :] = values.half()
