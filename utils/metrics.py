import numpy as np
import torch


def evaluation(est, target, mask=None):

    mse = mse_fn(est, target, mask)
    mad = mad_fn(est, target, mask)
    iou = iou_fn(est, target, mask)
    acc = acc_fn(est, target, mask)

    return {"mse": mse, "mad": mad, "iou": iou, "acc": acc}


def mse_fn(est, target, mask=None):

    if mask is not None:
        grid = mask * np.power(est - target, 2)
        grid = grid.astype(
            np.float32
        )  # required to not get inf values since we use float16 here as input grids
        metric = np.sum(grid) / np.sum(mask)
    else:
        metric = np.mean(np.power(est - target, 2))

    return metric


def mad_fn(est, target, mask=None):

    if mask is not None:
        grid = mask * np.abs(est - target)
        grid = grid.astype(
            np.float32
        )  # required to not get inf values since we use float16 here as input grids
        metric = np.sum(grid) / np.sum(mask)
    else:
        metric = np.mean(np.abs(est - target))

    return metric


def iou_fn(est, target, mask=None):

    est = est.astype(
        np.float32
    )  # required to not get inf values since we use float16 here as input grids
    target = target.astype(np.float32)
    if mask is not None:
        tp = (est < 0) & (target < 0) & (mask > 0)
        fp = (est < 0) & (target >= 0) & (mask > 0)
        fn = (est >= 0) & (target < 0) & (mask > 0)
    else:
        tp = (est < 0) & (target < 0)
        fp = (est < 0) & (target >= 0)
        fn = (est >= 0) & (target < 0)

    intersection = tp.sum()
    union = tp.sum() + fp.sum() + fn.sum()

    del tp, fp, fn
    metric = intersection / union
    return metric


def acc_fn(est, target, mask=None):

    est = est.astype(
        np.float32
    )  # required to not get inf values since we use float16 here as input grids
    target = target.astype(np.float32)
    if mask is not None:
        tp = (est < 0) & (target < 0) & (mask > 0)
        tn = (est >= 0) & (target >= 0) & (mask > 0)
    else:
        tp = (est < 0) & (target < 0)
        tn = (est >= 0) & (target >= 0)

    acc = (tp.sum() + tn.sum()) / mask.sum()

    del tp, tn
    metric = acc
    return metric
