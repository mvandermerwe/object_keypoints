import torch


def iou_metric(prediction, label, threshold=0.5, reduce=True):
    """
    Calculate the iou metric for the given (batched) predictions/labels.

    Args:
        prediction: predicted voxels [B, V, V, V] where V is voxel size
        label: gt voxels
        threshold: threshold to decide occupancy
        reduce: whether to take mean over batch
    """
    binary_prediction = prediction > threshold
    binary_label = label > 0.0

    union = torch.logical_or(binary_prediction, binary_label)
    intersection = torch.logical_and(binary_prediction, binary_label)

    iou = (intersection.sum(dim=(1, 2, 3))) / (union.sum(dim=(1, 2, 3)))

    if reduce:
        return iou.mean()
    else:
        return iou
