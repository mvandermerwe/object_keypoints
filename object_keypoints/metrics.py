import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from object_keypoints.data_generation.generate_voxels import voxels_to_pc_torch
from chamferdist import ChamferDistance
import object_keypoints.visualize as vis


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


def chamfer_distance_metric(prediction, label, threshold=0.5, n=1000, reduce=True):
    """
    Determine chamfer distance between voxels. First convert each
    voxel grid to a point cloud, then subsample, then find chamfer distance.

    Args:
        prediction: predicted voxels [B, V, V, V] where V is voxel size
        label: gt voxels
        threshold: threshold to decide occupancy
        n: num points to subsample from each voxel
        reduce: whether to take mean over batch
    """
    batch_size = prediction.shape[0]
    chamfer_dist = ChamferDistance()

    # Evaluate chamfer.
    chamfer_example = []
    for in_batch_idx in range(batch_size):
        pred_pc = voxels_to_pc_torch(prediction[in_batch_idx] > threshold)
        example_ids = np.arange(pred_pc.shape[0], dtype=int)
        np.random.shuffle(example_ids)
        pred_pc = pred_pc[example_ids[:n]]

        gt_pc = voxels_to_pc_torch(label[in_batch_idx] > 0.5)
        example_ids = np.arange(gt_pc.shape[0], dtype=int)
        np.random.shuffle(example_ids)
        gt_pc = gt_pc[example_ids[:n]]

        # fig = plt.figure()
        # grid_spec = GridSpec(1, 2, figure=fig)
        # ax_1 = fig.add_subplot(grid_spec[0, 0], projection='3d')
        # ax_2 = fig.add_subplot(grid_spec[0, 1], projection='3d')
        # vis.visualize_points(gt_pc.cpu().numpy(), axes=ax_1, show=False)
        # vis.visualize_points(pred_pc.cpu().numpy(), axes=ax_2, show=False)
        # plt.show()

        chamfer_example.append(
            chamfer_dist(pred_pc.unsqueeze(0), gt_pc.unsqueeze(0), bidirectional=True))

    if reduce:
        return np.mean(chamfer_example)
    else:
        return chamfer_example
