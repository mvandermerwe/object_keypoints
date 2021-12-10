import argparse
import pdb
import time

import mmint_utils
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from object_keypoints.data import VoxelDataset
from object_keypoints.model_utils import load_model_and_dataset
from object_keypoints.object_model.models.keypointnet import KeypointNet
from torch.utils.data.dataloader import DataLoader
from object_keypoints.object_model.training import get_data_from_batch
from object_keypoints.metrics import iou_metric, chamfer_distance_metric
import torch.nn.functional as F
from object_keypoints.data.transforms import point_cloud_to_voxel, rotate_point_cloud
import object_keypoints.visualize as vis
import vedo

################################################################################################################
# Experiment: load given model, load mug points, encode base config, then rotate and evaluate to rotate points #
################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run reconstruction experiment.")
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("--mode", "-m", type=str, default="val", help="Which split to vis [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model_best.pt", help="Which model save file to use.")
    args = parser.parse_args()

    dataset: VoxelDataset
    model_cfg, model, dataset, device = load_model_and_dataset(args.config,
                                                               dataset_mode=args.mode, model_file=args.model_file)
    model.eval()

    # Grab a point cloud model from the dataset.
    base_point_cloud = dataset.point_clouds[0]

    # Rotate 180 degrees for easier handle viewing.
    base_point_cloud, _ = rotate_point_cloud(base_point_cloud, [0.0, 0.0, np.pi])

    # Test angles.
    z_angles = np.arange(0.0, np.pi / 2.0 + 0.1, (np.pi / 2.0))
    alphas = np.arange(0.0, 1.1, 1.0)

    plt = vedo.Plotter(N=1)

    for z_angle, alpha in zip(z_angles, alphas):
        # Generate input voxels.
        rotated_point_cloud, _ = rotate_point_cloud(base_point_cloud, [0.0, 0.0, z_angle])
        rotated_voxel = point_cloud_to_voxel(rotated_point_cloud, 64).astype(np.float32)
        rotated_voxel = torch.from_numpy(rotated_voxel).unsqueeze(0).to(device)

        with torch.no_grad():
            z, _, voxel_recon = model.forward(rotated_voxel, None, None)

        # Visualize GT voxel.
        gt_voxel_volume = vedo.Volume(rotated_voxel.cpu().numpy()[0]).legosurface(vmin=0.5)
        gt_voxel_volume.color('b')
        plt.show(gt_voxel_volume, at=0, interactive=True)
        plt.clear([gt_voxel_volume])

        # Visualize keypoints.
        if type(model) == KeypointNet:
            kps = z.cpu().numpy()[0].reshape(model.k, 3)
            vis.visualize_points(kps)

        # Visualize recon voxel.
        gt_voxel_volume = vedo.Volume(voxel_recon.cpu().numpy()[0]).legosurface(vmin=0.5)
        gt_voxel_volume.color('b')
        plt.show(gt_voxel_volume, at=0, interactive=True)
        plt.clear([gt_voxel_volume])
