import argparse
import pdb

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from object_keypoints.data import VoxelDataset
from object_keypoints.model_utils import load_model_and_dataset
from object_keypoints.object_model.models.keypointnet import KeypointNet
from torch.utils.data.dataloader import DataLoader
from object_keypoints.object_model.training import get_data_from_batch
from object_keypoints.metrics import iou_metric
import torch.nn.functional as F
from object_keypoints.data.transforms import point_cloud_to_voxel, rotate_point_cloud
import object_keypoints.visualize as vis

################################################################################################################
# Experiment: load given model, load mug points, encode base config, then rotate and evaluate to rotate points #
################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run reconstruction experiment.")
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("dataset_config", type=str, help="Dataset config file.")
    parser.add_argument("--mode", "-m", type=str, default="val", help="Which split to vis [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model_best.pt", help="Which model save file to use.")
    args = parser.parse_args()

    dataset: VoxelDataset
    model: KeypointNet
    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_config=args.dataset_config,
                                                               dataset_mode=args.mode, model_file=args.model_file)
    model.eval()

    # Grab a point cloud model from the dataset.
    base_point_cloud = dataset.point_clouds[0]

    # Encode the base configuration.
    base_voxel = point_cloud_to_voxel(base_point_cloud, 64).astype(np.float32)
    base_voxel = torch.from_numpy(base_voxel).unsqueeze(0).to(device)
    with torch.no_grad():
        keypoints, _, _ = model.forward(base_voxel, None, None)
    keypoints_np = keypoints[0].cpu().numpy()

    # Measure BCE/IoU.
    bce = []
    iou = []
    z_angles = np.arange(0.0, np.pi / 2.0, np.pi / 8.0)

    for z_angle in z_angles:
        # Generate GT voxels.
        rotated_point_cloud, _ = rotate_point_cloud(base_point_cloud, [0.0, 0.0, z_angle])
        rotated_voxel = point_cloud_to_voxel(rotated_point_cloud, 64).astype(np.float32)
        rotated_voxel = torch.from_numpy(rotated_voxel).unsqueeze(0).to(device)

        # Rotate keypoints to get new latent space.
        rotated_kp, _ = rotate_point_cloud(keypoints_np, [0.0, 0.0, z_angle])
        rotated_z = torch.from_numpy(rotated_kp.astype(np.float32)).reshape(1, model.k * 3).to(device)
        with torch.no_grad():
            _, voxel_recon = model.decode(rotated_z)

        # Evaluate quality of reconstruction.
        bce.append(F.binary_cross_entropy(voxel_recon, rotated_voxel, reduction='mean').item())
        iou.append(iou_metric(voxel_recon, rotated_voxel).item())

        voxel_recon = voxel_recon > 0.5

        # TODO: Visualize intermediate reconstructions/ground truth/keypoints.
        fig = plt.figure()
        grid_spec = GridSpec(1, 2, figure=fig)
        ax_1 = fig.add_subplot(grid_spec[0, 0], projection='3d')
        ax_2 = fig.add_subplot(grid_spec[0, 1], projection='3d')
        vis.visualize_voxels(rotated_voxel.cpu().numpy()[0], axes=ax_1, show=False)
        vis.visualize_voxels(voxel_recon.cpu().numpy()[0], axes=ax_2, show=False)
        plt.show()

    # TODO: Plot results.
