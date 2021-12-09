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
    parser.add_argument("dataset_config", type=str, help="Dataset config file.")
    parser.add_argument("--out", default=None, type=str, help="Path to write video to.")
    parser.add_argument("--mode", "-m", type=str, default="val", help="Which split to vis [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model_best.pt", help="Which model save file to use.")
    parser.add_argument('--no_vis', dest='no_vis', action='store_true', help='Add flag to not visualize.')
    parser.set_defaults(no_vis=False)
    args = parser.parse_args()
    no_vis = args.no_vis

    dataset: VoxelDataset
    model: KeypointNet
    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_config=args.dataset_config,
                                                               dataset_mode=args.mode, model_file=args.model_file)
    model.eval()

    video_out_file = args.out

    # Grab a point cloud model from the dataset.
    base_point_cloud = dataset.point_clouds[0]

    # Rotate 180 degrees for easier handle viewing.
    base_point_cloud, _ = rotate_point_cloud(base_point_cloud, [0.0, 0.0, np.pi])

    # Rotate a further 90 degrees to get to the goal point cloud..
    goal_point_cloud, _ = rotate_point_cloud(base_point_cloud, [0.0, 0.0, np.pi/2.0])

    # Encode the base configuration.
    base_voxel = point_cloud_to_voxel(base_point_cloud, 64).astype(np.float32)
    base_voxel = torch.from_numpy(base_voxel).unsqueeze(0).to(device)
    with torch.no_grad():
        keypoints_base, _, _ = model.forward(base_voxel, None, None)
    keypoints_base_np = keypoints_base[0].cpu().numpy()

    # Encode the goal configuration.
    goal_voxel = point_cloud_to_voxel(goal_point_cloud, 64).astype(np.float32)
    goal_voxel = torch.from_numpy(goal_voxel).unsqueeze(0).to(device)
    with torch.no_grad():
        keypoints_goal, _, _ = model.forward(goal_voxel, None, None)
    keypoints_goal_np = keypoints_goal[0].cpu().numpy()

    # Measure BCE/IoU.
    bce = []
    iou = []
    chamfer = []

    # Test angles.
    z_angles = np.arange(0.0, np.pi / 2.0, (np.pi / 2.0) / 200)
    alphas = np.arange(0.0, 1.0, 1.0 / 200)

    if not no_vis:
        plt = vedo.Plotter(N=2)
        gt_voxel_volume = vedo.Volume(base_voxel.cpu().numpy()[0])
        gt_lego = gt_voxel_volume.legosurface(vmin=0.5)
        gt_lego.color('b')
        plt.show(gt_lego, at=0, interactive=True)
        plt.clear(gt_lego)

        if video_out_file is not None:
            video = vedo.Video(video_out_file)

    for z_angle, alpha in zip(z_angles, alphas):
        # Generate GT voxels.
        rotated_point_cloud, _ = rotate_point_cloud(base_point_cloud, [0.0, 0.0, z_angle])
        rotated_voxel = point_cloud_to_voxel(rotated_point_cloud, 64).astype(np.float32)
        rotated_voxel = torch.from_numpy(rotated_voxel).unsqueeze(0).to(device)

        # Rotate keypoints to get new latent space.
        rotated_base_kp, _ = rotate_point_cloud(keypoints_base_np, [0.0, 0.0, z_angle])
        rotated_goal_kp, _ = rotate_point_cloud(keypoints_goal_np, [0.0, 0.0, -(np.pi/2.0 - z_angle)])
        rotated_kp = ((1 - alpha) * rotated_base_kp) + (alpha * rotated_goal_kp)
        rotated_z = torch.from_numpy(rotated_kp.astype(np.float32)).reshape(1, model.k * 3).to(device)
        with torch.no_grad():
            _, voxel_recon = model.decode(rotated_z)

        # Evaluate quality of reconstruction.
        bce.append(F.binary_cross_entropy(voxel_recon, rotated_voxel, reduction='mean').item())
        iou.append(iou_metric(voxel_recon, rotated_voxel).item())
        chamfer.append(chamfer_distance_metric(voxel_recon, rotated_voxel).item())

        voxel_recon = voxel_recon > 0.5

        # fig = plt.figure()
        # grid_spec = GridSpec(1, 2, figure=fig)
        # ax_1 = fig.add_subplot(grid_spec[0, 0], projection='3d')
        # ax_2 = fig.add_subplot(grid_spec[0, 1], projection='3d')
        # vis.visualize_voxels(rotated_voxel.cpu().numpy()[0], axes=ax_1, show=False)
        # vis.visualize_voxels(voxel_recon.cpu().numpy()[0], axes=ax_2, show=False)
        # plt.show()
        if not no_vis:
            gt_voxel_volume = vedo.Volume(rotated_voxel.cpu().numpy()[0]).legosurface(vmin=0.5)
            gt_voxel_volume.color('b')

            vis_points = (rotated_kp.astype(np.float32) * 32) + 32
            vis_points_volume = vedo.Points(vis_points)

            pred_voxel_volume = vedo.Volume(voxel_recon.cpu().numpy()[0]).legosurface(vmin=0.5)
            pred_voxel_volume.color('b')

            plt.show(gt_voxel_volume, at=0, interactive=False)
            # plt.show(vis_points_volume, at=1, interactive=False)
            plt.show(pred_voxel_volume, at=1, interactive=False)
            if video_out_file is not None:
                video.addFrame()
            plt.clear([gt_voxel_volume, pred_voxel_volume])

    if video_out_file is not None:
        video.close()

    out_file = "out/results/kp_int_res.pkl.gzip"
    data = {
        'bce': bce,
        'iou': iou,
        'chamfer': chamfer,
    }
    mmint_utils.save_gzip_pickle(data, out_file)
