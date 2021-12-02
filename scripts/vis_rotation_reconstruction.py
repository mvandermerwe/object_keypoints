import argparse
import pdb

import numpy as np
import torch.utils.data.dataloader
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from object_keypoints.data_generation.generate_voxels import voxels_to_pc
from object_keypoints.model_utils import load_model_and_dataset
from object_keypoints.object_model.training import get_data_from_batch
import torch.nn.functional as F
import object_keypoints.visualize as vis
import transforms3d as tf3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize reconstruction.")
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("--mode", "-m", type=str, default="val", help="Which split to vis [train, val, test].")
    args = parser.parse_args()

    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_mode=args.mode)
    model.eval()

    colors = np.random.rand(model.k, 3)

    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        voxel_1, rot_1, scale_1, voxel_2, rot_2, scale_2 = get_data_from_batch(batch, device)

        with torch.no_grad():
            keypoints = model.encode(voxel_1)

        for z_rot in [0.0, np.pi / 4.0, np.pi / 2.0]:
            # Build rotation matrix.
            rotation = tf3d.euler.euler2mat(0.0, 0.0, z_rot)
            rot_torch = torch.from_numpy(rotation).float().to(device)

            # Rotate keypoints.
            kp_new = (rot_torch @ keypoints.transpose(1, 2)).transpose(1, 2)
            z_new = kp_new.reshape(1, model.k * 3)
            voxel_logits, voxel_recon = model.decode(z_new)

            best_threshold = 0.5
            voxel_1_recon = voxel_recon > best_threshold

            fig = plt.figure()
            grid_spec = GridSpec(1, 3, figure=fig)
            ax_1 = fig.add_subplot(grid_spec[0, 0], projection='3d')
            ax_2 = fig.add_subplot(grid_spec[0, 1], projection='3d')
            ax_3 = fig.add_subplot(grid_spec[0, 2], projection='3d')

            gt_pc = voxels_to_pc(voxel_1.cpu().numpy()[0])
            gt_pc = gt_pc[np.random.choice(gt_pc.shape[0], size=1024)]
            vis.visualize_points(gt_pc, axes=ax_1, show=False)

            pred_pc = voxels_to_pc(voxel_1_recon.cpu().numpy()[0])
            pred_pc = pred_pc[np.random.choice(pred_pc.shape[0], size=1024)]
            vis.visualize_points(pred_pc, axes=ax_2, show=False)

            kp_new = kp_new.cpu().numpy()[0]
            ax_3.scatter(kp_new[:, 0], kp_new[:, 1], kp_new[:, 2], c=colors)
            ax_3.set_xlim3d(left=-1.0, right=1.0)
            ax_3.set_ylim3d(bottom=-1.0, top=1.0)
            ax_3.set_zlim3d(bottom=-1.0, top=1.0)

            plt.show()
