import argparse
import pdb

import numpy as np
import torch.utils.data.dataloader
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from object_keypoints.model_utils import load_model_and_dataset
from object_keypoints.object_model.training import get_data_from_batch
import torch.nn.functional as F
import object_keypoints.visualize as vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize reconstruction.")
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("--mode", "-m", type=str, default="val", help="Which split to vis [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model_best.pt", help="Which model save file to use.")
    args = parser.parse_args()

    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_mode=args.mode,
                                                               model_file=args.model_file)
    model.eval()

    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        voxel_1, rot_1, scale_1, voxel_2, rot_2, scale_2 = get_data_from_batch(batch, device)

        with torch.no_grad():
            _, _, voxel_1_recon = model.forward(voxel_1, rot_1, scale_1)

        err_min = np.inf
        best_threshold = 0.5  # -1.0
        # for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #     err = F.binary_cross_entropy((voxel_1_recon > threshold).float(), voxel_1)
        #     if err < err_min:
        #         best_threshold = threshold
        #         err_min = err

        print("Using threshold: %f" % best_threshold)
        voxel_1_recon = voxel_1_recon > best_threshold

        fig = plt.figure()
        grid_spec = GridSpec(1, 2, figure=fig)
        ax_1 = fig.add_subplot(grid_spec[0, 0], projection='3d')
        ax_2 = fig.add_subplot(grid_spec[0, 1], projection='3d')
        vis.visualize_voxels(voxel_1.cpu().numpy()[0], axes=ax_1, show=False)
        vis.visualize_voxels(voxel_1_recon.cpu().numpy()[0], axes=ax_2, show=False)
        plt.show()
