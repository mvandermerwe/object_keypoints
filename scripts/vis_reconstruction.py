import argparse

import torch.utils.data.dataloader
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from object_keypoints.model_utils import load_model_and_dataset
from object_keypoints.object_model.training import get_data_from_batch
import object_keypoints.visualize as vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize reconstruction.")
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("--mode", "-m", type=str, default="val", help="Which split to vis [train, val, test].")
    args = parser.parse_args()

    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_mode=args.mode)
    model.eval()

    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=1, shuffle=True)

    fig = plt.figure()
    grid_spec = GridSpec(1, 2, figure=fig)
    ax_1 = fig.add_subplot(grid_spec[0, 0], projection='3d')
    ax_2 = fig.add_subplot(grid_spec[0, 1], projection='3d')

    for batch in dataloader:
        voxel_1, rot_1, scale_1, voxel_2, rot_2, scale_2 = get_data_from_batch(batch, device)

        with torch.no_grad():
            _, _, voxel_1_recon = model.forward(voxel_1, rot_1, scale_1)

        vis.visualize_voxels(voxel_1.cpu().numpy()[0], axes=ax_1, show=False)
        vis.visualize_voxels(voxel_1_recon.cpu().numpy()[0], axes=ax_2, show=False)
        plt.show()
