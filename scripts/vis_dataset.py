import mmint_utils
import numpy as np
import object_keypoints.config as config
import argparse
import object_keypoints.visualize as vis
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visualize dataset.")
    parser.add_argument("config", type=str, help="Configuration file.")
    args = parser.parse_args()

    cfg = mmint_utils.load_cfg(args.config)

    dataset = config.get_dataset("train", cfg)

    fig = plt.figure()
    d_idx = np.random.randint(len(dataset))
    data = dataset[d_idx]
    grid_spec = GridSpec(1, 2, figure=fig)

    ax_1 = fig.add_subplot(grid_spec[0, 0], projection='3d')
    vis.visualize_voxels(data['voxel'], axes=ax_1, show=False)

    ax_2 = fig.add_subplot(grid_spec[0, 1], projection='3d')
    vis.visualize_voxels(data['tf_voxel'], axes=ax_2, show=False)
    plt.show()
