import time

import numpy as np
import os
import binvox_rw.binvox_helpers as binvox_rw
import transforms3d as tf3d
from matplotlib import pyplot as plt

from object_keypoints.data_generation.generate_voxels import center_pc, pc_to_voxels, voxels_to_pc
import object_keypoints.visualize as vis
from tqdm import trange


def downsample_visualize(points):
    pc_idcs = np.random.choice(points.shape[0], size=1024)
    mug_pc_ds = points[pc_idcs]
    vis.visualize_points(mug_pc_ds, show=True)


def generate_mug_rotations():
    """
    Generate single mug dataset with rotations.
    """
    mug_dir = '/mnt/wwn-0x5000c500c71941f0-part1/ShapeNetCore.v2/03797390/'
    mug_name = '4815b8a6406494662a96924bce6ef687'
    mug_file = os.path.join(mug_dir, mug_name)
    out_dir = 'out/dataset/mug_v2/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load voxels.
    vox_file = os.path.join(mug_file, 'models/model_normalized.solid.binvox')
    with open(vox_file, 'rb') as vf:
        voxels = binvox_rw.read_as_3d_array(vf)

    # Convert to point cloud and center.
    mug_pc_large = center_pc(voxels_to_pc(voxels.data))

    # Rotate around x to get mug "upright."
    upright_rot = tf3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0)
    mug_pc_large = (upright_rot @ mug_pc_large.T).T

    # Write point cloud.
    out_file = os.path.join(out_dir, mug_name + '.npy')
    np.save(out_file, mug_pc_large)


if __name__ == '__main__':
    generate_mug_rotations()
