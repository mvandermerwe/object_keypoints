import numpy as np
import os
import binvox_rw.binvox_helpers as binvox_rw
import object_keypoints.visualize as vis
from tqdm import tqdm


def voxels_to_pc(voxels: np.ndarray):
    """
    Maps voxels to -1, 1 space in each dim.
    """
    voxel_shape = voxels.shape
    voxel_size = max(voxel_shape)

    # Convert to point cloud.
    voxels_sparse = np.argwhere(voxels.data)
    pc = ((voxels_sparse / voxel_size) * 2.0) - 1.0

    return pc


def downsample_voxel(voxels: np.ndarray, voxel_size: int):
    """
    Downsample to specified voxel size.
    """
    pc = voxels_to_pc(voxels)
    voxel_length = 2.0 / voxel_size

    voxel_locs = (pc / voxel_length).astype(int) + voxel_size // 2

    new_voxel = np.zeros([voxel_size, voxel_size, voxel_size], dtype=bool)
    for voxel_loc in voxel_locs:
        new_voxel[voxel_loc[0], voxel_loc[1], voxel_loc[2]] = True

    return new_voxel


def generate_downsampled_voxels():
    """
    Downsamples voxels and writes to file.
    """
    mug_dir = '/mnt/wwn-0x5000c500c71941f0-part1/ShapeNetCore.v2/03797390/'
    out_dir = 'out/dataset/mug/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mesh_folders = os.listdir(mug_dir)

    for mesh_folder in tqdm(mesh_folders):
        vox_file = os.path.join(mug_dir, mesh_folder, 'models/model_normalized.solid.binvox')
        with open(vox_file, 'rb') as vf:
            voxels = binvox_rw.read_as_3d_array(vf)

        # Downsample to 64x64x64.
        mug_voxels = downsample_voxel(voxels.data, 64)

        # To point cloud.
        mug_pc = voxels_to_pc(mug_voxels)

        # Write point cloud.
        out_file = os.path.join(out_dir, mesh_folder + '.npy')
        np.save(out_file, mug_pc)


if __name__ == '__main__':
    generate_downsampled_voxels()
