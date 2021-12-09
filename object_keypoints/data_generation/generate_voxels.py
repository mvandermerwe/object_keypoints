import numpy as np
import os
import binvox_rw.binvox_helpers as binvox_rw
import torch
import trimesh
from tqdm import tqdm


def voxels_to_pc_torch(voxels: torch.Tensor):
    """
    Maps voxels to -1, 1 pc space in each dim.
    """
    voxel_shape = voxels.shape
    voxel_size = max(voxel_shape)

    voxels_sparse = torch.nonzero(voxels)
    pc = ((voxels_sparse / voxel_size) * 2.0) - 1.0

    return pc


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


def center_pc(pc: np.ndarray):
    """
    Center point cloud at the origin.
    """
    x_min = pc[:, 0].min()
    x_max = pc[:, 0].max()
    x_center = (x_max + x_min) / 2.0
    y_min = pc[:, 1].min()
    y_max = pc[:, 1].max()
    y_center = (y_max + y_min) / 2.0
    z_min = pc[:, 2].min()
    z_max = pc[:, 2].max()
    z_center = (z_max + z_min) / 2.0
    pc[:, 0] -= x_center
    pc[:, 1] -= y_center
    pc[:, 2] -= z_center
    return pc


def pc_to_voxels(pc: np.ndarray, voxel_size: int):
    """
    Downsample to specified voxel size.
    """
    voxel_length = 2.0 / voxel_size
    voxel_locs = (pc / voxel_length).astype(int) + (voxel_size // 2)
    voxel_locs = np.clip(voxel_locs, a_min=0, a_max=63)

    new_voxel = np.zeros([voxel_size, voxel_size, voxel_size], dtype=bool)
    for voxel_loc in voxel_locs:
        new_voxel[voxel_loc[0], voxel_loc[1], voxel_loc[2]] = True

    return new_voxel


def vis_meshes():
    """
    Downsamples voxels and writes to file.
    """
    mug_dir = '/mnt/wwn-0x5000c500c71941f0-part1/ShapeNetCore.v2/03797390/'
    mesh_folders = os.listdir(mug_dir)

    for mesh_folder in tqdm(mesh_folders):
        mesh_file = os.path.join(mug_dir, mesh_folder, 'models/model_normalized.obj')
        mesh = trimesh.load(mesh_file)
        mesh.show()


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

        mug_pc_large = center_pc(voxels_to_pc(voxels.data))
        # pc_idcs = np.random.choice(mug_pc_large.shape[0], size=256)
        # mug_pc_ds = mug_pc_large[pc_idcs]
        # vis.visualize_points(mug_pc_ds, show=True)

        # Downsample to 64x64x64.
        mug_voxels = pc_to_voxels(mug_pc_large, 64)

        # To point cloud.
        mug_pc = voxels_to_pc(mug_voxels)
        # vis.visualize_points(mug_pc, show=True)
        #
        # vis.visualize_voxels(mug_voxels)

        # Write point cloud.
        out_file = os.path.join(out_dir, mesh_folder + '.npy')
        np.save(out_file, mug_pc)


if __name__ == '__main__':
    generate_downsampled_voxels()
    # vis_meshes()
