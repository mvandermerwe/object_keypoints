import numpy as np
import os
import torch.utils.data

from object_keypoints.data.transforms import random_scale_point_cloud, random_z_rotate_point_cloud, point_cloud_to_voxel


def load_split(split_file):
    split = []
    with open(split_file, 'r') as f:
        while True:
            next_split = f.readline()
            if next_split == "":
                break
            split.append(next_split.replace("\n", ""))
    return split


class VoxelDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir: str, voxel_size: int = 64, split: str = 'train', dataset_len: int = None,
                 transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.voxel_size = voxel_size
        self.dataset_len = dataset_len

        # Load split info (i.e., which points in dataset are for training).
        splits = load_split(os.path.join(self.dataset_dir, 'splits', split + '.txt'))

        # Preload untransformed point clouds.
        self.point_clouds = []
        for pc_file in splits:
            self.point_clouds.append(np.load(os.path.join(self.dataset_dir, pc_file)))

    def __len__(self):
        if self.dataset_len is None:
            return len(self.point_clouds)
        else:
            return self.dataset_len

    def __getitem__(self, idx):
        if self.dataset_len is not None:
            idx = idx % len(self.point_clouds)

        # Load untransformed point cloud.
        point_cloud = self.point_clouds[idx]

        # Apply two separate set of transforms to get paired data.
        s_pc_1, scale_1 = random_scale_point_cloud(point_cloud)
        r_pc_1, rot_1 = random_z_rotate_point_cloud(s_pc_1)
        vox_1 = point_cloud_to_voxel(r_pc_1, voxel_size=self.voxel_size)

        s_pc_2, scale_2 = random_scale_point_cloud(point_cloud)
        r_pc_2, rot_2 = random_z_rotate_point_cloud(s_pc_2)
        vox_2 = point_cloud_to_voxel(r_pc_2, voxel_size=self.voxel_size)

        # Fill in data dict.
        data = {
            'voxel_1': vox_1.astype(np.float32),
            'rot_1': rot_1.astype(np.float32),
            'scale_1': scale_1.astype(np.float32),
            'voxel_2': vox_2.astype(np.float32),
            'rot_2': rot_2.astype(np.float32),
            'scale_2': scale_2.astype(np.float32),
        }

        if self.transform is not None:
            data = self.transform(data)
        return data
