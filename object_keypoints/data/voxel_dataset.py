import numpy as np
import os
import torch.utils.data


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

    def __init__(self, dataset_dir: str, split: str = 'train', transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Load split info (i.e., which points in dataset are for training).
        splits = load_split(os.path.join(self.dataset_dir, 'splits', split + '.txt'))

        # Preload untransformed point clouds.
        self.point_clouds = []
        for pc_file in splits:
            self.point_clouds.append(np.load(os.path.join(self.dataset_dir, pc_file)))

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        data = {}

        # Load untransformed point cloud.
        point_cloud = self.point_clouds[idx]
        data['point_cloud'] = point_cloud

        if self.transform is not None:
            data = self.transform(data)

        return data