import numpy as np
import os
import torch.utils.data


class VoxelDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir: str, transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform

        # Preload untransformed point clouds.
        self.point_clouds = []
        for pc_file in os.listdir(self.dataset_dir):
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
