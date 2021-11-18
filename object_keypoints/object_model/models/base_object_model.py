import torch.nn as nn


class BaseObjectModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, voxel, rot, scale):
        raise NotImplementedError()

    def loss_forward(self, voxel_1, rot_1, scale_1, voxel_2, rot_2, scale_2):
        raise NotImplementedError()
