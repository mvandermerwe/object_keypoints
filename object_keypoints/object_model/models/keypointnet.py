import pdb

import torch
from object_keypoints.object_model.models.base_object_model import BaseObjectModel
from torch import nn
import torch.nn.functional as F


def separation_loss(keypoints):
    """

    Args:
        keypoints (torch.tensor): keypoint representation BxKx3
    Returns:
        separation_loss (torch.tensor): separation loss
    """
    num_points = keypoints.shape[1]
    xyz0 = keypoints[:, :, None, :].expand(-1, -1, num_points, -1)
    xyz1 = keypoints[:, None, :, :].expand(-1, num_points, -1, -1)
    sq_dist = torch.sum((xyz0 - xyz1) ** 2, 3)
    loss = 1 / (1000 * sq_dist + 1)
    return torch.mean(loss, [1, 2])


class KeypointNet(BaseObjectModel):

    def __init__(self, input_size=64, heatmap_size=32, k=16, device=None):
        super().__init__()
        self.device = device

        # Representation params.
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.epsilon = 1e-8
        self.k = k

        # Setup voxel location buffers.
        r_x = torch.arange(0, self.heatmap_size, 1.0)
        r_y = torch.arange(0, self.heatmap_size, 1.0)
        r_z = torch.arange(0, self.heatmap_size, 1.0)
        ranx, rany, ranz = torch.meshgrid(r_x, r_y, r_z)
        self.register_buffer("ranx", torch.FloatTensor(ranx).clone().to(self.device))
        self.register_buffer("rany", torch.FloatTensor(rany).clone().to(self.device))
        self.register_buffer("ranz", torch.FloatTensor(ranz).clone().to(self.device))

        # Build the network!
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, (5, 5, 5), padding="same"),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.ReLU(),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding='same'),
            nn.ReLU(),

            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding='same'),
            nn.ReLU(),

            nn.Conv3d(512, self.k, kernel_size=(3, 3, 3), padding='same')
        ).to(self.device)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.k * 3, 4 * 4 * 4 * 512),
            nn.ReLU(),
        ).to(self.device)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose3d(512, 256, (2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),

            nn.ConvTranspose3d(256, 128, (2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, (2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),

            nn.Conv3d(32, 1, (2, 2, 2), padding='same'),
        ).to(self.device)

    def heatmap_to_xyz(self, heatmap):
        heatmap = heatmap / (torch.sum(heatmap, dim=(1, 2, 3), keepdim=True) + self.epsilon)
        sx = torch.sum(heatmap * self.ranx[None], dim=(1, 2, 3))
        sy = torch.sum(heatmap * self.rany[None], dim=(1, 2, 3))
        sz = torch.sum(heatmap * self.ranz[None], dim=(1, 2, 3))
        xyz_normalized = torch.stack([sx, sy, sz], 1)
        return xyz_normalized

    def xyz_to_heatmap(self, xyz):
        batch_size = xyz.shape[0]
        xyz = xyz.reshape(batch_size * self.k, -1)
        grid = torch.stack([self.ranx, self.rany, self.ranz], 3)[None]

        # We use a fixed sigma here.
        sigma = 3.0  # TODO: What is a good sigma value?
        var = sigma ** 2.0

        # Calculate gaussian heatmap.
        g_heatmap = torch.exp(-torch.sum((grid - xyz[:, None, None, None, :3]) ** 2, 4) / (2 * var))
        g_heatmap = g_heatmap.reshape(batch_size, self.k, self.heatmap_size, self.heatmap_size, self.heatmap_size)

        return g_heatmap

    def encode(self, voxel):
        batch_size = voxel.shape[0]

        # Get keypoint heatmaps.
        z = self.encoder(voxel.unsqueeze(1))
        heatmap = F.softmax(z.reshape(batch_size, self.k, self.heatmap_size ** 3), dim=2)
        heatmap = heatmap.reshape(batch_size * self.k, self.heatmap_size, self.heatmap_size, self.heatmap_size)

        # Get xyz points from heatmap.
        xyz = self.heatmap_to_xyz(heatmap)
        xyz = xyz.reshape(batch_size, self.k, 3)

        # Convert points to range of [-1, 1].
        xyz = ((xyz / self.heatmap_size) * 2.0) - 1.0

        return xyz

    def decode(self, z):
        batch_size = z.shape[0]

        z_decode = self.decoder_mlp(z)
        z_decode_rs = torch.reshape(z_decode, (batch_size, 512, 4, 4, 4))
        recon_logits = self.decoder_cnn(z_decode_rs).squeeze(1)
        recon = torch.sigmoid(recon_logits)

        return recon_logits, recon

    def forward(self, voxel, rot, scale):
        batch_size = voxel.shape[0]

        # Encode
        xyz = self.encode(voxel)
        z = torch.reshape(xyz, (batch_size, self.k * 3))

        # Decode
        recon_logits, recon = self.decode(z)

        return xyz, recon_logits, recon

    def normalize_keypoints(self, xyz: torch.Tensor, rotation: torch.Tensor, scale: torch.Tensor):
        """
        We assume keypoints are in correct framing and scale, etc.
        """
        # Undo the rotation.
        rot_undo = torch.transpose(rotation, 1, 2)
        norm_xyz = (rot_undo @ xyz.transpose(1, 2)).transpose(1, 2)

        # Undo the scaling.
        norm_xyz[:, :, 0] /= scale[:, None, 0]
        norm_xyz[:, :, 1] /= scale[:, None, 1]
        norm_xyz[:, :, 2] /= scale[:, None, 2]
        return norm_xyz

    def loss_forward(self, voxel_1, rot_1, scale_1, voxel_2, rot_2, scale_2):
        loss_dict = {}

        # Run model forward.
        xyz_1, v_1_recon_logits, v_1_recon = self.forward(voxel_1, rot_1, scale_1)
        xyz_2, v_2_recon_logits, v_2_recon = self.forward(voxel_2, rot_2, scale_2)

        # Separation loss.
        sep_1 = separation_loss(xyz_1)
        sep_2 = separation_loss(xyz_2)
        sep_loss = torch.cat([sep_1, sep_2], dim=0).mean()
        loss_dict['separation'] = sep_loss

        # Keypoint consistency loss.
        norm_xyz_1 = self.normalize_keypoints(xyz_1, rot_1, scale_1)
        norm_xyz_2 = self.normalize_keypoints(xyz_2, rot_2, scale_2)
        consistency_loss = F.mse_loss(norm_xyz_1, norm_xyz_2)
        loss_dict['consistency'] = consistency_loss

        return v_1_recon_logits, v_1_recon, v_2_recon_logits, v_2_recon, loss_dict
