import pdb

import torch
from object_keypoints.object_model.models.base_object_model import BaseObjectModel
import torch.nn as nn


class CNNNet(BaseObjectModel):

    def __init__(self, input_size=64, z_dim=128, device=None):
        super().__init__()
        self.device = device

        # Representation params.
        self.input_size = input_size
        self.z_dim = z_dim

        # Build the network.
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(2, 2, 2), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(2, 2, 2), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(2, 2, 2), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=(2, 2, 2), padding="same"),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Flatten(),
            nn.Linear(4 * 4 * 4 * 512, self.z_dim)
        ).to(self.device)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.z_dim, 4 * 4 * 4 * 512),
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

    def forward(self, voxel, rot, scale):
        batch_size = voxel.shape[0]

        # CNN Net does not use rot or scale, since the latent space is unstructured.
        z = self.encoder(voxel.unsqueeze(1))
        z_decode = self.decoder_mlp(z)
        z_decode_rs = torch.reshape(z_decode, (batch_size, 512, 4, 4, 4))
        recon_logits = self.decoder_cnn(z_decode_rs).squeeze(1)

        recon = torch.sigmoid(recon_logits)

        return z, recon_logits, recon

    def loss_forward(self, voxel_1, rot_1, scale_1, voxel_2, rot_2, scale_2):
        loss_dict = {}

        v_1_recon_logits, v_1_recon = self.forward(voxel_1, rot_1, scale_1)
        # v_2_recon_logits, v_2_recon = self.forward(voxel_2, rot_2, scale_2)

        return v_1_recon_logits, v_1_recon, loss_dict
