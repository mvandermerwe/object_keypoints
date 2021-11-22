import numpy as np
import torch
from model_tester import ModelTester
import object_keypoints.visualize as vis
from object_keypoints.data_generation.generate_voxels import voxels_to_pc


class ShapeTester(ModelTester):

    def test_encoder_out_shape(self):
        voxel_in = self.get_random_batch()

        encoder_out = self.model.encoder(voxel_in.unsqueeze(1))
        print(encoder_out.shape)

    def test_decoder_out_shape(self):
        batch_size = 1
        heatmap_in = torch.rand([batch_size, self.model.k, 32, 32, 32], device=self.device)

        decoder_out = self.model.decoder(heatmap_in)
        print(decoder_out.shape)

    def test_gaussian_heatmap(self):
        batch_size = 1
        voxel_in = self.get_random_batch(batch_size)

        xyz, g_heatmap = self.model.encode(voxel_in)

        for b_idx in range(batch_size):
            g_heatmap_batch = g_heatmap[b_idx].cpu().detach().numpy()

            pc_gaussian = voxels_to_pc(g_heatmap_batch[0])
            points = np.zeros([32 * 32 * 32, 4])
            points[:, :3] = pc_gaussian
            points[:, 3] = g_heatmap_batch[0].flatten()
            vis.visualize_points(points)
