import numpy as np
import torch
import torch.nn.functional as F
from model_tester import ModelTester


class KPModelTester(ModelTester):

    #########################################
    # Test Keypoint from Heatmap            #
    #########################################

    def test_heatmap_to_xyz_naive(self):
        batch_size = 1

        for i in range(5):
            heatmap = torch.zeros(
                [batch_size, self.model.heatmap_size, self.model.heatmap_size, self.model.heatmap_size],
                device=self.device)
            x, y, z = np.random.randint(self.model.heatmap_size), np.random.randint(
                self.model.heatmap_size), np.random.randint(self.model.heatmap_size)
            heatmap[0, x, y, z] = 1.0

            xyz = self.model.heatmap_to_xyz(heatmap)

            self.assertTrue(np.allclose(xyz[0, 0].item(), x))
            self.assertTrue(np.allclose(xyz[0, 1].item(), y))
            self.assertTrue(np.allclose(xyz[0, 2].item(), z))

        heatmap = torch.ones([batch_size, self.model.heatmap_size, self.model.heatmap_size, self.model.heatmap_size],
                             device=self.device)
        xyz = self.model.heatmap_to_xyz(heatmap)
        self.assertTrue(np.allclose(xyz[0, 0].item(), 15.5))
        self.assertTrue(np.allclose(xyz[0, 1].item(), 15.5))
        self.assertTrue(np.allclose(xyz[0, 2].item(), 15.5))

    def test_batch_heatmap_to_xyz(self):
        """
        Make sure batched result matches brute force result.
        """
        for i in range(5):
            heatmap_logits = torch.rand([1, 4, 32, 32, 32], device=self.device)
            heatmap = F.softmax(heatmap_logits.reshape(1, 4, 32 * 32 * 32), dim=2).reshape((4, 32, 32, 32))

            xyz = self.model.heatmap_to_xyz(heatmap)

            xyz_gt = np.zeros([4, 3])
            for kp_idx in range(4):
                for x in range(32):
                    for y in range(32):
                        for z in range(32):
                            xyz_gt[kp_idx, 0] += x * heatmap[kp_idx, x, y, z].item()
                            xyz_gt[kp_idx, 1] += y * heatmap[kp_idx, x, y, z].item()
                            xyz_gt[kp_idx, 2] += z * heatmap[kp_idx, x, y, z].item()
            self.assertTrue(np.allclose(xyz.cpu().numpy(), xyz_gt))

