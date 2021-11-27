import numpy as np
import torch
import torch.nn.functional as F
from model_tester import ModelTester
import transforms3d as tf3d


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

    #########################################
    # Test Keypoint normalization           #
    #########################################

    def test_keypoint_normalization(self):
        for i in range(5):
            xyz = np.random.random([4, self.model.k, 3])
            xyz_tf = np.zeros_like(xyz)
            rots = []
            scales = []

            for b_idx in range(4):
                keypoints = xyz[b_idx]

                scale = 0.75 + (np.random.random(3) * 0.25)
                scales.append(scale)

                scale_keypoints = scale * keypoints

                random_euler = -np.pi + (2.0 * np.pi * np.random.random(3))
                rot = tf3d.euler.euler2mat(random_euler[0], random_euler[1], random_euler[2])
                rots.append(rot)

                rot_keypoints = (rot @ scale_keypoints.T).T

                xyz_tf[b_idx] = rot_keypoints
            rots = np.array(rots)
            scales = np.array(scales)

            # Move everything to torch.
            xyz = torch.from_numpy(xyz).to(self.device)
            xyz_tf = torch.from_numpy(xyz_tf).to(self.device)
            rots = torch.from_numpy(rots).to(self.device)
            scales = torch.from_numpy(scales).to(self.device)

            # Normalize keypoints.
            xyz_norm = self.model.normalize_keypoints(xyz_tf, rots, scales)

            self.assertTrue(torch.allclose(xyz_norm, xyz))
