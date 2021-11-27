import unittest
import object_keypoints.object_model.config as config

import torch
from object_keypoints.object_model.models.keypointnet import KeypointNet


class ModelTester(unittest.TestCase):

    def setup_cuda_device(self):
        """
        Setup CUDA device for tests.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_default_model(self):
        """
        Build a default model to run.
        """
        default_model_config = {
            "model": {
                "type": "keypointnet",
            },
        }
        self.model: KeypointNet = config.get_model(default_model_config, device=self.device)

    def get_random_voxel_batch(self, batch_size=1):
        voxel_in = torch.rand([batch_size, 64, 64, 64], device=self.device)
        return voxel_in

    def setUp(self) -> None:
        """
        Setup environment for tests.
        """
        self.setup_cuda_device()
        self.setup_default_model()
