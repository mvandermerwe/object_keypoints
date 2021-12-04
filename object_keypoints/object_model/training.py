import pdb
from collections import defaultdict

import torch
from object_keypoints.training import BaseTrainer
from object_keypoints.object_model.models.base_object_model import BaseObjectModel
import torch.nn.functional as F
from tqdm import tqdm


def get_data_from_batch(batch, device=None):
    voxel_1 = batch['voxel_1'].to(device)
    rot_1 = batch['rot_1'].to(device)
    scale_1 = batch['scale_1'].to(device)

    voxel_2 = batch['voxel_2'].to(device)
    rot_2 = batch['rot_2'].to(device)
    scale_2 = batch['scale_2'].to(device)

    return voxel_1, rot_1, scale_1, voxel_2, rot_2, scale_2


class Trainer(BaseTrainer):

    def __init__(self, model: BaseObjectModel, optimizer, logger, vis_dir, loss_weights, device=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.device = device
        self.vis_dir = vis_dir
        self.loss_weights = loss_weights

    def validation(self, val_loader, it):
        val_dict = defaultdict(lambda: 0.0)

        for val_batch in tqdm(val_loader):
            eval_loss_dict = self.eval_step(val_batch, it)
            val_dict['loss'] += eval_loss_dict['loss']

        return {
            "val_loss": val_dict['loss'] / len(val_loader)
        }

    def eval_step(self, data, it):
        """
        Perform evaluation step.

        Args:
        - data (dict): data dictionary
        """
        self.model.eval()

        with torch.no_grad():
            loss_dict = self.compute_loss(data, it)

        return loss_dict

    def visualize(self, data):
        """
        Visualize the predicted RGB/Depth images.
        """
        # TODO: Visualize!
        pass

    def train_step(self, data, it):
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data, it)

        for k, v in loss_dict.items():
            if k != 'loss':
                self.logger.add_scalar(k, v, it)

        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, data, it):
        voxel_1, rot_1, scale_1, voxel_2, rot_2, scale_2 = get_data_from_batch(data, device=self.device)

        voxel_1_recon_logits, voxel_1_recon, voxel_2_recon_logits, voxel_2_recon, loss_dict = \
            self.model.loss_forward(voxel_1, rot_1, scale_1, voxel_2, rot_2, scale_2)

        # Calculate reconstruction losses.
        recon_loss_1 = F.binary_cross_entropy_with_logits(voxel_1_recon_logits, voxel_1, reduction='none').mean(
            dim=[1, 2, 3])
        recon_loss_2 = F.binary_cross_entropy_with_logits(voxel_2_recon_logits, voxel_2, reduction='none').mean(
            dim=[1, 2, 3])
        recon_loss = torch.cat([recon_loss_1, recon_loss_2], dim=0).mean()
        loss_dict['reconstruction'] = recon_loss

        # Calculate final weighted loss.
        loss = 0.0
        for loss_key, loss_val in loss_dict.items():
            loss += self.loss_weights[loss_key] * loss_val
        loss_dict['loss'] = loss

        return loss_dict
