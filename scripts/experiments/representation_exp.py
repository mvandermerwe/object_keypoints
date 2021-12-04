import argparse
import numpy as np
import torch
from object_keypoints.model_utils import load_model_and_dataset
from torch.utils.data.dataloader import DataLoader
from object_keypoints.object_model.training import get_data_from_batch
from object_keypoints.metrics import iou_metric
import torch.nn.functional as F

#######################################################################
# Experiment: load given model, load test split, evaluate avg IoU/BCE #
#######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run reconstruction experiment.")
    parser.add_argument("config", type=str, help="Model/data config file.")
    parser.add_argument("dataset_config", type=str, help="Dataset config file.")
    parser.add_argument("--mode", "-m", type=str, default="val", help="Which split to vis [train, val, test].")
    parser.add_argument("--model_file", "-f", type=str, default="model_best.pt", help="Which model save file to use.")
    args = parser.parse_args()

    model_cfg, model, dataset, device = load_model_and_dataset(args.config, dataset_config=args.dataset_config,
                                                               dataset_mode=args.mode, model_file=args.model_file)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Evaluate BCE and IoU.
    bce = []
    iou = []

    for batch in dataloader:
        voxel_1, rot_1, scale_1, _, _, _ = get_data_from_batch(batch, device)

        with torch.no_grad():
            _, _, voxel_1_recon = model.forward(voxel_1, rot_1, scale_1)

        # Evaluate iou.
        iou_example = iou_metric(voxel_1_recon, voxel_1, threshold=0.5, reduce=False)

        # Evaluate bce.
        bce_example = F.binary_cross_entropy(voxel_1_recon, voxel_1, reduction='none').mean(dim=[1, 2, 3])

        bce.append(bce_example)
        iou.append(iou_example)
    bce = torch.cat(bce).cpu().numpy()
    iou = torch.cat(iou).cpu().numpy()

    # Get statistics from result.
    bce_avg = np.mean(bce).item()
    bce_std = np.std(bce).item()
    iou_avg = np.mean(iou).item()
    iou_std = np.std(iou).item()

    print("BCE: %f (%f), IoU: %f (%f)" % (bce_avg, bce_std, iou_avg, iou_std))
