import os
from torchvision import transforms

import object_keypoints.data as data
import object_keypoints.object_model as object_model

method_dict = {
    'object_model': object_model,
}


def get_model(cfg, device=None):
    """
    Args:
    - cfg (dict): training config.
    - device (device): pytorch device.
    """
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device)
    return model


def get_trainer(model, optimizer, cfg, logger, vis_dir, device=None):
    """
    Return trainer instance.

    Args:
    - model (nn.Module): model which is used
    - optimizer (optimizer): pytorch optimizer
    - cfg (dict): training config
    - logger (tensorboardX.SummaryWriter): logger for tensorboard
    - vis_dir (str): vis directory
    - device (device): pytorch device
    """
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, logger, vis_dir, device)
    return trainer


def get_dataset(mode, cfg, batch_size=16):
    """
    Args:
    - mode (str): dataset mode [train, val, test].
    - cfg (dict): training config.
    """
    dataset_type = cfg['data']['dataset']

    transforms_ = get_transforms(cfg)

    if dataset_type == "voxel":
        dataset = data.VoxelDataset(cfg['data']['dataset_dir'], split=mode, batch_size=batch_size,
                                    transform=transforms_)
    else:
        raise Exception("Unknown dataset type: %s." % dataset_type)

    return dataset


def get_transforms(cfg):
    transforms_info = cfg['data'].get('transforms')
    if transforms_info is None:
        return None

    transform_list = []

    for tf_info in transforms_info:
        tf_type = tf_info["type"]
        raise Exception("Unknown transform type: %s" % tf_type)

    composed = transforms.Compose(transform_list)
    return composed
