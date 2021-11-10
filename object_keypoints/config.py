import os
from torchvision import transforms

import object_keypoints.data as data

method_dict = {

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


def get_dataset(mode, cfg):
    """
    Args:
    - mode (str): dataset mode [train, val, test].
    - cfg (dict): training config.
    """
    dataset_type = cfg['data']['dataset']

    transforms_ = get_transforms(cfg)
    dataset = None

    return dataset


def get_transforms(cfg):
    transforms_info = cfg['data'].get('transforms')
    if transforms_info is None:
        return None

    transform_list = []
    composed = transforms.Compose(transform_list)
    return composed
