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

    if dataset_type == "voxel":
        dataset = data.VoxelDataset(cfg['data']['dataset_dir'], mode, transforms_)
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
        if tf_type == "pc_to_voxel":
            tf = data.PointCloudToVoxel(tf_info['in_key'], tf_info['out_key'], tf_info['voxel_size'])
        elif tf_type == "tf_pc":
            tf = data.RandomTransformPointCloud(tf_info['in_key'], tf_info['out_key'], tf_info['rot_key'])
        elif tf_type == "scale_pc":
            tf = data.ScalePointCloud(tf_info['in_key'], tf_info['out_key'], tf_info['scale_key'])
        else:
            raise Exception("Unknown transform type: %s" % tf_type)
        transform_list.append(tf)

    composed = transforms.Compose(transform_list)
    return composed
