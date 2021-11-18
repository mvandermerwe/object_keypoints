from object_keypoints.object_model.training import Trainer
from object_keypoints.object_model.models.keypointnet import KeypointNet


def get_model(cfg, device=None):
    model_cfg = cfg['model']
    model_type = model_cfg['type']

    if model_type == "keypointnet":
        model = KeypointNet(device=device)
    else:
        raise Exception("Unknown model type: %s" % model_type)

    return model


def get_trainer(model, optimizer, cfg, logger, vis_dir, device=None):
    """
    Return ReconNet trainer object.

    Args:
    - model (nn.Module): model which is used
    - optimizer (optimizer): pytorch optimizer
    - cfg (dict): training config
    - logger (tensorboardX.SummaryWriter): logger for tensorboard
    - vis_dir (str): vis directory
    - device (device): pytorch device
    """
    trainer = Trainer(model, optimizer, logger, vis_dir, loss_weights=cfg['training']['loss_weights'], device=device)
    return trainer
