from object_keypoints.object_model.training import Trainer


def get_model(cfg, device=None):
    model_cfg = cfg['model']

    return None


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
