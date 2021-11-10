import torch
import os
import sys
import torch.nn as nn


def get_num_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model_dict, model_file):
    """
    Load model dictionary elements from given model file.
    Returns dictionary with other elements in file (e.g., training
    iter info and val loss).

    Args:
    - model_dict (dict): dict of model key and pytorch objects to load.
    - model_file (str): path to saved model to load.
    """
    if os.path.exists(model_file):
        print('Loading checkpoint from local file...')
        state_dict = torch.load(model_file, map_location='cpu')
        for k, v in model_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print('Warning: Could not find %s in checkpoint' % k)
        load_dict = {k: v for k, v in state_dict.items()
                     if k not in model_dict}
    else:
        print("Couldn't find model file at ", model_file)
        load_dict = dict()

    return load_dict
