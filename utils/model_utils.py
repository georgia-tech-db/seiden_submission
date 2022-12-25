"""


In this file, we save the torch models


"""

import torch
import os


def save_model(model, directory):
    ### we need to package this
    ### make sure it ends with .pt
    os.makedirs(os.path.dirname(directory), exist_ok=True)
    assert(directory[-3:] == '.pt')
    model_dict = {'model_state_dict': model.state_dict()}

    torch.save(model_dict, directory)


def load_model(model_frame, directory):
    model_dict = torch.load(directory)

    model_state_dict = model_dict['model_state_dict']
    model_frame.load_state_dict(model_state_dict)
    return model_frame
