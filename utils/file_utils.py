import json
import os
import torch
import numpy as np


####################### json ############################

def save_json(file, directory):
    #### I assume the acceptable inputs will be list, json
    #### TODO: confirm it works on list of jsons

    ### need to create the directory if it doesn't exist
    os.makedirs(os.path.dirname(directory), exist_ok=True)

    with open(directory, 'w') as ff:
        json.dump(file, ff)


def load_json(directory):
    print(f"opening: {directory}")
    with open(directory, "r") as file:
        data = json.load(file)
    return data

########################## torch ###########################

def save_torch(file, filepath):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(file, filepath)
    print(f"saved to {filepath}")


def load_torch(filepath):

    return torch.load(filepath)


########################### numpy ###########################


def save_numpy(file, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, file)

def load_numpy(file):
    return np.load(file)
