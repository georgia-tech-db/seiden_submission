"""
Defines utility functions that is needed for the pipeline to run

"""


import numpy as np
import torch

from utils.logger import Logger
import json
import os
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import time
class Clock:

    def __init__(self):
        self.st = None

    def tic(self):
        self.st = time.perf_counter()

    def toc(self, objective=None):
        print(f"time taken for {objective}: {time.perf_counter() - self.st} seconds")


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def inference_transforms():
    ttransforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return ttransforms


class InferenceDataset(Dataset):
    def __init__(self, images, transforms = inference_transforms()):
        self.transform = transforms
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image


class ICTrainDataset(Dataset):
    def __init__(self, images, labels, class_list, transforms = inference_transforms()):
        self.images = images
        self.transforms = transforms
        self.labels = np.array(labels)
        self.classes = {}
        for i, class_name in enumerate(class_list):
            self.classes[class_name] = i

        assert(len(images) == len(labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, self.labels[idx]



class AverageMeter(object):
    """Computes and stores the average and current value
        AverageMeter is used for displaying things on tensorboard
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



