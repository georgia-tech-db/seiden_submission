"""
The model we will use is vgg16,
we will use this model to get some feature vectors
"""

import numpy as np
import torchvision
import torch
from tqdm import tqdm
import sys
sys.path.append('/nethome/jbang36/eko')

from utils.dataset_utils import InferenceDataset
from torch.utils.data import DataLoader

def max_pool_features(features, output_size = (300,300)):
    """
    As Input, we expect features to be a 4D output, we can do multiple images at once
    :param features:
    :param output_size:
    :return:
    """
    ### we input the features, then perform maxpooling, finally we resize to what we want
    max_elements, max_idxs = torch.max(features, dim=1)
    resize = torchvision.transforms.Resize(size=output_size)
    max_resized = resize(max_elements)

    #### if the first dimension is 1, then we need to squeeze
    if max_resized.shape[0] == 1:
        ### let's squeeze
        max_resized = max_resized.squeeze(0)
    return max_resized


def get_features(images, model_type = 'coco'):
    ### model_type == 'coco' -> object detection
    ### model_type == 'imagenet' -> image classification
    if model_type == 'coco':
        vgg16_vis = VGG16OD()
    elif model_type == 'ssd':
        vgg16_vis = SSDOD()
    else:
        vgg16_vis = VGG16Visualization()


    ## transform the image into batch
    ## run the batch
    ### unsqueeze the image
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    dataset = InferenceDataset(images)
    loader = DataLoader(dataset, batch_size = 1)

    all_results = []
    with torch.no_grad():
        for batch in tqdm(loader):
            results = vgg16_vis(batch)
            all_results.append(results)

    return all_results


class SSDOD(torch.nn.Module):
    def __init__(self):
        super(SSDOD, self).__init__()
        self.base = torchvision.models.detection.ssd300_vgg16(pretrained = True, progress = False)
        self.block1 = self.base.backbone.features
        self.block2 = self.base.backbone.extra[0]
        self.block3 = self.base.backbone.extra[1]
        self.block4 = self.base.backbone.extra[2]

    def forward(self, batch):
        block1 = self.block1(batch)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        return (block1, block2, block3, block4)

class VGG16OD(torch.nn.Module):

    def __init__(self):
        super(VGG16OD, self).__init__()
        self.base = torchvision.models.detection.ssd300_vgg16(pretrained = True, progress = False)
        self.block1 = self.base.backbone.features[:4]
        self.block2 = self.base.backbone.features[4:9]
        self.block3 = self.base.backbone.features[9:16]
        self.block4 = self.base.backbone.features[16:]
        self.block5 = self.base.backbone.extra[0][:7]

    def forward(self, batch):
        block1 = self.block1(batch)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        return (block1, block2, block3, block4, block5)



class VGG16Visualization(torch.nn.Module):

    def __init__(self):
        super(VGG16Visualization, self).__init__()
        self.base = torchvision.models.vgg16(pretrained = True)
        self.block1 = self.base.features[0:5]
        self.block2 = self.base.features[5:10]
        self.block3 = self.base.features[10:17]
        self.block4 = self.base.features[17:24]
        self.block5 = self.base.features[24:]


    def forward(self, batch):
        block1 = self.block1(batch)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        return (block1, block2, block3, block4, block5)


