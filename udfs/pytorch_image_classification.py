"""

We use pretrained image classification models form pytorch for experiments


"""


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision


import sys
sys.path.append('/nethom/jbang36/eko')
from utils.dataset_utils import InferenceDataset
from udfs.image_net_classes import classes as IMAGENET_CLASSES

from tqdm import tqdm




class PytorchICConfig:
    model_name_library = {
        'vgg16': torchvision.models.vgg16,
        'mobilenet_v3_large': torchvision.models.mobilenet_v3_large,
        'efficientnet_b0': torchvision.models.efficientnet_b0,
        'efficientnet_b1' : torchvision.models.efficientnet_b1,
        'efficientnet_b2' : torchvision.models.efficientnet_b2,
        'efficientnet_b3' : torchvision.models.efficientnet_b3,
        'efficientnet_b4' : torchvision.models.efficientnet_b4,
        'efficientnet_b5' : torchvision.models.efficientnet_b5,
        'efficientnet_b6' : torchvision.models.efficientnet_b6,
        'efficientnet_b7' : torchvision.models.efficientnet_b7
    }

    classes = IMAGENET_CLASSES

    transforms = torchvision.transforms.Compose([
        # transforms.ToPILImage(),
        torchvision.transforms.ToTensor()
    ])




class PytorchIC:
    def __init__(self, model_name):
        self.model = self.get_model(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_model(self, model_name):
        if model_name not in PytorchICConfig.model_name_library:
            raise ValueError
        return PytorchICConfig.model_name_library[model_name](pretrained = True, progress = False)


    def train(self, images, annotations):
        raise NotImplementedError

    def evaluate(self, images, labels):
        raise NotImplementedError

    def inference(self, images, batch_size = 16):
        torch_dataset = InferenceDataset(images, transforms=PytorchICConfig.transforms)
        dataloader = DataLoader(torch_dataset, shuffle=False, batch_size=batch_size)
        outputs = []
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for data_batch in tqdm(dataloader):
                data_batch = data_batch.to(self.device)
                output = self.model(data_batch)
                ### as the output, you get a batchsize X n_classes
                ### we have to convert this back to cpu
                outputs.append(output.to('cpu'))


        return outputs







