"""
We create a wrapper for faster rcnn



"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision


import sys
sys.path.append('/nethom/jbang36/eko')
from utils.dataset_utils import InferenceDataset

from tqdm import tqdm




class PytorchODConfig:
    model_name_library = {
        'faster_rcnn'          : torchvision.models.detection.fasterrcnn_resnet50_fpn,
        'faster_rcnn_mobilenet': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        'ssd'                  : torchvision.models.detection.ssd300_vgg16,
        'ssd_mobilenet'        : torchvision.models.detection.ssdlite320_mobilenet_v3_large
    }

    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    transforms = torchvision.transforms.Compose([
        # transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    throughput_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((320,320)),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])




class PytorchOD:
    def __init__(self, model_name, device=None):
        self.model = self.get_model(model_name)
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def get_model(self, model_name):
        if model_name == 'yolov5':
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        elif model_name not in PytorchODConfig.model_name_library:
            raise ValueError
        else:
            model = PytorchODConfig.model_name_library[model_name](pretrained = True, progress = False)
        return model


    def train(self, images, annotations):
        raise NotImplementedError

    def evaluate(self, images, labels):
        raise NotImplementedError


    def inference_dataloader(self, dataloader):
        self.model.eval()
        self.model.to(self.device)
        outputs = []

        with torch.no_grad():
            for data_batch in tqdm(dataloader):
                data_batch = data_batch.to(self.device)
                output = self.model(data_batch)

                ### we have to move things to the cpu...
                for out in output:
                    out['boxes'] = out['boxes'].to('cpu')
                    out['labels'] = out['labels'].to('cpu')
                    out['scores'] = out['scores'].to('cpu')
                    outputs.append( out )

        return outputs


    def inference(self, images, batch_size = 16, threshold = 0.2, throughput = False):
        if throughput:
            torch_dataset = InferenceDataset(images, transforms=PytorchODConfig.throughput_transforms)
        else:
            torch_dataset = InferenceDataset(images, transforms=PytorchODConfig.transforms)
        dataloader = DataLoader(torch_dataset, shuffle=False, batch_size=batch_size)
        outputs = []
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for data_batch in tqdm(dataloader):
                data_batch = data_batch.to(self.device)
                output = self.model(data_batch)

                ### we have to move things to the cpu...
                for out in output:
                    #### let's only output boxes that are above the threshold
                    relevant_indices = out['scores'].to('cpu') >= threshold
                    relevant_boxes = out['boxes'].to('cpu')[relevant_indices]
                    relevant_labels = out['labels'].to('cpu')[relevant_indices]
                    relevant_scores = out['scores'].to('cpu')[relevant_indices]
                    out['boxes'] = relevant_boxes
                    out['labels'] = relevant_labels
                    out['scores'] = relevant_scores
                    outputs.append( out )

        output_dict = {
            'categories': PytorchODConfig.classes,
            'annotations': outputs
        }
        return output_dict










