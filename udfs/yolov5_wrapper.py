"""

Wrapper around YOLOv5 for inference

"""
import torch
import numpy as np
from tqdm import tqdm
#### we need to access some source files from yolo
from udfs.yolov5.utils.general import non_max_suppression, scale_coords

class YOLOv5Wrapper:
    def __init__(self, model_name = 'yolov5s', device=None):
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        assert(model_name in ['yolov5s', 'yolov5m6', 'yolov5s6', 'yolov5n6', 'yolov5m'])
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)


    def train(self, images, annotations):
        raise NotImplementedError

    def evaluate(self, images, labels):
        raise NotImplementedError

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def inference_dali(self, dali_dataloader, new_shape = None, old_shape = None):
        ### shape: [height, width] 
        classes = self.model.names
        self.model.eval()
        self.model.to(self.device)

        organized_output = []

        conf = 0.25  # NMS confidence threshold
        iou = 0.45
        classes = None
        names = self.model.names
        with torch.no_grad():
            for batch in tqdm(dali_dataloader):
                image_batch = batch[0]['images'] ## we expect
                output = self.model(image_batch)

                ### start of postprocessing
                y = non_max_suppression(output, conf_thres=conf, iou_thres=iou, classes=classes)  # NMS
                if new_shape is not None and old_shape is not None:
                    for i in range(len(y)):
                        scale_coords(new_shape, y[i], old_shape)

                ### now y is a list so we must do it one by one
                for prediction in y:
                    out = {}
                    prediction = prediction.to('cpu')
                    boxes = prediction[:, :4]
                    scores = prediction[:, 4]
                    labels = prediction[:, 5].int()
                    out['boxes'] = boxes
                    out['labels'] = labels
                    out['scores'] = scores
                    organized_output.append(out)

            organized_dict = {
                'categories' : classes,
                'annotations': organized_output
            }

        return organized_dict


    def inference(self, images, batch_size = 8, new_shape = None, old_shape = None):
        classes = self.model.names
        self.model.eval()
        self.model.to(self.device)
        print('evaluating on device', self.device)
        conf = 0.25  # NMS confidence threshold
        iou = 0.45

        ### give maybe 100 images at a time??
        organized_output = []
        #### i would rather create an iter
        with torch.no_grad():
            for image_batch in tqdm(self.batch(images, batch_size)):

                output = self.model(image_batch)

                ### start of postprocessing
                y = non_max_suppression(output, conf_thres=conf, iou_thres=iou, classes=classes)  # NMS
                if new_shape is not None and old_shape is not None:
                    for i in range(len(y)):
                        scale_coords(new_shape, y[i], old_shape)

                ### now y is a list so we must do it one by one
                for prediction in y:
                    out = {}
                    prediction = prediction.to('cpu')
                    boxes = prediction[:, :4]
                    scores = prediction[:, 4]
                    labels = prediction[:, 5].int()
                    out['boxes'] = boxes
                    out['labels'] = labels
                    out['scores'] = scores
                    organized_output.append(out)

        organized_dict = {
            'categories' : classes,
            'annotations': organized_output
        }

        return organized_dict



    def inference_slow(self, images, batch_size = 8, throughput = False):

        classes = self.model.names
        self.model.eval()
        self.model.to(self.device)
        print('evaluating on device', self.device)

        ### give maybe 100 images at a time??
        organized_output = []
        #### i would rather create an iter
        with torch.no_grad():
            for image_batch in tqdm(self.batch(images, batch_size)):
                image_list = [image_batch[ii] for ii in range(len(image_batch))]
                output_list = self.model(image_list, size = 960)
                for prediction in output_list.pred:
                    out = {}
                    prediction = prediction.to('cpu')
                    boxes = prediction[:,:4]
                    scores = prediction[:,4]
                    labels = prediction[:,5].int()
                    out['boxes'] = boxes
                    out['labels'] = labels
                    out['scores'] = scores
                    organized_output.append( out )

        organized_dict = {
            'categories': classes,
            'annotations': organized_output
        }

        return organized_dict


