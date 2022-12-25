import torch
import torchvision
from tqdm import tqdm
from udfs.yolov5.utils.general import non_max_suppression, scale_coords



class YoloWrapper:

    def __init__(self, model_name = 'yolov5s', device=None):
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        assert(model_name in ['yolov5s', 'yolov5m6', 'yolov5s6', 'yolov5n6', 'yolov5m'])
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)



    def inference(self, images, new_shape = None, old_shape = None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        dataset = InferenceDataset(images)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=1)
        self.model.to(device)
        self.model.eval()
        conf = 0.25  # NMS confidence threshold
        iou = 0.45
        classes = None
        organized_output = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                output = self.model(batch)

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

        classes = self.model.names

        organized_dict = {
            'categories' : classes,
            'annotations': organized_output
        }

        return organized_dict



def inference_transforms():
    ttransforms = torchvision.transforms.Compose([
        # transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((320, 320)),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return ttransforms



class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.transform = inference_transforms()
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
