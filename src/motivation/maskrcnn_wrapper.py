
import torch
import torchvision
from tqdm import tqdm



class MaskRCNNWrapper:

    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.to(device)
        self.model = model
        self.device = device
        self.categories = ["__background__", "person", "bicycle", "car",
                            "motorcycle", "airplane", "bus", "train",
                            "truck", "boat", "traffic light", "fire hydrant",
                            "N/A", "stop sign", "parking meter", "bench", "bird",
                            "cat", "dog", "horse", "sheep", "cow", "elephant",
                            "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
                            "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
                            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                            "skateboard", "surfboard", "tennis racket", "bottle", "N/A",
                            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                            "pizza", "donut", "cake", "chair", "couch", "potted plant",
                            "bed", "N/A", "dining table", "N/A","N/A",
                            "toilet","N/A","tv","laptop","mouse","remote","keyboard","cell phone","microwave",
                            "oven","toaster","sink","refrigerator","N/A","book","clock","vase","scissors",
                            "teddy bear","hair drier", "toothbrush"]


    def inference_features(self, images, batch_size = 8):
        dataset = InferenceDataset(images)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

        model_features = self.model.backbone

        outputs = []
        with torch.no_grad():
            model_features.eval()
            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                output = model_features(batch)
                outputs.append(output)


        return outputs


    def inference(self, images, batch_size = 8):
        dataset = InferenceDataset(images)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers=4)


        outputs = []
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                output = self.model(batch)
                new_outputs = []
                for o in output:
                    new_dict = {}
                    new_dict['boxes'] = o['boxes'].cpu()
                    new_dict['labels'] = o['labels'].cpu()
                    new_dict['scores'] = o['scores'].cpu()
                    new_outputs.append(new_dict)

                outputs.extend(new_outputs)


        final_output = {}
        final_output['categories'] = self.categories
        final_output['annotations'] = outputs
        return final_output


def inference_transforms():
    ttransforms = torchvision.transforms.Compose([
        # transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Resize((320, 320)),
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
