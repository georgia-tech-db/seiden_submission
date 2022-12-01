
import torch
import torchvision
from tqdm import tqdm



class ResnetWrapper:

    def inference(self, images, image_size = None):
        print('inference image size is ', image_size)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        dataset = InferenceDataset(images, image_size = image_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=1)
        model = torchvision.models.resnet18()
        model.to(device)
        model.eval()


        outputs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                output = model(batch)
                output = output.cpu()
                for o in output:
                    outputs.append(o)
        return outputs







def inference_transforms(image_size):
    if image_size is not None:
        ttransforms = torchvision.transforms.Compose([
            # transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        ttransforms = torchvision.transforms.Compose([
            # transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return ttransforms

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, images, image_size = None):
        self.transform = inference_transforms(image_size)
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
