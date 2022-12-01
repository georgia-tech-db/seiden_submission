"""
We will create a model that can be used for training and inference
The query we want to work with is straightforward -- let's just hard code it for now

"""


import torch
import torchvision
from tqdm import tqdm
import numpy as np



class ResnetWrapper:

    def __init__(self):
        self.model = torchvision.models.resnet18(pretrained = True) ### when we are training / inference, let's only adjust the latter layers
        for param in self.model.parameters():
            param.requires_grad = False
        fc = torch.nn.Linear(512, 7) ## 0, 1, 2, 3, 4, 5, 6
        self.model.fc = fc

        ### after creating the model, we need to curate the dataset.


    def train(self, images, labels):
        ### we expect the labels to be in tasti_labels.csv format. This is what's needed for blazeit, supg anyways....
        ### we perform iterative random sampling.... sample 1 percent of the entire video and use it for training.
        train_dataloader = self.initialize_train_dataset(images, labels)
        ### perform the training step...
        n_epochs = 5
        train_loss = []
        train_acc = []
        total_step = len(train_dataloader)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.model.to(device)


        for epoch in range(1, n_epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0
            print(f'Epoch {epoch}\n')
            for batch_idx, (data_, target_) in enumerate(train_dataloader):
                data_, target_ = data_.to(device), target_.to(device)
                optimizer.zero_grad()

                outputs = self.model(data_)
                loss = criterion(outputs, target_) ## that's the thing, we need convert the labels to something similar
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)
                _, tar = torch.max(target_, dim = 1)
                correct += torch.sum(pred == tar).item()
                total += tar.size(0)
                if (batch_idx) % 20 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
            train_acc.append(100 * correct / total)
            train_loss.append(running_loss / total_step)
            print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')

        #### since we update the model parameters, there is nothing to return... maybe the train_loss, train_acc?



    def initialize_train_dataset(self, images, labels):
        """
        We generate the training dataset for given images and labels
        random selection creates networks that are bad..
        :param images:
        :param labels:
        :return:
        """

        start_idx = 0
        end_idx = len(images)

        rand_arr = np.arange(start_idx, end_idx)
        random_idx = np.arange(0, len(images), 100)
        #random_idx = np.random.choice(rand_arr, 1000)
        random_idx = sorted(random_idx)
        train_images = images[random_idx]
        train_labels = self._get_train_labels(labels, random_idx)

        #### need to make the dataset and dataloader object
        dataset = TrainDataset(train_images, train_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle = True)


        return dataloader


    def _load_labels(self, images, filename, category = 'car'):

        label_file = filename

        import pandas as pd
        from collections import defaultdict
        labels_fp = label_file
        length = len(images)

        df = pd.read_csv(labels_fp)
        df = df[df['object_name'].isin([category])]
        frame_to_rows = defaultdict(list)
        for row in df.itertuples():
            frame_to_rows[row.frame].append(row)
        labels = []
        for frame_idx in range(length):
            labels.append(len(frame_to_rows[frame_idx]))
        return labels


    def _get_train_labels(self, labels, idxs):
        train_labels = [labels[idx] for idx in idxs]

        return train_labels

    def inference_aggregate(self, images, image_size = None, save_directory = None):
        print('inference image size is ', image_size)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        dataset = InferenceDataset(images, image_size = image_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=1)

        self.model.to(device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data_ in tqdm(dataloader):
                data_ = data_.to(device)

                outputs = self.model(data_)
                _, pred = torch.max(outputs, dim=1)
                pred = pred.to('cpu')

                predictions.append(pred)

        print(predictions[0].shape)
        predictions = np.hstack(predictions)

        print(predictions.shape)
        #### let's just save this
        #if save_directory

        return predictions

    def inference_retrieval(self, images, image_size = None):
        print('inference image size is ', image_size)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        dataset = InferenceDataset(images, image_size=image_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=1)

        self.model.to(device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data_ in tqdm(dataloader):
                data_ = data_.to(device)

                outputs = self.model(data_)
                outputs = torch.nn.functional.softmax(outputs, dim = 1)
                ### now they are probabilities

                """
                tmp = []
                for output in outputs:
                    tmp.append( 1 - output[0] )
                """
                outputs = outputs.to('cpu')
                ## now we need to append the probability
                for output in outputs:
                    predictions.append(1 - output[0])

        predictions = np.array(predictions)

        return predictions



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
    def __init__(self, images, image_size = 224):
        self.transform = inference_transforms(image_size)
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, images:np.array, labels:np.array):
        self.transform = inference_transforms(image_size = 224)
        self.images = images
        ### for the labels, we will assume there are 7 different type of objects...
        self.labels = labels
        self.max_value = 7


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if label >= self.max_value: label = self.max_value - 1
        new_label = torch.zeros(self.max_value, dtype = torch.float32)
        new_label[label] = 1

        if self.transform:
            image = self.transform(image)

        return image, new_label
