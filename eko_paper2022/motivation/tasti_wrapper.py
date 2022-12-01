"""
In this file, we implement a wrapper for tasti
"""

from benchmarks.stanford.tasti.tasti.index import Index
from benchmarks.stanford.tasti.tasti.config import IndexConfig
import torchvision
import cv2
from scipy.spatial import distance
import os
import torch
import pandas as pd
from collections import defaultdict
import numpy as np



'''
Defines our notion of 'closeness' as described in the paper for two labels for only one object type.
'''


class MotivationTasti(Index):
    def __init__(self, config, images):
        self.images = images
        super().__init__(config)


    def get_num_workers(self):
        return 1

    def get_cache_dir(self):
        return '/srv/data/jbang36/tasti_data/cache/tasti_triplet'

    def get_target_dnn(self):
        '''
        In this case, because we are running the target dnn offline, so we just return the identity.
        '''
        model = torch.nn.Identity()
        return model

    def get_embedding_dnn(self):
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Linear(512, 128)
        return model

    def get_pretrained_embedding_dnn(self):
        '''
        Note that the pretrained embedding dnn sometime differs from the embedding dnn.
        '''
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Identity()
        return model

    def get_target_dnn_dataset(self, train_or_test):
        ### just convert the loaded data into a dataset.
        print('Image size is ', self.config.image_size)
        dataset = InferenceDataset(self.images, image_size = self.config.image_size)
        return dataset

    def get_embedding_dnn_dataset(self, train_or_test):
        return self.get_target_dnn_dataset(train_or_test)

    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        root = '/srv/data/jbang36/video_data'
        ROOT_DATA = self.config.video_name
        if self.config.category == 'car':
            labels_fp = os.path.join(root, ROOT_DATA, 'tasti_labels.csv')
        else:
            labels_fp = os.path.join(root, ROOT_DATA, f'tasti_labels_{self.config.category}.csv')
        labels = LabelDataset(
            labels_fp=labels_fp,
            length=len(target_dnn_cache),
            category = self.config.category
        )
        return labels

    def is_close(self, label1, label2):
        objects = set()
        for obj in (label1 + label2):
            objects.add(obj.object_name)
        for current_obj in list(objects):
            label1_disjoint = [obj for obj in label1 if obj.object_name == current_obj]
            label2_disjoint = [obj for obj in label2 if obj.object_name == current_obj]
            is_redundant = night_street_is_close_helper(label1_disjoint, label2_disjoint)
            if not is_redundant:
                return False
        return True




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


class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, labels_fp, length, category):
        df = pd.read_csv(labels_fp)
        df = df[df['object_name'].isin([category])]
        frame_to_rows = defaultdict(list)
        for row in df.itertuples():
            frame_to_rows[row.frame].append(row)
        labels = []
        for frame_idx in range(length):
            labels.append(frame_to_rows[frame_idx])
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]






def night_street_is_close_helper(label1, label2):
    if len(label1) != len(label2):
        return False
    counter = 0
    for obj1 in label1:
        xavg1 = (obj1.xmin + obj1.xmax) / 2.0
        yavg1 = (obj1.ymin + obj1.ymax) / 2.0
        coord1 = [xavg1, yavg1]
        expected_counter = counter + 1
        for obj2 in label2:
            xavg2 = (obj2.xmin + obj2.xmax) / 2.0
            yavg2 = (obj2.ymin + obj2.ymax) / 2.0
            coord2 = [xavg2, yavg2]
            if distance.euclidean(coord1, coord2) < 100:
                counter += 1
                break
        if expected_counter != counter:
            break
    return len(label1) == counter


'''
Preprocessing function of a frame before it is passed to the Embedding DNN.
'''
def night_street_embedding_dnn_transform_fn(frame):
    xmin, xmax, ymin, ymax = 0, 960, 0, 540
    frame = frame[ymin:ymax, xmin:xmax]
    #frame = cv2.resize(frame, (224, 224))
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame

def night_street_target_dnn_transform_fn(frame):
    xmin, xmax, ymin, ymax = 0, 960, 0, 540
    frame = frame[ymin:ymax, xmin:xmax]
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame



class MotivationConfig(IndexConfig):
    def __init__(self, video_name, do_train, do_infer = False, category = 'car', image_size = None, nb_buckets = 7000):
        super().__init__()
        self.video_name = video_name
        self.do_mining = do_train
        self.do_training = do_train
        self.do_infer = do_infer
        self.do_bucketting = True
        self.image_size = image_size
        self.category = category

        self.batch_size = 16
        self.nb_train = 3000
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = nb_buckets
        self.nb_training_its = 12000
