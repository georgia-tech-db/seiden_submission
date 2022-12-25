
import os
import cv2
import swag
import json
import tasti
import torch
import pandas as pd
import numpy as np
import torchvision
from scipy.spatial import distance
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm.autonotebook import tqdm
from blazeit.aggregation.samplers import ControlCovariateSampler
from PIL import Image



class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, video_fp, list_of_idxs=[], transform_fn=lambda x: x):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.video_fp = video_fp
        self.list_of_idxs = []
        self.transform_fn = transform_fn
        base_directory = os.path.join(self.video_fp, 'images')
        self.length = len(os.listdir(base_directory))
        self.current_idx = 0
        self.init()

    def init(self):
        if len(self.list_of_idxs) == 0:
            self.frames = None
        else:
            self.frames = []
            for idx in tqdm(self.list_of_idxs, desc="Video"):
                frame = self.read(idx)
                self.frames.append(frame)

    def transform(self, frame):
        frame = self.transform_fn(frame)
        return frame


    def read(self, idx):
        idx_str = str(idx).zfill(9)
        img_name = os.path.join(self.video_fp, 'images', idx_str + '.jpg')
        image = Image.open(img_name)
        image = np.array(image)
        image = self.transform(image)
        idx += 1
        return image

    def __len__(self):
        return self.length if len(self.list_of_idxs) == 0 else len(self.list_of_idxs)

    def __getitem__(self, idx):
        if len(self.list_of_idxs) == 0:
            frame = self.read(idx)
        else:
            frame = self.frames[idx]
        return frame




class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_fp, list_of_idxs=[], transform_fn=None):
        self.video_fp = video_fp
        self.list_of_idxs = []
        self.transform_fn = transform_fn
        self.cap = swag.VideoCapture(self.video_fp)
        self.video_metadata = json.load(open(self.video_fp + '.json', 'r'))
        self.cum_frames = np.array(self.video_metadata['cum_frames'])
        self.cum_frames = np.insert(self.cum_frames, 0, 0)
        self.length = self.cum_frames[-1]
        self.current_idx = 0
        #### we need to measure the timings...we use averagemeter for this
        self.init()

    def init(self):
        if len(self.list_of_idxs) == 0:
            self.frames = None
        else:
            self.frames = []
            for idx in tqdm(self.list_of_idxs, desc="Video"):
                self.seek(idx)
                frame = self.read()
                self.frames.append(frame)

    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform_fn is not None:
            frame = self.transform_fn(frame)
        return frame

    def seek(self, idx):
        if self.current_idx != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
            self.current_idx = idx

    def read_only(self):
        _, frame = self.cap.read()
        self.current_idx += 1
        return frame

    def read(self):
        _, frame = self.cap.read()
        frame = self.transform(frame)
        self.current_idx += 1
        return frame

    def __len__(self):
        return self.length if len(self.list_of_idxs) == 0 else len(self.list_of_idxs)

    def __getitem__(self, idx):
        if len(self.list_of_idxs) == 0:
            self.seek(idx)
            frame = self.read()
        else:
            frame = self.frames[idx]
        return frame




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
