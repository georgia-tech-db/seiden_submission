"""

Here we will write the training and execution script for svm



"""


import torch
import torchvision
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
import time




class SVMWrapper:

    def __init__(self):

        ### after creating the model, we need to curate the dataset.
        ## how will we transform this matrix to fit the image size?
        self.clf = SVC(gamma='auto', kernel = 'linear', probability = True)
        self.image_size = 30

    def inference_aggregate(self, images, image_size = None):
        ### we need to resize the images
        st = time.perf_counter()
        new_images = self._resize_images(images, image_size)
        et = time.perf_counter()
        print('total time taken for resizing the images: ', et - st, ' seconds')
        print(new_images.shape)
        return self.clf.predict(new_images)


    def inference_retrieval(self, images, image_size = None):

        new_images = self._resize_images(images, image_size)
        probs = self.clf.predict_proba(new_images)

        ## need to transform this to the predicate confidence value...
        ## we will always assume it's either 0 or 1 of the object it has been trained on

        final_output = []
        for i in range(len(probs)):
            final_output.append( 1 - probs[i][0] )

        final_output = np.array(final_output)

        return final_output



    def train(self, images, labels):

        new_images = self._resize_images(images)

        ### we need to sample the train images....
        train_images, train_labels = self.initialize_train_dataset(new_images, labels)

        ### we need to try out the svm model
        print('SVM training start....')
        self.clf.fit(train_images, train_labels)
        print('SVM training done!!')


    def _resize_images(self, images, image_size = None):
        width, height = images.shape[1], images.shape[2]
        if image_size is not None:
            width_division = width // image_size
            height_division = height // image_size
            new_images = images[:, ::width_division, ::height_division, :]
        else:
            width_division = width // self.image_size
            height_division = height // self.image_size
            new_images = images[:, ::width_division, ::height_division, :]
        new_images = new_images.reshape(len(new_images), -1)
        return new_images


    def initialize_train_dataset(self, images, labels):
        """
        We generate the training dataset for given images and labels
        :param images:
        :param labels:
        :return:
        """

        start_idx = 0
        end_idx = len(images)

        rand_arr = np.arange(start_idx, end_idx)
        random_idx = np.random.choice(rand_arr, 1000)
        random_idx = sorted(random_idx)
        train_images = images[random_idx]
        train_labels = self._get_train_labels(labels, random_idx)

        #### need to make the dataset and dataloader object
        return train_images, train_labels



    def _load_labels(self, images, filename):

        label_file = filename

        import pandas as pd
        from collections import defaultdict
        labels_fp = label_file
        length = len(images)

        df = pd.read_csv(labels_fp)
        df = df[df['object_name'].isin(['car'])]
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





