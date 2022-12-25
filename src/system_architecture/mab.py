"""
This file is the newest version of EKO --
We will implement three different optimizations

1. I-frames
    Instead of using constant rate sampling, we will retrieve the I-frame information
    If # of bduget is greater than the number of I-frames, take entire i-frame set
    Select the rest using exploration / exploitation during query execution

    else:
        randomly select subset of i-frames

2. Label Propagation -- sigmoid function (aka. smoothing function)


3. MAB -- this replaces the alternating exploration and exploitation method we currently have.
"""

import os
import torch
import numpy as np
import torchvision
import sys
sys.path.append('/nethome/jbang36/seiden')

from sklearn.linear_model import LinearRegression
from benchmarks.stanford.tasti.tasti.index import Index
from benchmarks.stanford.tasti.tasti.config import IndexConfig
from benchmarks.stanford.tasti.tasti.seiden.data.data_loader import ImageDataset, LabelDataset
from tqdm import tqdm
import time
from src.motivation.tasti_wrapper import InferenceDataset
from src.system_architecture.alternate import EKO_alternate
from src.iframes.pyav_utils import PYAV_wrapper
from collections import OrderedDict
import random


class EKO_mab(EKO_alternate):
    def __init__(self, config, images, video_f, anchor_percentage = 0.8, c_param = 2, keep = False):
        self.images = images
        self.pyav = PYAV_wrapper(video_f) ## need up to .mp4
        self.c_param = c_param
        self.anchor_percentage = anchor_percentage
        self.keep = keep
        super().__init__(config, images)

    def __repr__(self):
        return 'EKO'

    def do_bucketting(self, percent_fpf=0.75):
        if self.config.do_bucketting:
            self.reps, self.topk_reps, self.topk_dists = self.calculate_rep_methodology()
            np.save(os.path.join(self.cache_dir, 'reps.npy'), self.reps)
            np.save(os.path.join(self.cache_dir, 'topk_reps.npy'), self.topk_reps)
            np.save(os.path.join(self.cache_dir, 'topk_dists.npy'), self.topk_dists)
        else:
            print(os.path.join(self.cache_dir, 'reps.npy'))
            self.reps = np.load(os.path.join(self.cache_dir, 'reps.npy'))
            self.topk_reps = np.load(os.path.join(self.cache_dir, 'topk_reps.npy'))
            self.topk_dists = np.load(os.path.join(self.cache_dir, 'topk_dists.npy'))


    def calculate_rep_methodology(self):
        rep_indices, dataset_length = self.get_reps()  ### we need to sort the reps
        top_reps = self.calculate_top_reps(dataset_length, rep_indices)
        top_dists = self.calculate_top_dists(dataset_length, rep_indices, top_reps)
        return rep_indices, top_reps, top_dists


    def get_reps(self):
        """
        We will select the i-frames and that's it
        :param dataset_length:
        :return:
        """
        dataset_length = len(self.images)
        iframe_indices = self.pyav.get_iframe_indices()
        n_reps = self.config.nb_buckets

        print('adfasdfasdfsaf', n_reps, self.anchor_percentage, int(n_reps * self.anchor_percentage))
        index_construction_reps = int(n_reps * self.anchor_percentage)

        print('total number of iframes: ', len(iframe_indices))
        print('total number of anchors selected in index construction: ', index_construction_reps)

        ### but we always have to include the first and last indices of the dataset
        if n_reps < 2:
            raise ValueError('Number of Reps too Low')

        rep_indices = [0, dataset_length - 1]
        ### If it ends up happening that total number of i-frames is less than index_construction_reps, then we just sample them in the latter phase.
        if index_construction_reps >= len(iframe_indices) + 2:
            rep_indices.extend(iframe_indices)
            rep_indices = list(set(rep_indices))
            rep_indices = sorted(rep_indices)
        else:
            subset = list(np.random.choice(iframe_indices, index_construction_reps, replace = False))
            rep_indices.extend(subset)
            rep_indices = list(set(rep_indices))
            rep_indices = rep_indices[:index_construction_reps]
            rep_indices = sorted(rep_indices)

        print('final number that has been selected: ', len(rep_indices))

        self.base_reps = rep_indices
        return rep_indices, dataset_length


    def build_additional_anchors(self, target_dnn_cache, scoring_func):
        """
        We will replace this function with the mab implementation...
        Steps are as follows:
        1. We create the clusters....and keep them fixed
        2. We will randomly draw from the cluster
        3. We will
        :param target_dnn_cache:
        :param scoring_func:
        :return:
        """
        n_reps = self.config.nb_buckets
        ### we need to keep track of cluster boundaries and rep_indices
        rep_indices = self.base_reps
        rep_indices = sorted(rep_indices)
        curr_len = len(rep_indices)
        topk_reps = self.topk_reps

        ### init distances / clusters
        ### for given distances, we iterate by selecting the argmax
        ### top reps, top dists are calculated in the same manner as before.
        length = rep_indices[-1] - rep_indices[0] + 1
        cluster_dict = self.init_label_distances(rep_indices, target_dnn_cache, scoring_func, length)

        for i in tqdm(range(n_reps - curr_len)):

            new_rep, cluster_key = self.select_rep(cluster_dict, rep_indices)
            rep_indices.append(new_rep)
            cluster_dict = self.update(cluster_dict, cluster_key, new_rep, target_dnn_cache, scoring_func)

        rep_indices = sorted(rep_indices)
        assert(len(rep_indices) == len(set(rep_indices)))

        dataset_length = len(topk_reps)
        top_reps = self.calculate_top_reps(dataset_length, rep_indices)
        top_dists = self.calculate_top_dists(dataset_length, rep_indices, top_reps)
        self.cluster_dict = cluster_dict
        self.reps = rep_indices
        self.topk_reps = top_reps
        self.topk_dists = top_dists

        np.save(os.path.join(self.cache_dir, 'reps.npy'), self.reps)
        np.save(os.path.join(self.cache_dir, 'topk_reps.npy'), self.topk_reps)
        np.save(os.path.join(self.cache_dir, 'topk_dists.npy'), self.topk_dists)

        if self.keep:
            self.base_reps = rep_indices



    def update(self, cluster_dict, cluster_key, rep_idx, target_dnn_cache, scoring_func):
        ### I mean we can just update everything here.....
        start_idx, end_idx = cluster_key
        cluster_dict[(start_idx, end_idx)]['members'].append(rep_idx)
        cluster_dict[(start_idx, end_idx)]['cache'].append(scoring_func(target_dnn_cache[rep_idx]))
        cluster_dict[(start_idx, end_idx)]['distance'] = np.var(cluster_dict[(start_idx, end_idx)]['cache'])

        return cluster_dict


    def select_rep(self, cluster_dict, rep_indices):
        ### compute the distances of each cluster and select based on the formula
        mab_values = []
        c_param = self.c_param
        for cluster_key in cluster_dict:
            reward = cluster_dict[cluster_key]['distance']
            members = cluster_dict[cluster_key]['members']

            mab_value = reward + c_param * np.sqrt(2 * np.log(len(rep_indices)) / len(members))
            mab_values.append(mab_value)

        assert(len(mab_values) == len(cluster_dict.keys()))
        #print(mab_values)

        selected_cluster = np.argmax(mab_values)
        #print('selected cluster: ', list(cluster_dict.keys())[selected_cluster], 'mab value is: ', mab_values[selected_cluster])

        start_idx, end_idx = list(cluster_dict.keys())[selected_cluster]

        choices = []
        for i in range(start_idx, end_idx+1):
            if i not in cluster_dict[(start_idx, end_idx)]['members']:
                choices.append(i)

        if len(choices) == 0:
            cluster_dict[(start_idx, end_idx)]['distance'] = 0
            rep_idx, (start_idx, end_idx) = self.select_rep(cluster_dict, rep_indices)
        else:
            rep_idx = np.random.choice(choices, 1)[0]

        #print('selected cluster: ', selected_cluster, 'selected ', rep_idx, 'max mab value: ', np.max(mab_values))
        return rep_idx, (start_idx, end_idx)


    def init_label_distances(self, cluster_boundaries, target_dnn_cache, scoring_func, length):
        #### we will generate the cluster_dictionary
        cluster_dict = OrderedDict()

        #### instead of making cluster dict based on cluster boundaries, we have to get the total number of frames and create boundaries based on that
        step_size = length // 100
        ####
        rep_idx = 0
        for i in range(0, length, step_size):
            start, end = i, min(i+step_size - 1, length - 1)
            #### find all rep_indices that fall within this boundary
            corresponding_reps = []
            while rep_idx < len(cluster_boundaries):
                if cluster_boundaries[rep_idx] <= end:
                    corresponding_reps.append(cluster_boundaries[rep_idx])
                else:
                    break
                rep_idx += 1
            cache = []
            for rep in corresponding_reps:
                cache.append(scoring_func(target_dnn_cache[rep]))

            cluster_dict[(start, end)] = {
                'members': corresponding_reps,
                'cache': cache,
                'distance': np.var(cache)
            }



        return cluster_dict



    def calculate_top_reps(self, dataset_length, rep_indices):
        """
        Choose representative frames based on systematic sampling

        :param dataset_length:
        :param rep_indices:
        :return:
        """
        top_reps = np.ndarray(shape=(dataset_length, 2), dtype=np.int32)

        ### do things before the first rep
        for i in range(len(rep_indices) - 1):
            start = rep_indices[i]
            end = rep_indices[i + 1]
            top_reps[start:end, 0] = start
            top_reps[start:end, 1] = end

        ### there could be some left over at the end....
        last_rep_indices = rep_indices[-1]
        top_reps[last_rep_indices:, 0] = rep_indices[-2]
        top_reps[last_rep_indices:, 1] = rep_indices[-1]
        return top_reps

    def calculate_top_dists(self, dataset_length, rep_indices, top_reps):
        """
        Calculate distance based on temporal distance between current frame and closest representative frame
        :param dataset_length:
        :param rep_indices:
        :param top_reps:
        :return:
        """
        ### now we calculate dists
        top_dists = np.ndarray(shape=(dataset_length, 2), dtype=np.int32)

        for i in range(dataset_length):
            #top_dists[i, 0] = abs(i - top_reps[i, 1])
            #top_dists[i, 1] = abs(i - top_reps[i, 0])
            top_dists[i, 0] = abs(i - top_reps[i, 0])
            top_dists[i, 1] = abs(i - top_reps[i, 1])

        return top_dists


    def _get_closest_reps(self, rep_indices, curr_idx):
        result = []

        for i, rep_idx in enumerate(rep_indices):
            if rep_idx - curr_idx >= 0:
                result.append(rep_indices[i-1])
                result.append(rep_indices[i])
                break
        return result


    ##### Unimportant functions ######

    def get_cache_dir(self):
        os.makedirs(self.config.cache_dir, exist_ok=True)
        return self.config.cache_dir


    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        root = '/srv/data/jbang36/video_data'
        ROOT_DATA = self.config.video_name
        category = self.config.category
        if category != 'car':
            labels_fp = os.path.join(root, ROOT_DATA, f'tasti_labels_{category}.csv')
        else:
            labels_fp = os.path.join(root, ROOT_DATA, 'tasti_labels.csv')
        labels = LabelDataset(
            labels_fp=labels_fp,
            length=len(target_dnn_cache),
            category = self.config.category
        )
        return labels


    def get_target_dnn_dataset(self, train_or_test):
        ### just convert the loaded data into a dataset.
        dataset = InferenceDataset(self.images)
        return dataset


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


    def get_embedding_dnn_dataset(self, train_or_test):
        dataset = InferenceDataset(self.images)
        return dataset


    def get_pretrained_embedding_dnn(self):
        '''
        Note that the pretrained embedding dnn sometime differs from the embedding dnn.
        '''
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        model.fc = torch.nn.Identity()
        return model


