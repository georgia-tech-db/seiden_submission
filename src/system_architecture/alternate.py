"""
Advancement from simple algorithm.
We will adjust exploration and exploitation when selecting the anchors

@jaehobang
"""

"""
Version of EKO that only uses
temporal distance and actual labels -- we will be using the scoring function from the query to build additional anchors

Index Construction:
1. Select Temporal Anchors (50%)
2. Calculate Temporal Distances

Query Processing:
1. Get labels for each anchor location
2. eliminate max discrepancy locs for each iteration

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



class EKO_alternate(Index):
    def __init__(self, config, images, initial_anchor = 0.8):
        self.images = images
        self.initial_anchor = initial_anchor
        super().__init__(config)


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
        How to choose the representative frames
        50% of the rep frames are chosen using even temporal spacing
        50% of the rep frames are chosen using a different strat
        use alpha * normalized_content_diff + beta * time diff.

        :param dataset_length:
        :return:
        """
        dataset_length = len(self.images)

        ### we need one at the start and end
        n_reps = self.config.nb_buckets
        print(self.initial_anchor, n_reps)
        initial_anchor_count = int(self.initial_anchor * n_reps)
        skip_rate = dataset_length // (initial_anchor_count - 1)

        rep_indices = np.arange(dataset_length, dtype=np.int32)[::skip_rate]
        tmp_buffer = int(n_reps * 0.1)
        print(f'rep indices stats', n_reps, len(rep_indices))
        rep_indices = rep_indices.tolist()
        ### append only if not in list..
        if 0 not in rep_indices: rep_indices.append(0)
        if dataset_length - 1 not in rep_indices: rep_indices.append(dataset_length - 1)

        print(len(rep_indices), n_reps)
        ## make sure all the rep indices are different and total length matches what we requested
        self.base_reps = rep_indices
        return rep_indices, dataset_length



    def build_additional_anchors(self, target_dnn_cache, scoring_func):
        #### we assume self.reps is already built
        #### we will have to modify self.reps, self.topk_reps, self.topk_dists
        n_reps = self.config.nb_buckets
        rep_indices = self.base_reps
        rep_indices = sorted(rep_indices)
        curr_len = len(rep_indices)
        topk_reps = self.topk_reps
        dists, temporal_dists = self.init_label_distances(rep_indices, target_dnn_cache, scoring_func)

        self.debug_dists = dists
        self.debug_temporal_dists = temporal_dists
        self.initial_count = curr_len
        self.additional_count = n_reps - curr_len

        for i in range(n_reps - curr_len):
            ### get the location with max dist
            if i % 2 == 0: ### alternate between exploration and exploitation
                chosen_section_start = np.argmax(dists)
            else:
                chosen_section_start = np.argmax(temporal_dists)

            left, right = rep_indices[chosen_section_start], rep_indices[chosen_section_start+1]
            middle = (left + right) // 2
            rep_indices, dists, temporal_dists = self.update(rep_indices, dists, temporal_dists,
                                                             left, middle, right, scoring_func, target_dnn_cache)



        dataset_length = len(topk_reps)
        top_reps = self.calculate_top_reps(dataset_length, rep_indices)
        top_dists = self.calculate_top_dists(dataset_length, rep_indices, top_reps)
        self.reps = rep_indices
        self.topk_reps = top_reps
        self.topk_dists = top_dists

        np.save(os.path.join(self.cache_dir, 'reps.npy'), self.reps)
        np.save(os.path.join(self.cache_dir, 'topk_reps.npy'), self.topk_reps)
        np.save(os.path.join(self.cache_dir, 'topk_dists.npy'), self.topk_dists)



    def update(self, rep_indices, dists, temporal_dists, left, middle, right, scoring_func, target_dnn_cache):
        ## put middle in rep_indices
        loc = rep_indices.index(right)
        rep_indices.insert(loc, middle)

        ## update dists and temporal dists
        temporal_dists.pop(loc - 1)
        temporal_dists.insert(loc - 1, middle - left)
        temporal_dists.insert(loc, right - middle)
        max_temporal_dist = max(temporal_dists)
        base_digits = len(str(max_temporal_dist))  ## calculate how many digits

        #### get the label distance for middle - left, right - middle
        left_label = scoring_func(target_dnn_cache[left])
        middle_label = scoring_func(target_dnn_cache[middle])
        right_label = scoring_func(target_dnn_cache[right])

        dists.pop(loc - 1)
        dists.insert(loc - 1, abs(middle_label - left_label) * (10**base_digits) + temporal_dists[loc - 1])
        dists.insert(loc, abs(right_label - middle_label) * (10**base_digits) + temporal_dists[loc])

        return rep_indices, dists, temporal_dists



    def init_label_distances(self, rep_indices, target_dnn_cache, scoring_func):
        relevant_cache = [scoring_func(target_dnn_cache[rep_index]) for rep_index in rep_indices]
        ### get max temporal gap
        temporal_dist = []
        for i in range(len(rep_indices) - 1):
            temporal_dist.append( rep_indices[i+1] - rep_indices[i] )
        max_temporal_dist = max(temporal_dist)
        base_digits = len(str(max_temporal_dist)) ## calculate how many digits


        ### compute the label distances but also incorporate temporal dist as well
        distances = []
        for i in range(len(relevant_cache) - 1):
            distances.append(abs(relevant_cache[i+1] - relevant_cache[i]) * (10**base_digits) + temporal_dist[i])

        return distances, temporal_dist


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

