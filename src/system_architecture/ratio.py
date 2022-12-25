"""
Built on top of alternate, just modifying the exploration and exploitation parameter
"""


import os
import torch
import numpy as np
import torchvision
import sys
sys.path.append('/nethome/jbang36/seiden')
import random
from sklearn.linear_model import LinearRegression
from benchmarks.stanford.tasti.tasti.index import Index
from benchmarks.stanford.tasti.tasti.config import IndexConfig

from benchmarks.stanford.tasti.tasti.seiden.data.data_loader import ImageDataset, LabelDataset
from tqdm import tqdm
import time
from src.motivation.tasti_wrapper import InferenceDataset
from src.system_architecture.alternate import EKO_alternate




class EKO_ratio(EKO_alternate):
    def __init__(self, config, images, initial_anchor = 0.8, exploit_ratio = 0.5):
        self.images = images
        self.initial_anchor = initial_anchor
        self.exploit_ratio = exploit_ratio
        super().__init__(config, images, initial_anchor = initial_anchor)


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
        self.explore_count = 0
        self.exploit_count = 0

        for i in range(n_reps - curr_len):
            ### get the location with max dist
            random_number = random.random()
            if random_number <= self.exploit_ratio:
                chosen_section_start = np.argmax(dists)
                self.exploit_count += 1
            else:
                chosen_section_start = np.argmax(temporal_dists)
                self.explore_count += 1


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

