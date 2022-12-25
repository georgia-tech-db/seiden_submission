
import os
import torch
import numpy as np
import torchvision
import sys
sys.path.append('/nethome/jbang36/seiden')

from benchmarks.stanford.tasti.tasti.index import Index
from benchmarks.stanford.tasti.tasti.config import IndexConfig


from benchmarks.stanford.tasti.tasti.seiden.data.data_loader import ImageDataset, LabelDataset
from tqdm import tqdm
import time
from src.motivation.tasti_wrapper import InferenceDataset



class EKO(Index):
    def __init__(self, config, images):
        self.images = images
        super().__init__(config)

    def get_cache_dir(self):
        os.makedirs(self.config.cache_dir, exist_ok = True)
        return self.config.cache_dir

    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        root = '/srv/data/jbang36/video_data'
        ROOT_DATA = self.config.video_name
        labels_fp = os.path.join(root, ROOT_DATA, 'tasti_labels.csv')
        labels = LabelDataset(
            labels_fp=labels_fp,
            length=len(target_dnn_cache)
        )
        return labels

    def get_target_dnn_dataset(self, train_or_test):
        ### just convert the loaded data into a dataset.
        dataset = InferenceDataset(self.images)
        return dataset

    def calculate_rep_methodology(self):
        rep_indices, dataset_length = self.get_reps()  ### we need to sort the reps
        top_reps = self.calculate_top_reps(dataset_length, rep_indices)
        top_dists = self.calculate_top_dists(dataset_length, rep_indices, top_reps)

        return rep_indices, top_reps, top_dists


    def _get_closest_reps(self, rep_indices, curr_idx):
        result = []

        for i, rep_idx in enumerate(rep_indices):
            if rep_idx - curr_idx >= 0:
                result.append(rep_indices[i-1])
                result.append(rep_indices[i])
                break
        return result


    def calculate_distance_uncertainty(self, all_images, rep_indices, dataset_length):
        distance = np.zeros(dataset_length)
        rep_indices = sorted(rep_indices)
        #### the images have been downsampled and converted to int32
        for i in range(dataset_length):
            if i in rep_indices: continue
            left, right = self._get_closest_reps(rep_indices, i)
            curr_image = all_images[i]
            left_image = all_images[left]
            right_image = all_images[right]
            first = np.linalg.norm(curr_image - left_image)
            second = np.linalg.norm(curr_image - right_image)
            distance[i] = min(first, second)
        return distance


    def calculate_temporal_uncertainty(self, all_images, rep_indices, dataset_length):
        temporal_distance = np.zeros(dataset_length)
        ## get the two closest reps and get the mean right?
        rep_indices = sorted(rep_indices)

        for i in range(dataset_length):
            if i in rep_indices: continue
            left, right = self._get_closest_reps(rep_indices, i)
            temporal_distance[i] = min(abs(left - i), abs(right - i))

        return np.array(temporal_distance)



    def update_distance_uncertainty(self, distance_uncertainty, images_downsampled, left_rep, middle_rep, right_rep):
        for i in range(left_rep+1, right_rep):
            if i == middle_rep: distance_uncertainty[i] = 0
            if i < middle_rep:
                left_image = images_downsampled[left_rep]
                right_image = images_downsampled[middle_rep]
                curr_image = images_downsampled[i]
            else:
                left_image = images_downsampled[middle_rep]
                right_image = images_downsampled[right_rep]
                curr_image = images_downsampled[i]

            distance_uncertainty[i] = min(np.linalg.norm(left_image - curr_image),
                                          np.linalg.norm(right_image - curr_image))

        return distance_uncertainty


    def update_temporal_uncertainty(self, temporal_uncertainty, images_downsampled, left_rep, middle_rep, right_rep):
        for i in range(left_rep+1, right_rep):
            if i == middle_rep: temporal_uncertainty[i] = 0
            if i < middle_rep:
                temporal_uncertainty[i] = min(i - left_rep, middle_rep - i)
            else:
                temporal_uncertainty[i] = min(i - middle_rep, right_rep - i)
        return temporal_uncertainty


    def get_other_half(self, rep_indices:list, n_reps):
        ### first, let's just get the images....
        ### we will exclude this time
        dataset_length = len(self.images)
        st = time.perf_counter()
        all_images = self.images
        images_downsampled = []
        for image in all_images:
            new_image = image[::20, ::20, :].mean(axis=2).astype(np.int32)
            images_downsampled.append(new_image)
        self.exclude_time = time.perf_counter() - st
        alpha = self.config.dist_param
        beta = self.config.temp_param
        ##### just for debugging purposes... we will save normalized distance uncertainty, temporal uncertainty, final_uncertainty
        curr_size = len(rep_indices)

        distance_uncertainty = self.calculate_distance_uncertainty(images_downsampled, rep_indices, dataset_length)
        temporal_uncertainty = self.calculate_temporal_uncertainty(images_downsampled, rep_indices, dataset_length)


        for i in tqdm(range(n_reps - curr_size), desc = "Choosing Other Rep Indices.."):
            ### okay, now that we have the image data, we need to calculate uncertainty
            ### play it easy first, we can optimize further later.
            normalized_distance_uncertainty = distance_uncertainty / distance_uncertainty.max()
            normalized_temporal_uncertainty = temporal_uncertainty / temporal_uncertainty.max()
            final_uncertainty = alpha * normalized_distance_uncertainty + beta * normalized_temporal_uncertainty
            ### find the index with max uncertainty and pick it
            chosen_rep = np.argmax(final_uncertainty)
            assert(chosen_rep not in rep_indices)
            ### get left, right
            left_rep, right_rep = self._get_closest_reps(rep_indices, chosen_rep) ### there will ALWAYS be a left and right
            right_rep_idx = rep_indices.index(right_rep)
            chosen_rep_idx = right_rep_idx
            rep_indices.insert(chosen_rep_idx, chosen_rep)
            right_rep_idx += 1

            #### we need to update the distances....
            #### the rep_indices array is already sorted, we only need to update values in a portion....
            #### TODO: Need to debug this place!!!!!
            distance_uncertainty = self.update_distance_uncertainty(distance_uncertainty, images_downsampled, left_rep, chosen_rep, right_rep)
            temporal_uncertainty = self.update_temporal_uncertainty(temporal_uncertainty, images_downsampled, left_rep, chosen_rep, right_rep)


        self.t_uncertainty = normalized_temporal_uncertainty
        self.d_uncertainty = normalized_distance_uncertainty
        self.f_uncertainty = final_uncertainty
        rep_indices = sorted(rep_indices)
        return rep_indices


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
        half_n_reps = int(self.config.rep_ratio * n_reps)
        skip_rate = dataset_length // (half_n_reps - 1)

        rep_indices = np.arange(dataset_length, dtype=np.int32)[::skip_rate]
        print(f'rep indices stats', n_reps, len(rep_indices))
        rep_indices = rep_indices.tolist()
        if dataset_length - 1 not in rep_indices: rep_indices.append(dataset_length - 1)

        rep_indices = self.get_other_half(rep_indices, n_reps)
        assert(len(rep_indices) == len(set(rep_indices)))


        return rep_indices, dataset_length


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



    def do_bucketting(self, percent_fpf=0.75):
        '''
        Given our embeddings, cluster them and store the reps, topk_reps, and topk_dists to finalize our TASTI.
        '''
        if self.config.do_bucketting:
            ### how do we get the total length of the video?

            self.reps, self.topk_reps, self.topk_dists = self.calculate_rep_methodology()

            np.save(os.path.join(self.cache_dir, 'reps.npy'), self.reps)
            np.save(os.path.join(self.cache_dir, 'topk_reps.npy'), self.topk_reps)
            np.save(os.path.join(self.cache_dir, 'topk_dists.npy'), self.topk_dists)
        else:
            print(os.path.join(self.cache_dir, 'reps.npy'))
            self.reps = np.load(os.path.join(self.cache_dir, 'reps.npy'))
            self.topk_reps = np.load(os.path.join(self.cache_dir, 'topk_reps.npy'))
            self.topk_dists = np.load(os.path.join(self.cache_dir, 'topk_dists.npy'))

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
            top_dists[i, 0] = abs(i - top_reps[i, 0])
            top_dists[i, 1] = abs(i - top_reps[i, 1])

        return top_dists



class EKOConfig(IndexConfig):
    def __init__(self, dataset_name, nb_buckets = 7000, rep_ratio = 0.1, dist_param = 0.1, temp_param = 0.9):
        super().__init__()
        self.do_mining = False
        self.do_training = False
        self.do_infer = False
        self.do_bucketting = True
        self.video_name = dataset_name

        self.cache_dir = f'/srv/data/jbang36/tasti_data/cache/{dataset_name}/seiden'

        self.device = 'gpu'
        self.num_threads = 16
        self.batch_size = 16
        self.nb_train = 1
        self.train_margin = 1.0
        self.train_lr = 1e-4
        self.max_k = 5
        self.nb_buckets = nb_buckets
        self.nb_training_its = 12000
        self.rep_ratio = rep_ratio
        self.dist_param = dist_param
        self.temp_param = temp_param


