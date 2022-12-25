"""
There are some hypothesis we want to confirm.

1. for static cameras, we think that pixel difference is a better indicator than temporal distance.
2. For moving cameras, we think that temporal difference is a better indicator than pixel distance.

3. Rather than using one or the other, using both indicators will give us better samples
"""

from src.system_architecture.parameter_search import EKO_PS, EKOPSConfig
from scipy.stats import pearsonr, spearmanr
import numpy as np
from benchmarks.stanford.tasti.tasti.query import BaseQuery
from sklearn.metrics import precision_score, recall_score, f1_score
#### confirmation of the 1st and 2nd hypothesis.


### we need to understand where the anchors are placed and how the labels change.
### first, get the labels for a given query.
### second, get where the anchors are placed.




#### create a custom query to examine the results
def query_process3(index):
    times = []
    query = CustomQuery(index)
    y_pred, y_true = query.execute()

    return y_pred, y_true



def query_process4_uadetrac(index):
    times = []
    query = CustomQuery(index)
    y_pred, y_true = query.execute()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_pred, y_true)))

    query = Custom5(index)
    y_pred, y_true = query.execute()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_pred, y_true)))

    query = Custom7(index)
    y_pred, y_true = query.execute()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_pred, y_true)))

    return times


def query_process4(index):
    times = []
    reps = []
    query = CustomQuery(index)
    y_pred, y_true = query.execute()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    if np.count_nonzero(np.isnan(y_pred)) != 0: print('nan in y_pred', np.count_nonzero(np.isnan(y_pred)))
    elif np.isnan(y_true).any(): print('nan in y_true', y_true)
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    reps.append(index.reps)

    query = Custom7(index)
    y_pred, y_true = query.execute()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    reps.append(index.reps)

    query = Custom8(index)
    y_pred, y_true = query.execute()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    reps.append(index.reps)

    query = Custom9(index)
    y_pred, y_true = query.execute()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    reps.append(index.reps)

    query = Custom10(index)
    y_pred, y_true = query.execute()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    reps.append(index.reps)

    return times, reps

def query_process4_simple(index):
    times = []
    query = CustomQuery(index)
    y_pred, y_true = query.execute_simple()
    all_reps = []
    y_trues = []
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    if np.count_nonzero(np.isnan(y_pred)) != 0:
        print('nan in y_pred', np.count_nonzero(np.isnan(y_pred)))
    elif np.isnan(y_true).any():
        print('nan in y_true', y_true)
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    all_reps.append(index.reps)
    y_trues.append(y_true)

    query = Custom2(index)
    y_pred, y_true = query.execute_simple()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    all_reps.append(index.reps)
    y_trues.append(y_true)

    query = Custom3(index)
    y_pred, y_true = query.execute_simple()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    all_reps.append(index.reps)
    y_trues.append(y_true)


    query = Custom4(index)
    y_pred, y_true = query.execute_simple()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    all_reps.append(index.reps)
    y_trues.append(y_true)

    query = Custom5(index)
    y_pred, y_true = query.execute_simple()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    times.append((precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)))
    all_reps.append(index.reps)
    y_trues.append(y_true)


    return times, all_reps, y_trues


class CustomQuery(BaseQuery):

    def finish_index_building(self):
        index = self.index
        target_dnn = self.index.target_dnn_cache
        scoring_func = self.score
        index.build_additional_anchors(target_dnn, scoring_func)


    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0

    def execute_simple(self, err_tol=0.01, confidence=0.05, y=None):

        self.finish_index_building()

        if y == None:
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists
            )
        else:
            y_pred, y_true = y

        ## convert y_true

        y_true = np.array([float(tmp) for tmp in y_true])
        return y_pred, y_true

    def execute(self, err_tol=0.01, confidence=0.05, y=None):
        if y == None:
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists
            )
        else:
            y_pred, y_true = y

        ## convert y_true

        y_true = np.array([float(tmp) for tmp in y_true])
        return y_pred, y_true

class Custom5(CustomQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 4 else 0.0

class Custom6(CustomQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 5 else 0.0

class Custom7(CustomQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 6 else 0.0

class Custom8(CustomQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 7 else 0.0

class Custom9(CustomQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 8 else 0.0


class Custom10(CustomQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 9 else 0.0


class Custom2(CustomQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 1 else 0.0

class Custom3(CustomQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 2 else 0.0

class Custom4(CustomQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 3 else 0.0

def generate_label_dist(eko, images, random_indices, done):
    train_or_test = None
    target_dnn_cache = [0] * len(images)
    labels = eko.override_target_dnn_cache(target_dnn_cache, train_or_test)
    car_counts = []
    for i in range(len(random_indices) - 1):
        j = i + 1
        idx_i = random_indices[i]
        idx_j = random_indices[j]
        if (idx_i, idx_j) in done:
            l1 = labels[idx_i]
            l2 = labels[idx_j]
            label_diff = abs(len(l1) - len(l2)) * 100
            if label_diff == 0 and len(l1) != 0:
                for j in range(len(l1)):
                    loc1 = l1[j].xmin
                    loc2 = l2[j].xmin
                    label_diff += abs(loc1 - loc2) / 300 * 100

            done[(idx_i, idx_j)].append(label_diff)
            car_counts.append(label_diff)

    return car_counts, done


def pixel_vs_temporal(images, video_name):
    ## images are from decoded frames
    ## labels are parsed from the index -- generate_label_dist from ekops

    nb_buckets = 1000
    ekoconfig = EKOPSConfig(video_name, nb_buckets=nb_buckets)

    eko = EKO_PS(ekoconfig, images)
    choices = np.arange(len(images))
    n_sample = nb_buckets
    random_indices = np.random.choice(choices, n_sample, replace=False)

    random_indices = sorted(random_indices)

    images_downsampled = []
    for image in images:
        new_image = image[::20, ::20, :].mean(axis=2).astype(np.int32)
        images_downsampled.append(new_image)

    ### compute the pixel distances
    dataset_length = len(images)
    block, done = eko.generate_matrix(images_downsampled, random_indices)
    print(block.shape)
    pixel_uncertainty = block[:,0]
    temporal_uncertainty = block[:,1]
    label_uncertainty, _ = generate_label_dist(eko, images, random_indices, done)

    for i in range(len(pixel_uncertainty)):
        print(pixel_uncertainty[i], temporal_uncertainty[i], label_uncertainty[i])

    #### normalize these things respectively
    normalized_pixel_uncertainty = pixel_uncertainty / pixel_uncertainty.max()
    normalized_temporal_uncertainty = temporal_uncertainty / temporal_uncertainty.max()

    ### now compute the correlation between these and the label
    print(len(label_uncertainty))
    print(len(normalized_temporal_uncertainty))
    print(len(normalized_pixel_uncertainty))
    assert(len(label_uncertainty) == len(normalized_pixel_uncertainty))
    assert(len(label_uncertainty) == len(normalized_temporal_uncertainty))

    p_pixel, _ = pearsonr(normalized_pixel_uncertainty, label_uncertainty)
    p_temp, _  = pearsonr(normalized_temporal_uncertainty, label_uncertainty)

    s_pixel, _ = spearmanr(normalized_pixel_uncertainty, label_uncertainty)
    s_temp, _ = spearmanr(normalized_temporal_uncertainty, label_uncertainty)

    print(p_pixel, p_temp)
    print(s_pixel, s_temp)

    return p_pixel, p_temp, s_pixel, s_temp



