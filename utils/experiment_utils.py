import numpy as np
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics  ## we will use precision_score, recall_score, f1_score
from utils.logger import Logger

logger = Logger()


def get_mapping(rep_indices: np.array, cluster_labels: np.array):
    """
    When evaluating whether the clustering method is correct, we need a way to propagate to all the frames

    if rep_indices = [2,4,7] and cluster_labels = [1,1,1,2,2,3,3,3]
    mapping is  [0,0,0,1,1,2,2,2] ## returns the indices of the rep_indices -- YESSSSS

    :param rep_indices: indices that are chosen as representative frames (based on the original images_compressed array
    :param cluster_labels: the cluster labels that are outputted by the algorithm (basically labels)
    :return: mapping from rep frames to all frames (basically tells us what each chosen frames is representing (normally used for deprecated purposes)


    """
    rep_indices = np.array(rep_indices)
    cluster_labels = np.array(cluster_labels)

    mapping = np.zeros(len(cluster_labels))
    for i, value in enumerate(rep_indices):
        corresponding_cluster_number = cluster_labels[value]
        # print(f"corresponding cluster number: {corresponding_cluster_number}")
        members_in_cluster_indices = cluster_labels == corresponding_cluster_number
        # print(f"member indices in the cluster: {members_in_cluster_indices} ")
        mapping[members_in_cluster_indices] = i
        # print(f"mapping: {mapping}")
        # print(f"----------------------------------")

    ##mapping should be integers
    mapping = mapping.astype(np.int)

    return mapping


def get_benchmark_results_sampled(gt_labels, sampled_predictions, mapping):
    data_pack = evaluate_with_gt5(gt_labels, sampled_predictions, mapping)
    ### data_pack has keys ( query_accuracy, precision, recall, f1_score
    f1_score = data_pack['f1_score']
    precision_score = data_pack['precision']
    recall_score = data_pack['recall']
    benchmark_results = {'F1-Score': f1_score, 'Precision': precision_score, 'Recall': recall_score}
    return benchmark_results


def create_binary_annotation(annotations:list, category:str, label_map:dict):
    converted_category = label_map[category]
    binary_labels = np.zeros(len(annotations))
    for i, frame_annotation in enumerate(annotations):
        labels = frame_annotation['labels']
        if converted_category in labels:
            binary_labels[i] = 1 ### binary label = 1 is whether there are cars

    return binary_labels



def propagate_labels(sampled_predicted_labels: dict, mapping):
    ## we propagate the labels from sampling to all frames
    new_dict = {}
    for key, value in sampled_predicted_labels.items():
        # print(f"{key}, type of key {type(key)}")
        new_dict[key] = np.zeros(len(mapping))
        for i in range(len(mapping)):
            new_dict[key][i] = sampled_predicted_labels[key][mapping[i]]

    return new_dict




def evaluate_with_gt5(labels, rep_labels, mapping):
    """
    This function differs from evaluate with gt in the aspect that we have already converted things to binary
    :param images:
    :param labels:
    :param boxes:
    :param rep_images:
    :param rep_labels:
    :param rep_boxes:
    :param mapping:
    :param labelmap:
    :return:
    """

    all_gt_labels = {}
    all_gt_labels['foo'] = labels

    all_rep_labels = {}
    all_rep_labels['foo'] = rep_labels

    sampled_propagated_predicted_labels = propagate_labels(all_rep_labels, mapping)

    for key, value in all_gt_labels.items():
        length = min(len(all_gt_labels[key]), len(sampled_propagated_predicted_labels[key]))
        modified_gt = all_gt_labels[key][:length]
        modified_sampled = sampled_propagated_predicted_labels[key][:length]
        accuracy = accuracy_score(modified_gt, modified_sampled)
        precision = metrics.precision_score(modified_gt, modified_sampled)
        recall = metrics.recall_score(modified_gt, modified_sampled)
        f1_score = metrics.f1_score(modified_gt, modified_sampled)

    data_pack = {}
    data_pack['query_accuracy'] = accuracy
    data_pack['precision'] = precision
    data_pack['recall'] = recall
    data_pack['f1_score'] = f1_score

    return data_pack

def evaluate_gt_proposed(labels, rep_labels):
    """
    This function differs from evaluate with gt in the aspect that we have already converted things to binary
    :param images:
    :param labels:
    :param boxes:
    :param rep_images:
    :param rep_labels:
    :param rep_boxes:
    :param mapping:
    :param labelmap:
    :return:
    """



    accuracy = accuracy_score(labels, rep_labels)
    precision = metrics.precision_score(labels, rep_labels)
    recall = metrics.recall_score(labels, rep_labels)
    f1_score = metrics.f1_score(labels, rep_labels)

    data_pack = {}
    data_pack['query_accuracy'] = accuracy
    data_pack['precision'] = precision
    data_pack['recall'] = recall
    data_pack['f1_score'] = f1_score

    return data_pack


def sample3_middle(images, labels, boxes, sampling_rate=30):
    ## for uniform sampling, we will say all the frames until the next selected from is it's 'property'
    reference_indexes = []
    length = len(images[::sampling_rate])

    if sampling_rate % 2 == 1:
        start = -(sampling_rate // 2)
        end = sampling_rate // 2
    else:
        start = -(sampling_rate // 2)
        end = sampling_rate // 2 - 1

    print(f"{sampling_rate} {start} {end}")
    for i in range(length):
        for j in range(start, end + 1):
            if (i * sampling_rate + j) < 0:
                continue
            if i * sampling_rate + j >= len(images):
                break
            reference_indexes.append(i)

    while len(reference_indexes) != len(images):
        reference_indexes.append(length - 1)

    print(f"{len(reference_indexes)} {len(images)}")
    assert (len(reference_indexes) == len(images))
    new_images = None
    new_labels = None
    new_boxes = None
    if images is not None:
        new_images = images[::sampling_rate]
    if labels is not None:
        new_labels = labels[::sampling_rate]
    if boxes is not None:
        new_boxes = boxes[::sampling_rate]

    return new_images, new_labels, new_boxes, reference_indexes
