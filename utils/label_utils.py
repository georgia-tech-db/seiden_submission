"""
In this file, we define some util functions related to creating labels
"""
import sys
sys.path.append('/nethome/jbang36/eko')

from utils.file_utils import load_torch
from utils.label_maps import COCO_LABEL_MAP
import torch
import numpy as np

from utils.file_utils import save_torch
from tqdm import tqdm


THRESHOLD_MAP = {
    'car': {
        1: 0.5,
        2: 0.4,
        3: 0.4},
    'bus': {
        1: 0.2},
    'motorcycle':
        {
            1: 0.2},
    'truck': {
        1: 0.2}
}


def convert_OD_to_binary(od_label_filename, binary_label_filename, query_object, query_count, object_threshold):
    """
    This function is used for od_label_filenames such as json files in
    /srv/data/jbang36/video_data/cityscapes/udf_annotations

    These json files are saved with torch.
    These json files give {'image_filenames': ,
                            'categories': ,
                            'annotations': [{'boxes':, 'scores':, 'labels':}]}
    :param od_label_filename:
    :param binary_label_filename:
    :param query_object:
    :param query_count:
    :param object_threshold:
    :return:
    """
    annotations = load_torch(od_label_filename)
    ### we need to find query object index
    class_idx = annotations['categories'].index(query_object)
    if class_idx == -1:
        print(f'object not in categories: {od_label_filename}')
        raise ValueError

    new_annotations = []
    for annotation in tqdm(annotations['annotations']):
        scores = annotation['scores']
        labels = annotation['labels']
        max_scores = []  ## we save the max score per relevant_class

        relevant_scores = scores[labels == class_idx]
        ### we want to sort these scores my maximum
        if len(relevant_scores) < query_count:
            new_annotations.append(0)
        else:
            ### the scores for these objects are above a certain point
            sorted_scores, indices_scores = torch.sort(relevant_scores)
            ### we just get from the back
            if torch.min(sorted_scores[-query_count]) > object_threshold:
                new_annotations.append(1)
            else:
                new_annotations.append(0)

    assert(len(new_annotations) == len(annotations['annotations']))

    new_annotations = torch.Tensor(new_annotations)
    save_torch(new_annotations, binary_label_filename)




def convert_OD_to_IC(od_label_filename, ic_label_filename, relevant_classes, background_threshold):
    """
    1. The process we perform is first we load the od file,
    2. Then we extract the relevant classes and just save the most dominant one
    3. If none of the objects exceed the background_threshold, we just save the label as __background__
    4. we also need to output a class_list -- list of classes we were interested in


    :param od_label_filename: file to object detection label location
    :param ic_label_filename: file to image classification label location -- we save things here
    :return:
    """
    annotations = load_torch(od_label_filename)

    relevant_classes_idx = []
    for relevant_class in relevant_classes:
        relevant_classes_idx.append(COCO_LABEL_MAP[relevant_class])

    new_annotations = []
    for annotation in tqdm(annotations):
        scores = annotation['scores']
        labels = annotation['labels']
        max_scores = []  ## we save the max score per relevant_class
        for class_idx in relevant_classes_idx:
            relevant_scores = scores[labels == class_idx]

            max_score = torch.max(relevant_scores) if len(relevant_scores) > 0 else 0
            max_scores.append(max_score)
        ### we get the class_idx of the max score or zero otherwise
        max_max_score = np.max(max_scores)
        max_max_class_idx = np.argmax(max_scores)
        final_class_idx = max_max_class_idx + 1 if max_max_score > background_threshold else 0
        new_annotations.append(final_class_idx)

    ### save this folder....
    ### we need to make updates to relevant_classes
    relevant_classes.insert(0, '__background__')

    save_dictionary = {
        'label_list': relevant_classes,
        'labels'    : new_annotations
    }

    save_torch(save_dictionary, ic_label_filename)


if __name__ == "__main__":

    #### You would do something like this to convert the labels


    import os

    location = os.path.join('/srv/data/jbang36/video_data/dashcam2/coco_label', 'objects.json')

    from utils.file_utils import load_torch

    annotations = load_torch(location)

    relevant_classes = ['car', 'truck', 'bus', 'person']

    from utils.label_utils import convert_OD_to_IC

    for dataset_name in ['dashcam2']:
        od_label_filename = os.path.join(f'/srv/data/jbang36/video_data/{dataset_name}/coco_label', 'objects.json')
        ic_label_filename = os.path.join(f'/srv/data/jbang36/video_data/{dataset_name}/ic_labels', 'labels.json')
        relevant_classes = relevant_classes
        background_threshold = 0.7

        convert_OD_to_IC(od_label_filename, ic_label_filename, relevant_classes, background_threshold)
