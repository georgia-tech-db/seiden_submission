"""
In this file, we implement all the helper functions that are needed to prove the motivation section.
"""
import os
from data.data_loader import Loader
from udfs.yolov5_wrapper import YOLOv5Wrapper
from src.motivation.yolo_wrapper import YoloWrapper
from src.motivation.tasti_wrapper import MotivationConfig, MotivationTasti
from src.motivation.eko_wrapper import EKOConfig, EKO
from src.motivation.resnet_wrapper import ResnetWrapper
from src.motivation.maskrcnn_wrapper import MaskRCNNWrapper

### import the queries
from benchmarks.stanford.tasti.tasti.seiden.queries.queries import NightStreetAggregateQuery, \
                                                                NightStreetAveragePositionAggregateQuery, \
                                                                NightStreetSUPGPrecisionQuery, \
                                                                NightStreetSUPGRecallQuery

from src.system_architecture.alternate import EKO_alternate
from src.system_architecture.parameter_search import EKOPSConfig, EKO_PS

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from utils.file_utils import load_json


import torch
import torchvision
from tqdm import tqdm
import time
from sklearn.svm import SVC
import numpy as np



##################################
#### Evaluation Code #############
##################################

def evaluate_object_detection(gt_file, dt_file):
    cocoGT = COCO(gt_file)
    ## open the dt_file and put in annotations
    dt = load_json(dt_file)
    cocoDT = cocoGT.loadRes(dt['annotations'])
    cocoEVAL = COCOeval(cocoGT, cocoDT, 'bbox')
    cocoEVAL.evaluate()
    cocoEVAL.accumulate()
    cocoEVAL.summarize()





###################################
### Code for Query Execution ######
###################################
THROUGHPUT = 1 / 140

def query_process_aggregate(index):
    st = time.perf_counter()
    times = []
    query = NightStreetAggregateQuery(index)
    result = query.execute_metrics(err_tol=0.1, confidence=0.05)
    times.append(result['nb_samples'])

    et = time.perf_counter()
    times.append(et - st + times[0] * THROUGHPUT)

    return times


def query_process_precision(index):
    st = time.perf_counter()
    times = []

    query = NightStreetSUPGPrecisionQuery(index)
    result = query.execute_metrics(1000)
    times.append((result['precision'], result['recall']))

    et = time.perf_counter()
    times.append(et - st + 1000 * THROUGHPUT)

    return times


def query_process_recall(index):
    st = time.perf_counter()
    times = []

    query = NightStreetSUPGRecallQuery(index)
    result = query.execute_metrics(1000)
    times.append((result['precision'], result['recall']))

    et = time.perf_counter()
    times.append(et - st + 1000 * THROUGHPUT)

    return times


def query_process1(index):
    times = []

    query = NightStreetAggregateQuery(index)
    result = query.execute_metrics(err_tol=0.01, confidence=0.05)
    times.append(result['nb_samples'])

    result = query.execute_metrics(err_tol=0.1, confidence=0.05)
    times.append(result['nb_samples'])

    result = query.execute_metrics(err_tol=1, confidence=0.05)
    times.append(result['nb_samples'])

    return times





def query_process(index):

    times = []

    query = NightStreetAggregateQuery(index)
    result = query.execute_metrics(err_tol=0.01, confidence=0.05)
    times.append( result['nb_samples'] )

    im_size = 360
    query = NightStreetAveragePositionAggregateQuery(index, im_size)
    result = query.execute_metrics(err_tol=0.001, confidence=0.05)
    times.append( result['nb_samples'] )

    query = NightStreetSUPGPrecisionQuery(index)
    result = query.execute_metrics(7000)
    times.append( (result['precision'], result['recall']) )

    return times


def query_process2(index):
    times = []
    query = NightStreetSUPGPrecisionQuery(index)
    result = query.execute_metrics(5000)
    times.append((result['precision'], result['recall']))

    return times






###################################
### Code for index construction ###
###################################

def load_dataset(video_name):
    ### load video to memory
    loader = Loader()
    video_fp = os.path.join('/srv/data/jbang36/video_data/', video_name)
    images = loader.load_video(video_fp)

    return images


def execute_svm(images, image_size = None):


    ## how will we transform this matrix to fit the image size?
    width, height = images.shape[1], images.shape[2]
    if image_size is not None:
        width_division = width // image_size
        height_division = height // image_size
        new_images = images[:,::width_division, ::height_division,:]
    else:
        new_images = images
    new_images = new_images.reshape(len(new_images), -1)

    train_images = new_images[::1000]
    y_random = np.random.randint(2, size=len(train_images))

    ### we need to try out the svm model
    clf = SVC(gamma='auto')
    clf.fit(train_images, y_random)

    st = time.perf_counter()
    output = clf.predict(new_images)
    et = time.perf_counter()

    return et - st




def execute_resnet(images, image_size = None):
    resnet = ResnetWrapper()
    output = resnet.inference(images, image_size)

    return output


def execute_yolo(images):
    yolo = YOLOv5Wrapper()
    output = yolo.inference(images)

    return output


def execute_yolo2(images):
    yolo = YoloWrapper()
    output = yolo.inference(images)

    return output





def execute_maskrcnn(images, batch_size = 8):
    mask = MaskRCNNWrapper()

    output = mask.inference(images, batch_size = batch_size)

    return output


def execute_maskrcnn_features(images, batch_size = 8):
    mask = MaskRCNNWrapper()

    output = mask.inference_features(images, batch_size = batch_size)
    return output


def execute_eko(images, video_name, nb_buckets = 7000, dist_param=0.1, temp_param=0.9):
    ekoconfig = EKOConfig(video_name, nb_buckets = nb_buckets, dist_param = dist_param, temp_param = temp_param)
    eko = EKO(ekoconfig, images)
    eko.init()

    return eko


def execute_ekoalt(images, video_name, category = 'car', nb_buckets = 7000):
    ekoconfig = EKOPSConfig(video_name, category = category, nb_buckets = nb_buckets)
    ekoalt = EKO_alternate(ekoconfig, images)
    ekoalt.init()

    return ekoalt


def execute_tastipt(images, video_name, category = 'car', redo = False, image_size = None, nb_buckets = 7000):
    ### call tasti -- init, bucket... we must exclude dataloading time, we must include execution time (or at least count)
    do_train = False
    do_infer = redo
    motivationconfig = MotivationConfig(video_name, do_train, do_infer, image_size = image_size, nb_buckets = nb_buckets, category = category)
    motivationtasti = MotivationTasti(motivationconfig, images)
    motivationtasti.init()

    return motivationtasti



def execute_tasti(images, video_name, nb_buckets = 7000):
    ### call tasti -- init, bucket... we must exclude dataloading time, we must include execution time (or at least count)
    do_train = True
    motivationconfig = MotivationConfig(video_name, do_train, nb_buckets)
    motivationtasti = MotivationTasti(motivationconfig, images)
    motivationtasti.init()

    return motivationtasti






