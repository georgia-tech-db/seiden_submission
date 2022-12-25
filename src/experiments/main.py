from src.system_architecture.alternate import EKO_alternate
from src.system_architecture.parameter_search import EKOPSConfig, EKO_PS
from src.system_architecture.ratio import EKO_ratio
from src.system_architecture.mab import EKO_mab
import time
import os



from benchmarks.stanford.tasti.tasti.seiden.queries.queries import NightStreetAggregateQuery, \
                                                                NightStreetAveragePositionAggregateQuery, \
                                                                NightStreetSUPGPrecisionQuery, \
                                                                NightStreetSUPGRecallQuery



def execute_ekoalt_rq3(images, video_name, anchor_percentage = 0.5, nb_buckets = 10000):
    ekoconfig = EKOPSConfig(video_name, nb_buckets = nb_buckets)
    ekoalt = EKO_alternate(ekoconfig, images, initial_anchor=anchor_percentage)
    ekoalt.init()

    return ekoalt


def execute_ekoalt_rq4(images, video_name, anchor_percentage = 0.5, nb_buckets = 10000, exploit_ratio = 0.5):
    ekoconfig = EKOPSConfig(video_name, nb_buckets=nb_buckets)
    ekoalt = EKO_ratio(ekoconfig, images, initial_anchor=anchor_percentage, exploit_ratio = exploit_ratio)
    ekoalt.init()

    return ekoalt


def execute_ekomab(images, video_name, keep = False,  category = 'car', nb_buckets = 10000, anchor_percentage = 0.8, c_param = 2):
    ekoconfig = EKOPSConfig(video_name, category = category, nb_buckets=nb_buckets)
    base = '/srv/data/jbang36/video_data'
    directory = os.path.join(base, video_name, 'video.mp4')
    ekomab = EKO_mab(ekoconfig, images, directory, c_param = c_param, anchor_percentage=anchor_percentage, keep = keep)
    ekomab.init()


    return ekomab


THROUGHPUT = 1 / 140

def query_process_aggregate(index, error = 0.1, y = None):
    if index is None:
        assert(y is not None)
    if y is None:
        assert(index is not None)

    st = time.perf_counter()
    query = NightStreetAggregateQuery(index)
    result = query.execute_metrics(err_tol=error, confidence=0.05, y = y)
    nb_samples = result['nb_samples']

    et = time.perf_counter()
    t = et - st + nb_samples * THROUGHPUT

    return query, t


def query_process_precision(index, dnn_invocation = 1000,  y = None):
    if index is None:
        assert(y is not None)
    if y is None:
        assert(index is not None)

    query = NightStreetSUPGPrecisionQuery(index)
    result = query.execute_metrics(dnn_invocation, y=y)
    precision = result['precision']
    recall = result['recall']

    return precision, recall



def get_labels(index, dnn_invocation = 1000, y = None):
    if index is None:
        assert(y is not None)
    if y is None:
        assert(index is not None)

    query = NightStreetSUPGRecallQuery(index)

    result = query.execute_metrics(dnn_invocation, y=y)
    precision = result['precision']
    recall = result['recall']

    return query

def query_process_recall(index, dnn_invocation = 1000, y = None):
    if index is None:
        assert(y is not None)
    if y is None:
        assert(index is not None)

    query = NightStreetSUPGRecallQuery(index)

    result = query.execute_metrics(dnn_invocation, y=y)
    precision = result['precision']
    recall = result['recall']

    return precision, recall
