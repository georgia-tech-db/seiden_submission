"""
In this script, we implement functions related to exploring the novel system architecture.

First experiment we want to conduct is does the alpha, beta values matter -- do they influence the final accuracy outcome? - parameter_search.py

"""


from src.system_architecture.parameter_search import EKOPSConfig, EKO_PS
from src.system_architecture.simple import EKO_simple
from src.system_architecture.alternate import EKO_alternate


def execute_ekops(images, video_name, nb_buckets = 7000):
    ekoconfig = EKOPSConfig(video_name, nb_buckets = nb_buckets)
    eko = EKO_PS(ekoconfig, images)
    eko.init()

    return eko


def execute_ekos(images, video_name, nb_buckets = 7000):
    ekoconfig = EKOPSConfig(video_name, nb_buckets = nb_buckets)
    ekos = EKO_simple(ekoconfig, images)
    ekos.init()

    return ekos


def execute_ekoalt(images, video_name, nb_buckets = 7000):
    ekoconfig = EKOPSConfig(video_name, nb_buckets = nb_buckets)
    ekoalt = EKO_alternate(ekoconfig, images)
    ekoalt.init()

    return ekoalt
