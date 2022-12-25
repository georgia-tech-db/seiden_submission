import sys
sys.path.append('/nethome/jbang36/eko')

from tqdm import tqdm

import os
import numpy as np
from PIL import Image


def load_one_image(filename, size = (300,300)):
    image = Image.open(filename)  ### we need to convert to the designated size
    if size is not None:
        pil_size = (size[1], size[0])

        image = image.resize(pil_size)  ### not PIL expects width, height
    #### what if there is a black and white image??
    image_np = np.array(image)
    if image_np.ndim == 2:
        ### we need to expand it to rgb
        c = np.stack([image_np, image_np, image_np], axis=2)
        image_np = c

    return image_np


def load_compressed_images_to_np(load_directory, size = (300,300)):
    """

    :param load_directory:
    :param size: we assume size is (height, width)
    :return:
    """
    ### we load all images in the directory
    filenames = os.listdir(load_directory)
    filenames = sorted(filenames)

    if size is not None:
        pil_size = (size[1], size[0]) ###pil expects (width, height)
    else:
        pil_size = None

    all_images = []
    for i,filename in enumerate(tqdm(filenames)):
        full_filename = os.path.join(load_directory, filename)
        image = Image.open(full_filename) ### we need to convert to the designated size
        if pil_size is not None:
            image = image.resize(pil_size) ### not PIL expects width, height
        #### what if there is a black and white image??
        image_np = np.array(image)
        if image_np.ndim == 2:
            ### we need to expand it to rgb
            c = np.stack([image_np, image_np, image_np], axis=2)
            image_np = c
        #print(image_np.shape)
        all_images.append(image_np)

    result = np.stack(all_images)

    return result


def save_np_to_compressed_images(np_array, save_directory):
    for i in tqdm(range(len(np_array))):
        image_frame = np_array[i]

        im = Image.fromarray(image_frame)

        image_name = 'img{0:010d}.jpg'.format(i)
        filename = os.path.join(save_directory, image_name)
        im.save(filename)
