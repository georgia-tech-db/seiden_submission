"""
In this file we define all the video util functions

some example functions are

1. loading from a compressed video to numpy array
2. saving a numpy array to a compressed video
"""
import cv2
import sys
sys.path.append('/nethome/jbang36/eko')

from utils.video_decompression import DecompressionModule
from tqdm import tqdm


def load_compressed_video_to_np(load_directory, size = (300,300)):
    dc = DecompressionModule()
    return dc.convert2images(load_directory, size=size)



def save_np_to_compressed_video(np_array, save_directory):

    if type(np_array) == list:
        tmp = np_array[0]
        frame_width = tmp.shape[1]
        frame_height = tmp.shape[0]
    else:
        frame_width = np_array.shape[2]
        frame_height = np_array.shape[1]

    size = (frame_width, frame_height)
    print(size)

    result = cv2.VideoWriter(save_directory,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             24, size)

    for i in tqdm(range(len(np_array))):
        image_frame = np_array[i]

        # Write the frame into the
        # file 'filename.avi'
        result.write(image_frame)


    # When everything done, release
    # the video capture and video
    # write objects
    result.release()




