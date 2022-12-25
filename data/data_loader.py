"""

In this file, we implement the video data and annotation loader for sigmod
"""

import os
import json
from PIL import Image
import numpy as np


from utils.video_decompression import DecompressionModule

class Loader:

    def __init__(self):
        self.dc = DecompressionModule()


    def load_video(self, path):
        final_path = os.path.join(path, 'video.mp4')
        images = self.dc.convert2images(final_path)

        return images

    def load_annotations(self, path):
        final_path = os.path.join(path, 'TEST_objects.json')

        with open(final_path, 'r') as f:
            blob = json.load(f)

        return blob
