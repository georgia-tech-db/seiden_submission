"""

In this folder, we will define various pyav utility functions

"""

import av
import numpy as np
import time
from tqdm import tqdm

class Clock:

    def __init__(self):
        self.st = None
        self.et = None


    def tic(self):
        self.st = time.perf_counter()

    def toc(self):
        self.et = time.perf_counter()

        return self.et - self.st


def pyav_inference(dataset_name, n_samples=None):
    video_directory = f'/srv/data/jbang36/video_data/{dataset_name}/video.mp4'
    pyav_wrapper = PYAV_wrapper(video_directory)
    iframe_indices = pyav_wrapper.get_iframe_indices()
    if n_samples is not None:
        iframe_indices = iframe_indices[:n_samples]

    #### I assume cutoff is like the length of video
    video_length = pyav_wrapper.get_video_length()
    iframe_mapping = pyav_wrapper.get_mapping(iframe_indices, video_length)

    return iframe_indices, iframe_mapping


class PYAV_wrapper:

    def __init__(self, directory, mode = 'r'):
        self.directory = directory
        self.mode = mode
        self.clock = Clock()


    def load_video(self):
        frames = []
        self.clock.tic()

        container = av.open(self.directory, self.mode)
        stream = container.streams.video[0]
        codec = stream.codec_context

        for frame in tqdm(container.decode(stream), desc='Loading Video Frames'):
            frames.append(frame.to_ndarray(format='rgb24')) ### you can't be doing this, there is some other way...

        frames = np.stack(frames)
        duration = self.clock.toc()
        print(f"returning {len(frames)} frames in {duration} seconds")
        return frames

    def load_keyframes(self):
        container = av.open(self.directory, self.mode)
        stream = container.streams.video[0]
        codec = stream.codec_context
        stream.codec_context.skip_frame = 'NONKEY'

        self.clock.tic()
        frames = []
        key_indexes = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray())
            key_indexes.append(round(frame.time * stream.average_rate))

        frames = np.stack(frames)
        duration = self.clock.toc()
        print(f'returning {len(frames)} frames in {duration} seconds')

        return frames, key_indexes

    def get_iframe_indices(self):
        ### can we get the indexes just by doing packets???

        self.clock.tic()
        container = av.open(self.directory, self.mode)
        stream = container.streams.video[0]

        key_indexes = []
        count = 0


        for packets in container.demux(stream):
            if packets.is_keyframe:
                key_indexes.append(count)
            count += 1

        duration = self.clock.toc()
        print(f'returning {len(key_indexes)} frames in {duration} seconds')

        key_indexes = np.array(key_indexes)
        return key_indexes

    def get_video_length(self):
        container = av.open(self.directory, self.mode)
        stream = container.streams.video[0]
        return stream.frames

    def get_mapping(self, iframe_indices, video_length):
        print('----')

        print(iframe_indices.shape)
        print('----')
        mapping = np.ndarray(shape=(video_length), dtype=np.int)
        curr_iframe = 0
        assert (curr_iframe == iframe_indices[0])
        curr_iframe_ii = 0
        for i in range(video_length):
            if not len(iframe_indices) == curr_iframe_ii + 1:
                if i == iframe_indices[curr_iframe_ii + 1]:
                    curr_iframe_ii += 1
            mapping[i] = curr_iframe_ii

        print(f'mapping shape is {mapping.shape}')
        ### we need to make some assertions

        return mapping


    def get_bytes(self):
        self.clock.tic()
        container = av.open(self.directory, self.mode)
        stream = container.streams.video[0]

        count = 0
        bytes = []
        for packet in container.demux(stream):
            if not packet.size == 0:
                bytes.append( packet.size )

                count += 1

        print(count)
        duration = self.clock.toc()
        print(f'returning {len(bytes)} frames in {duration} seconds')

        return bytes


    def get_metadata(self):
        #### this is going to getting all frame level metadata packed into pandas
        pass



#### Basic structure of the video container -> stream -> codec
#### from stream we can get packets -> frames -> images


def get_container(directory):
    container = av.open(directory, 'r')
    return container

def get_stream(container):
    video_stream = container.streams.video[0]
    return video_stream

def get_codec(video_stream):
    codec = video_stream.codec_context
    return codec

##### end of basica structure

##### now the util functions that we will utilize

def load_video(directory):
    print('loading video', directory)
    container = av.open(directory, 'r')
    stream = container.streams.video[0]
    n_frames = stream.frames


    frames = []

    for frame in container.decode(stream):
        #for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format='rgb24'))

    frames = np.stack(frames)
    return frames


def write_video(images, save_directory):
    n_samples, height, width = images.shape[0], images.shape[1], images.shape[2]

    #### you might not have enough memory for the whole thing.... it's super high resolution
    container = av.open(save_directory, 'w')
    stream = container.add_stream("mpeg4", rate=30)
    stream.width = 1920
    stream.height = 1080
    stream.pix_fmt = "yuv420p"

    for i in tqdm(range(n_samples)):
        img = images[i]
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    container.close()



def extract_key_frames(directory):
    # content = av.datasets.curated('pexels/time-lapse-video-of-night-sky-857195.mp4')
    count = 0
    content = directory
    with av.open(content) as container:
        # Signal that we only want to look at keyframes.
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = 'NONKEY'

        ## TODO: the problem is the index doesn't reflect the position in video.... we would have to calculate ourselves...
        ## TODO: we can do this by accessing tmp.time -- this gives the exact timestamp of the image frame.
        ## TODO: therefore, if we know the FPS (can be accessed by container., then we can calculate the EXACT INDEX.
        for frame in container.decode(stream):
            tmp = frame
            print(tmp.index)
            print(tmp.key_frame)
            print(tmp.pict_type)
            print('------------')
            if count == 10:
                break
            count += 1


#### but for some reason, this seems to be wrong information?? why are we able to set this value?
#### we need a real way to get the gop size....... can we do seeking?
def get_gop_size(directory):
    container = av.open(directory, 'r')
    video_stream = container.streams.video[0]
    codec = video_stream.codec_context
    return codec.gop_size

def extract_mvs(directory):
    ua_detrac = av.open(directory)
    video_stream = ua_detrac.streams.video[0]
    video_codec = video_stream.codec_context
    video_codec.export_mvs = True
    mvs_data = []
    count = 0
    for video_frame in ua_detrac.decode(video_stream):
        print(video_frame)
        for data in video_frame.side_data:
            mvs_data.append(data.to_ndarray())

    return mvs_data

def randoms():
    codec = None
    container = None
    stream = None
    codec.skip_frame = 'NONKEY' ### this allows us to traverse through the i frames only.


    ### this allows us to separate demuxing and decoding
    for video_packet in container.demux(stream):
        video_frame = video_packet.decode()




