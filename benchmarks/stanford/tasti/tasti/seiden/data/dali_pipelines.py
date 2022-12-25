



import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from tqdm import tqdm
import nvidia.dali.plugin as plugin



import sys
sys.path.append('/nethome/jbang36/eko')
from data.dali_wrappers import ExternalSourcePipeline, get_dali_dataloader



class EKODNNPipeline(ExternalSourcePipeline):
    def define_graph(self):
        self.jpegs = self.input()
        #self.labels = self.input_label()
        images = fn.decoders.image(self.jpegs, device = 'mixed', output_type = types.RGB)
        ### we are going to perform a crop, resize, normalize
        #xmin, xmax, ymin, ymax = 0, 1750, 540, 1080

        output = fn.crop_mirror_normalize(images, device = 'gpu', dtype = types.FLOAT, output_layout = 'CHW',
                                          std = 255)

        return output


class JacksonDNNPipeline(ExternalSourcePipeline):

    def define_graph(self):
        self.jpegs = self.input()
        # self.labels = self.input_label()
        images = fn.decoders.image(self.jpegs, device='mixed', output_type=types.BGR)
        ### we have to do cropping and we also have to divide by 255
        ### this would be incorrect, we need resizing not cropping
        new_x = 1920
        new_y = 1024

        output = fn.resize(images, device='gpu', resize_x=new_x, resize_y=new_y, interp_type=types.INTERP_TRIANGULAR)
        output = fn.crop_mirror_normalize(output,
                                          device='gpu', dtype=types.FLOAT,
                                          output_layout='CHW',
                                          std=255)

        return output


class TargetDNNPipeline(ExternalSourcePipeline):

    def define_graph(self):
        self.jpegs = self.input()
        #self.labels = self.input_label()
        images = fn.decoders.image(self.jpegs, device = 'mixed', output_type = types.BGR)
        ### we are going to perform a crop, resize, normalize
        #xmin, xmax, ymin, ymax = 0, 1750, 540, 1080

        crop_pos_x = 0
        crop_pos_y = 0.5
        crop_w = 1750
        crop_h = 1080
        images = fn.crop(images, crop_w = crop_w, crop_h =crop_h,
                         crop_pos_x = crop_pos_x, crop_pos_y = crop_pos_y) ##TODO: check if this works
        output = fn.crop_mirror_normalize(images, device = 'gpu', dtype = types.FLOAT, output_layout = 'CHW',
                                          std = 255)

        return output



class EmbeddingDNNPipeline(ExternalSourcePipeline):

    def define_graph(self):
        self.jpegs = self.input()
        #self.labels = self.input_label()
        images = fn.decoders.image(self.jpegs, device = 'mixed', output_type = types.BGR)
        ### we are going to perform a crop, resize, normalize
        #xmin, xmax, ymin, ymax = 0, 1750, 540, 1080
        crop_pos_x = 0
        crop_pos_y = 0.5
        crop_w = 1750
        crop_h = 1080

        images = fn.crop(images, crop_w=crop_w, crop_h=crop_h,
                         crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y) ##TODO: check if this works
        output = fn.resize(images, device='gpu', resize_x=224, resize_y=224, interp_type=types.INTERP_TRIANGULAR)
        output = fn.crop_mirror_normalize(output, device = 'gpu', dtype = types.FLOAT, output_layout = 'CHW',
                                          std = 255)

        return output

