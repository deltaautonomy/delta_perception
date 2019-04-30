"""Run inference on an image or group of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from protos import pipeline_pb2
from builders import model_builder
from libs.exporter import deploy_segmentation_inference_graph
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS

class SegmentationModel:
    def __init__(self, config_path, trained_checkpoint_prefix):
        self.config_path = config_path
        self.trained_checkpoint_prefix = trained_checkpoint_prefix
        self.input_shape = [1024, 2048, 3]
        self.pad_to_shape = [1025, 2049]
        self.label_color_map = (CITYSCAPES_LABEL_COLORS)
    
    def setup(self):

        self.pipeline_config = pipeline_pb2.PipelineConfig()
        with tf.gfile.GFile(self.config_path, 'r') as f:
            text_format.Merge(f.read(), self.pipeline_config)

        self.num_classes, self.segmentation_model = model_builder.build(self.pipeline_config.model, is_training=False)
        self.outputs, self.placeholder_tensor = deploy_segmentation_inference_graph(
            model=self.segmentation_model,
            input_shape=self.input_shape,
            pad_to_shape=self.pad_to_shape,
            label_color_map=self.label_color_map)

        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        self.input_graph_def = tf.get_default_graph().as_graph_def()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.trained_checkpoint_prefix)

    def close(self):
        self.sess.close()

    def run(self, image_raw):
        self.predictions = self.sess.run(self.outputs,
            feed_dict={ self.placeholder_tensor: image_raw })
        self.predictions = self.predictions.astype(np.uint8)
        if len(self.label_color_map[0]) == 1: self.predictions = np.squeeze(self.predictions, -1)
        return self.predictions[0]
