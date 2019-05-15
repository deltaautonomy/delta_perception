#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 30, 2019

References:
http://wiki.ros.org/message_filters
http://wiki.ros.org/cv_bridge/Tutorials/
http://docs.ros.org/api/image_geometry/html/python/
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscribe
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Standalone
if __name__ == '__main__':
    import sys
    sys.path.append('..')

    # Handle paths and OpenCV import
    from scripts.init_paths import *
    from scripts.utils import pil_image

# Run from scripts module
else:
    # Handle paths and OpenCV import
    from init_paths import *
    from utils import pil_image

# Built-in modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# External modules
import tensorflow as tf

# Handle package path 
print(osp.join(PKG_PATH, 'segmentation'))
add_path(osp.join(PKG_PATH, 'segmentation'))

# Local python modules
from protos import pipeline_pb2
from builders import model_builder
from google.protobuf import text_format
from libs.exporter import deploy_segmentation_inference_graph
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS

# Segmentation models
# MODEL="0818_icnet_0.5_1025_resnet_v1"
MODEL="0818_icnet_1.0_1025_resnet_v1"
# MODEL="0818_pspnet_1.0_713_resnet_v1"


class SegmentationModel:
    def __init__(self, config_path=None, trained_checkpoint_prefix=None):
        if trained_checkpoint_prefix is None:
            self.trained_checkpoint_prefix = osp.join(PKG_PATH, 'segmentation/weights/%s/model.ckpt' % MODEL)
        else: self.trained_checkpoint_prefix = trained_checkpoint_prefix
        
        if config_path is None:
            self.config_path = osp.join(PKG_PATH, 'segmentation/weights/%s/pipeline.config' % MODEL)
        else: self.config_path = config_path
        
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
        image_raw = pil_image(cv2.resize(image_raw, (2048, 1024)))
        self.predictions = self.sess.run(self.outputs,
            feed_dict={ self.placeholder_tensor: image_raw })
        self.predictions = self.predictions.astype(np.uint8)
        if len(self.label_color_map[0]) == 1: self.predictions = np.squeeze(self.predictions, -1)
        return self.predictions[0]
