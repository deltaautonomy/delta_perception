#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Apoorv Singh, Heethesh Vhavle
Email   : apoorvs@cmu.edu
Version : 1.0.0
Date    : Apr 20, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Standalone
if __name__ == '__main__':
    import sys
    sys.path.append('..')

    # Handle paths and OpenCV import
    from scripts.init_paths import *

# Run from scripts module
else:
    # Handle paths and OpenCV import
    from init_paths import *

# External modules
import tensorflow as tf
import matplotlib.pyplot as plt

# Set environment variables
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import logging
# tf.get_logger().setLevel(logging.ERROR)

# Local python modules
from lanenet.lanenet_model import lanenet_cluster
from lanenet.lanenet_model import lanenet_merge_model
from lanenet.lanenet_model import lanenet_postprocess

# Global variables
VGG_MEAN = [103.939, 116.779, 123.68]


class LaneNetModel:
    def __init__(self, weights_path=None):
        if weights_path is None: self.weights_path = osp.join(PKG_PATH, 'lanenet/weights/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000')
        else: self.weights_path = weights_path

    def setup(self):
        # Setup session
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.phase_tensor = tf.constant('test', tf.string)
        self.net = lanenet_merge_model.LaneNet(phase=self.phase_tensor, net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='lanenet_model')
        self.cluster = lanenet_cluster.LaneNetCluster()
        self.postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
        self.saver = tf.train.Saver()
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True
        # device_count={'GPU': 1})
        self.sess = tf.Session(config=self.sess_config)

        # Load weights
        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path=self.weights_path)

    def run(self, img):
        with self.sess.as_default():
            img = self.preprocess(img)

            self.binary_seg_image, self.instance_seg_image = self.sess.run([self.binary_seg_ret, 
                self.instance_seg_ret], feed_dict={self.input_tensor: [img]})

            self.binary_seg_image[0] = self.postprocessor.postprocess(self.binary_seg_image[0])

            self.mask_image, self.point_list, self.valid_list = self.cluster.get_lane_mask(self.binary_seg_image[0],
                self.instance_seg_image[0])
            for i in range(4):
                self.instance_seg_image[0][:, :, i] = self.instance_seg_image[0][:, :, i]
            self.embedding_image = np.array(self.instance_seg_image[0], np.uint8)

        return self.mask_image, self.point_list, self.valid_list

    def preprocess(self, img):
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        frame = image - VGG_MEAN
        return frame

    def close(self):
        self.sess.close()

    def minmax_scale(self, img):
        min_val = np.min(img)
        max_val = np.max(img)
        if (max_val == min_val):
            return img
        else:
            output_arr = (img - min_val) * 255.0 / (max_val - min_val)
            return output_arr

    def visualize(self, input_frame):
        # Draw on image
        self.mask_image = cv2.resize(self.mask_image, (input_frame.shape[1], input_frame.shape[0]))
        self.output = cv2.addWeighted(input_frame, 0.5, self.mask_image[:, :, (2, 1, 0)], 1, 0)
        return self.output


if __name__ == '__main__':
    lanenet = LaneNetModel('weights/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000')
    lanenet.setup()

    cap = cv2.VideoCapture('1_pune.mp4')
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', (640, 360))
    cv2.moveWindow('output', 100, 300)

    while cap.isOpened():
        ret, frame = cap.read()
        mask_image, point_list = lanenet.run(frame)

        # Visualize detects
        output = lanenet.visualize(frame)
        cv2.imshow('output', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    lanenet.close()
