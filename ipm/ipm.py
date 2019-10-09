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

# Run from scripts module
else:
    # Handle paths and OpenCV import
    try: from init_paths import *
    except ImportError: from scripts.init_paths import *

# Built-in modules
import yaml

# External modules
from transformations import euler_matrix, translation_matrix


class InversePerspectiveMapping:
    def __init__(self, filename='ipm/camera_info.yaml'):
        self.info = self.load_camera_info(osp.join(PKG_PATH, filename))
        self.frame_width = self.info['width']
        self.frame_height = self.info['height']

        # Intrinsics
        self.K = np.zeros((3, 4))
        self.K[:, :3] = np.asarray(self.info['intrinsics']['K']).reshape(3, 3)
        self.D = np.asarray(self.info['intrinsics']['D'])

        # Extrinsics
        self.extrinsics_euler = np.radians(self.info['extrinsics']['rpy'])
        self.extrinsics_rotation = euler_matrix(self.extrinsics_euler[0],
            self.extrinsics_euler[1], self.extrinsics_euler[2])
        self.extrinsics_translation = translation_matrix(self.info['extrinsics']['position'])

        # Overwrite calibration data if available
        if 'calibration' not in self.info:
            print('ERROR: Calibration not performed, run InversePerspectiveMapping.calibrate() first!')
            return
        else:
            print('Loading calibration data from file...')
            self.ipm_matrix = np.asarray(self.info['calibration']['ipm_matrix']).reshape(3, 3)
            self.ipm_matrix_inv = np.linalg.inv(self.ipm_matrix)
            self.ipm_image_dims = tuple(self.info['calibration']['ipm_image_dims'][::-1])
            self.ipm_px_to_m = self.info['calibration']['ipm_px_to_m']
            self.ipm_m_to_px = self.info['calibration']['ipm_m_to_px']
            self.calibration_ego_y = self.info['calibration']['calibration_ego_y']

    def load_camera_info(self, filename):
        with open(filename, 'r') as f:
            camera_info = yaml.load(f)
        return camera_info

    def transform_image(self, img):
        img = cv2.warpPerspective(img, self.ipm_matrix, self.ipm_image_dims)
        return img

    def transform_points_to_px(self, points, squeeze=True, inverse=False):
        ones = np.ones((len(points), 1))
        if len(np.array(points).shape) == 1:
            points = np.expand_dims(points, axis=0)
            ones = np.array([[1]])

        H = self.ipm_matrix_inv if inverse else self.ipm_matrix
        points_px = np.matmul(H, np.hstack([points, ones]).T)
        points_px = points_px / points_px[-1]

        if squeeze: return points_px.T[:, :2].squeeze()
        return points_px.T[:, :2]

    def transform_points_to_m(self, points, inverse=False, transform=True):
        points_px = self.transform_points_to_px(points, squeeze=False, inverse=inverse) if transform else points
        points_px[:, 0] = points_px[:, 0] - self.ipm_image_dims[0] / 2
        points_px[:, 1] = self.ipm_image_dims[1] - points_px[:, 1]

        points_m = points_px * self.ipm_px_to_m
        points_m[:, 1] += self.calibration_ego_y

        return points_m.squeeze()


if __name__ == '__main__':
    ipm = InversePerspectiveMapping()
    img = np.zeros((720, 1280))
    print(ipm.transform_image(img).shape)
    points = np.array([[  0,  28],
     [869,  42],
     [  0,  80],
     [869,  93],
     [  0, 127],
     [869, 140]])
    print(ipm.transform_points_to_px(np.fliplr(points), inverse=True))
