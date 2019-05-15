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
import yaml

# External modules
from tf.transformations import euler_matrix, translation_matrix


class InversePerspectiveMapping:
    def __init__(self):
        pass
        # self.info = self.load_camera_info(filename)
        # self.width = self.info['width']
        # self.height = self.info['height']
        
        # Intrinsics
        # self.K = np.zeros((3, 4))
        # self.K[:, :3] = np.asarray(self.info['intrinsics']['K']).reshape(3, 3)
        # self.D = np.asarray(self.info['intrinsics']['D'])

        # Extrinsics
        # self.rpy = np.radians(self.info['extrinsics']['rpy'])
        # self.rotation = euler_matrix(self.rpy[0], self.rpy[1], self.rpy[2])
        # self.translation = translation_matrix(self.info['extrinsics']['position'])
        # self.H = (self.translation @ self.rotation)

        # self.projection_matrix = np.asarray([[1, 0, -self.width/2],
        #                                      [0, 1, -self.height/2],
        #                                      [0, 0, 0],
        #                                      [0, 0, 1]])
        # self.perspective_matrix = self.K @ self.H @ self.projection_matrix

    def load_camera_info(self, filename):
        with open(filename, 'r') as f:
            camera_info = yaml.load(f)
        return camera_info

    def transform(self, img):
        img = cv2.warpPerspective(img, self.perspective_matrix, (self.width, self.height))
        return img

    def run(self, frame):
        h, w = 500, 500
        base = 1100
        s = 1
        # in_points = np.float32([(581, 155), (661, 155), (0, 406), (860, 406)])
        in_points = np.float32([(478, 407), (815, 408), (316, 680), (980, 684)])
        out_points = np.float32([(base, base), (h + base, base), (base, w + base), (h + base, w + base)])
        persM = cv2.getPerspectiveTransform(in_points, out_points)
        trans = np.asarray([[0, 0, 0], [0, 0, -1700], [0, 0, 0]]).reshape(3, 3)
        # scale = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]]).reshape(3, 3)
        persM = persM + trans
        # persM = np.matmul(persM, scale)
        dst = cv2.warpPerspective(frame, persM, (2700, 2500))
        # dst = cv2.resize(dst, (0, 0), fx=0.08, fy=0.3)
        return dst, persM


if __name__ == '__main__':
    ipm = InversePerspectiveMapping('camera_info.yaml')
    folder = '../dataset/carla_dataset_02'
    image_files = os.listdir(folder)
    image_files.sort()

    win_name = 'Perspective Transform'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 100, 100)
    cv2.resizeWindow(win_name, 400, 400)
    
    for image_file in image_files:
        image = cv2.imread('calibration-00392.jpg')
        # image = cv2.imread(os.path.join(folder, image_file))
        # output = ipm.transform(image)
        output, M = ipm.test(image)
        cv2.imshow(win_name, output)
        # time.sleep(1/20.0)
        cv2.waitKey(0)
        break

    cv2.destroyAllWindows()
        # break


