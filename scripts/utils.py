#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019
'''

# Handle paths and OpenCV import
from init_paths import *

# External modules
import PIL.Image

# ROS modules
import rospy
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticStatus
from tf.transformations import (translation_matrix,
                                quaternion_matrix,
                                concatenate_matrices,
                                euler_from_quaternion)

# Global variables
CV_BRIDGE = CvBridge()


########################### Functions ###########################


class FPSLogger:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.fps = None
        self.last = None
        self.total_time = 0
        self.total_frames = 0

    def lap(self):
        self.last = time.time()

    def tick(self, count=1):
        self.total_time += time.time() - self.last
        self.total_frames += count
        self.fps = self.total_frames / self.total_time

    def log(self, tick=False):
        if tick: self.tick()
        print('\033[94m %s FPS:\033[00m \033[93m%.1f\033[00m' % (self.name, self.fps))

    def get_log(self, tick=False):
        if tick: self.tick()
        return '\033[94m %s FPS:\033[00m \033[93m%.1f\033[00m' % (self.name, self.fps)


def pil_image(img):
    return PIL.Image.fromarray(img)


def cv_image(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def message_to_cv2(msg, coding='bgr8'):
    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(msg, coding)
        return img
    except CvBridgeError as e:
        print(e)
        rospy.logerr(e)
        return None


def cv2_to_message(img, pub, coding='bgr8'):
    # Publish image using CV bridge
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        pub.publish(CV_BRIDGE.cv2_to_imgmsg(img, coding))
    except CvBridgeError as e: 
        print(e)
        rospy.logerr(e)


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def get_bbox_vertices(pose, dims, scale=None):
    '''
    Returns:
        vertices - 8 * [x, y, z, 1] ndarray
    '''
    if scale is not None: dims = [scale.x, scale.y, scale.z]
    dx, dy, dz = dims[0] / 2.0, dims[1] / 2.0, dims[2]
    vertices = [[dx, dy, 0, 1],
                [dx, dy, dz, 1],
                [dx, -dy, 0, 1],
                [dx, -dy, dz, 1],
                [-dx, dy, 0, 1],
                [-dx, dy, dz, 1],
                [-dx, -dy, 0, 1],
                [-dx, -dy, dz, 1]]
    vertices = np.matmul(pose, np.asarray(vertices).T).T
    return vertices


def position_to_numpy(position):
    return np.asarray([position.x, position.y, position.z])


def orientation_to_numpy(orientation):
    return np.asarray([orientation.x, orientation.y, orientation.z, orientation.w])


def numpy_to_position(numpy_pos, position):
    position.position.x = numpy_pos[0]
    position.position.y = numpy_pos[1]
    position.position.z = numpy_pos[2]
    return position


def orientation_to_rpy(orientation):
    return euler_from_quaternion(orientation_to_numpy(quaternion))


def quaternion_to_rpy(quaternion):
    return euler_from_quaternion(quaternion)


def pose_to_transformation(pose=None, position=None, orientation=None):
    if position is None:
        position = position_to_numpy(pose.position)
    if orientation is None:
        orientation = orientation_to_numpy(pose.orientation)
    return concatenate_matrices(translation_matrix(position), quaternion_matrix(orientation))


def make_diagnostics_status(name, pipeline, fps, level=DiagnosticStatus.OK):
    msg = DiagnosticStatus()
    msg.level = DiagnosticStatus.OK
    msg.name = name
    msg.message = fps
    msg.hardware_id = pipeline
    return msg
