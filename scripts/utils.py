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

# ROS modules
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

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


def message_to_cv2(msg):
    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(msg, 'bgr8')
        return img
    except CvBridgeError as e:
        print(e)
        rospy.logerr(e)
        return None


def cv2_to_message(img, topic):
    # Publish image using CV bridge
    try:
        topic.publish(CV_BRIDGE.cv2_to_imgmsg(img, "bgr8"))
    except CvBridgeError as e: 
        print(e)
        rospy.logerr(e)


