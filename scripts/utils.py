#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019
'''

# Handle paths and OpenCV import
from init_paths import PATH, PKG_PATH

# ROS modules
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo


########################### Functions ###########################


def message_to_cv2(msg):
    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(msg, 'bgr8')
        return img
    except CvBridgeError as e: 
        rospy.logerr(e)
        return None
