#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle, Apoorv Singh
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019

References:
http://wiki.ros.org/message_filters
http://wiki.ros.org/cv_bridge/Tutorials/
http://docs.ros.org/api/image_geometry/html/python/
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscribe
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Handle paths and OpenCV import
from init_paths import *

# Built-in modules

# External modules
import matplotlib.pyplot as plt

# ROS modules
import tf
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError

# ROS messages
from sensor_msgs.msg import Image

# Local python modules
from utils import *
from lanenet.lanenet import LaneNetModel

# Global objects
cmap = plt.get_cmap('tab10')
VALIDATE = False

# Perception models
lanenet = LaneNetModel()

# FPS loggers
FRAME_COUNT = 0
lane_fps = FPSLogger('LaneNet')


########################### Functions ###########################


def lane_validation(img, image_gt_msg, lanes, image_pub, **kwargs):
    pass


def lane_detection(img, image_pub, vis=True, **kwargs):
    # Preprocess
    # img = increase_brightness(img)

    # Lane detection
    lane_fps.lap()
    try:
        lane_img, lanes = lanenet.run(img)
    except Exception as e:
        return
    lane_fps.tick()

    # Display FPS logger status
    # sys.stdout.write('\r%s | %s | %s | %s | %s ' % (all_fps.get_log(), yolo_fps.get_log(),
    #     lane_fps.get_log(), sort_fps.get_log(), fusion_fps.get_log()))
    # sys.stdout.flush()

    # Visualize and publish image message
    if vis:
        # Display lane markings
        overlay = cv2.resize(lane_img, (img.shape[1], img.shape[0]))
        img = cv2.addWeighted(img, 1.0, overlay, 1.0, 0)
        cv2_to_message(img, image_pub)

    return lanes


def callback(image_msg, image_pub, **kwargs):
    # Read image message
    img = message_to_cv2(image_msg)
    if img is None: return

    # Run the perception pipeline
    lanes = lane_detection(img, image_pub)


def shutdown_hook():
    print('\n\033[95m' + '*' * 30 + ' Lane Detection Shutdown ' + '*' * 30 + '\033[00m\n')
    lanenet.close()
    if VALIDATE:
        pass


def run(**kwargs):
    # Start node
    rospy.init_node('main', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())

    # Wait for main perception to start first
    rospy.wait_for_message('/delta_perception/object_detection_tracking', Image)

    # Setup models
    lanenet.setup()

    # Handle params and topics
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    # if VALIDATE: image_gt = rospy.get_param('~image_gt', '/carla/ego_vehicle/camera/semantic_segmentation/front/image_segmentation')
    output_image = rospy.get_param('~output_image', '/delta_perception/lane_detection')

    # Display params and topics
    rospy.loginfo('Image topic: %s' % image_color)

    # Publish output topic
    image_pub = rospy.Publisher(output_image, Image, queue_size=5)

    # Subscribe to topics
    image_sub = message_filters.Subscriber(image_color, Image)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer([image_sub], queue_size=1, slop=0.1)
    ats.registerCallback(callback, image_pub, **kwargs)

    # Shutdown hook
    rospy.on_shutdown(shutdown_hook)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    # Start perception node
    run()
