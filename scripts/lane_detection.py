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

# External modules
import matplotlib.pyplot as plt

# ROS modules
import tf
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError

# ROS messages
from sensor_msgs.msg import Image
from delta_perception.msg import LaneMarking, LaneMarkingArray

# Local python modules
from utils import *
from lanenet.lanenet import LaneNetModel

# Frames
CAMERA_FRAME = 'ego_vehicle/camera/rgb/front'

# Perception models
lanenet = LaneNetModel()

# FPS loggers
FRAME_COUNT = 0
lane_fps = FPSLogger('LaneNet')


########################### Functions ###########################


def lane_detection(image_msg, image_pub, lane_pub, vis=True, **kwargs):
    # Read image message
    img = message_to_cv2(image_msg)
    if img is None: return

    # Preprocess
    # img = increase_brightness(img)

    # Lane detection
    lane_fps.lap()
    try:
        lane_img, points, _ = lanenet.run(img)
    except Exception as e:
        print(e)
        return
    lane_fps.tick()

    # Run lane detection
    if len(points) < 3:
        print('Less than 3 lanes in detected lanes visible')
        return

    # Fit lines on contours
    rows, cols = img.shape[:2]
    fy = img.shape[0] / float(lane_img.shape[0])
    fx = img.shape[1] / float(lane_img.shape[1])

    # Scale predicted points
    points[:, 0] = points[:, 0] * fx
    points[:, 1] = points[:, 1] * fy
    points[:, 2] = points[:, 2] * fx
    points[:, 3] = points[:, 3] * fy

    # Convert to lane marking array
    lane_array = LaneMarkingArray()
    lane_array.header.stamp = image_msg.header.stamp
    lane_array.header.frame_id = CAMERA_FRAME
    for xbot, ybot, xtop, ytop in points:
        lane = LaneMarking()
        lane.xtop, lane.ytop = xtop, ytop
        lane.xbot, lane.ybot = xbot, ybot
        lane_array.lanes.append(lane)

    # Publish the lane data
    lane_pub.publish(lane_array)

    # Visualize and publish image message
    if vis:
        # Display lane markings
        overlay = cv2.resize(lane_img, (img.shape[1], img.shape[0]))
        img = cv2.addWeighted(img, 1.0, overlay, 1.0, 0)
        cv2_to_message(img, image_pub)


def callback(image_msg, image_pub, lane_pub, **kwargs):
    # Run the perception pipeline
    lane_detection(image_msg, image_pub, lane_pub)


def shutdown_hook():
    print('\n\033[95m' + '*' * 30 + ' Lane Detection Shutdown ' + '*' * 30 + '\033[00m\n')
    lanenet.close()


def run(**kwargs):
    # Start node
    rospy.init_node('main', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())

    # Wait for main perception to start first
    # rospy.wait_for_message('/delta/perception/object_detection_tracking/image', Image)
    rospy.wait_for_message('/delta/perception/segmentation/image', Image)

    # Setup models
    lanenet.setup()

    # Handle params and topics
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    lane_output = rospy.get_param('~lane_output', '/delta/perception/lane/markings')
    output_image = rospy.get_param('~output_image', '/delta/perception/lane/image')

    # Display params and topics
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('Lane marking topic: %s' % lane_output)
    rospy.loginfo('Output topic: %s' % output_image)

    # Publish output topic
    lane_pub = rospy.Publisher(lane_output, LaneMarkingArray, queue_size=5)
    image_pub = rospy.Publisher(output_image, Image, queue_size=5)

    # Subscribe to topics
    image_sub = message_filters.Subscriber(image_color, Image)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer([image_sub], queue_size=1, slop=0.1)
    ats.registerCallback(callback, image_pub, lane_pub, **kwargs)

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
