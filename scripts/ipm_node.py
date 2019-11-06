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

# Handle paths and OpenCV import
from init_paths import *

# ROS modules
import tf
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError

# ROS messages
from sensor_msgs.msg import Image
from delta_msgs.msg import LaneMarking, LaneMarkingArray

# Local python modules
from utils import *
from ipm.ipm import InversePerspectiveMapping

# Frames
CAMERA_FRAME = 'ego_vehicle/camera/rgb/front'

# Perception models
ipm = InversePerspectiveMapping()

# FPS loggers
FRAME_COUNT = 0
ipm_fps = FPSLogger('IPM')


########################### Functions ###########################


def process(image_msg, image_pub, vis=True, **kwargs):
    # Read image message
    img = message_to_cv2(image_msg)
    # seg_img = message_to_cv2(seg_msg)

    # Preprocess
    # img = increase_brightness(img)

    # Run segmentation
    ipm_img = ipm.transform_image(img.copy())

    # Extract the lane points
    # lane_points_top, lane_points_bot = [], []
    # for lane in lane_array.lanes:
    #     # Top
    #     lane_point = ipm.transform_points_to_px([lane.xtop, lane.ytop])
    #     lane_points_top.append(tuple(lane_point[:2].astype('int').tolist()))

    #     # Bottom
    #     lane_point = ipm.transform_points_to_px([lane.xbot, lane.ybot])
    #     lane_points_bot.append(tuple(lane_point[:2].astype('int').tolist()))

    # Project segmentation image to IPM
    # seg_img = ipm.transform_image(seg_img)

    # Visualize and publish image message
    if vis:
        # Display lane points
        # for top, bot in zip(lane_points_top, lane_points_bot):
        #     cv2.line(ipm_img, top, bot, (0, 255, 0), 2)

        # Overlay segmentation
        # ipm_img = cv2.addWeighted(ipm_img, 1.0, seg_img, 0.6, 0)

        # Publish the output
        cv2_to_message(ipm_img, image_pub)


def callback(image_msg, image_pub, **kwargs):
    # Run the IPM pipeline
    process(image_msg, image_pub)


def shutdown_hook():
    print('\n\033[95m' + '*' * 30 + ' Inverse Perspective Mapping Shutdown ' + '*' * 30 + '\033[00m\n')


def run(**kwargs):
    # Start node
    rospy.init_node('main', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())

    # Setup models
    # ipm.setup()

    # Handle params and topics
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    # lane_marking = rospy.get_param('~lane_marking', '/delta/perception/lane/markings')
    # segmentation = rospy.get_param('~segmentation', '/delta/perception/segmentation/image')
    output_image = rospy.get_param('~output_image', '/delta/perception/ipm/image')

    # Display params and topics
    rospy.loginfo('Image topic: %s' % image_color)
    # rospy.loginfo('Lane Marking topic: %s' % lane_marking)
    # rospy.loginfo('Segmentation topic: %s' % segmentation)
    rospy.loginfo('Output topic: %s' % output_image)

    # Publish output topic
    image_pub = rospy.Publisher(output_image, Image, queue_size=5)

    # Subscribe to topics
    image_sub = message_filters.Subscriber(image_color, Image)
    # seg_sub = message_filters.Subscriber(segmentation, Image)
    # lane_sub = message_filters.Subscriber(lane_marking, LaneMarkingArray)

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
