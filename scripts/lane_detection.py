#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle, Apoorv Singh
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019
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
from diagnostic_msgs.msg import DiagnosticArray
from delta_msgs.msg import LaneMarking, LaneMarkingArray

# Local python modules
from utils import *
from erfnet.lane_detection import ERFNetLaneDetector
from ipm.ipm import InversePerspectiveMapping

# Frames
CAMERA_FRAME = 'ego_vehicle/camera/rgb/front'
EGO_VEHICLE_FRAME = 'ego_vehicle'

# Classes
lane_detector = ERFNetLaneDetector()
ipm = InversePerspectiveMapping()
lane_fps = FPSLogger('LaneNet')


########################### Functions ###########################


def publish_diagnostics(publishers):
    msg = DiagnosticArray()
    msg.status.append(make_diagnostics_status('lane_detection', 'perception_', str(lane_fps.fps)))
    publishers['diag_pub'].publish(msg)


def lane_detection(image_msg, publishers, vis=True, **kwargs):
    # Read image message
    img = message_to_cv2(image_msg)
    if img is None: return

    # Lane detection
    lane_fps.lap()
    output, lanes = lane_detector.run(img, rospy.Time.now())
    lane_fps.tick()

    # Publish diagnostics status
    publish_diagnostics(publishers)

    if output is None or lanes is None:
        return

    # Convert to lane marking array
    lane_array = LaneMarkingArray()
    lane_array.header.stamp = image_msg.header.stamp
    lane_array.header.frame_id = EGO_VEHICLE_FRAME
    for slope, intercept in lanes:
        lane = LaneMarking()
        lane.slope, lane.intercept = slope, intercept
        lane_array.lanes.append(lane)

    # Publish the lane data
    publishers['lane_pub'].publish(lane_array)

    # Visualize and publish image message
    cv2_to_message(output, publishers['image_pub'])


def callback(image_msg, publishers, **kwargs):
    # Run the perception pipeline
    lane_detection(image_msg, publishers)


def shutdown_hook():
    print('\n\033[95m' + '*' * 30 + ' Lane Detection Shutdown ' + '*' * 30 + '\033[00m\n')
    lane_detector.close()


def run(**kwargs):
    # Start node
    rospy.init_node('lane_detection', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())

    # Setup models
    lane_detector.setup()

    # Handle params and topics
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    lane_output = rospy.get_param('~lane_output', '/delta/perception/lane_detection/markings')
    output_image = rospy.get_param('~output_image', '/delta/perception/lane_detection/image')
    diagnostics = rospy.get_param('~diagnostics', '/delta/perception/lane_detection/diagnostics')

    # Display params and topics
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('Lane marking topic: %s' % lane_output)
    rospy.loginfo('Output topic: %s' % output_image)

    # Publish output topic
    publishers = {}
    publishers['lane_pub'] = rospy.Publisher(lane_output, LaneMarkingArray, queue_size=5)
    publishers['image_pub'] = rospy.Publisher(output_image, Image, queue_size=5)
    publishers['diag_pub'] = rospy.Publisher(diagnostics, DiagnosticArray, queue_size=5)

    # Subscribe to topics
    image_sub = message_filters.Subscriber(image_color, Image)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer([image_sub], queue_size=1, slop=0.1)
    ats.registerCallback(callback, publishers, **kwargs)

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
