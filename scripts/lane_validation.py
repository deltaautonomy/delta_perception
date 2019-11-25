#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.1
Date    : Nov 25, 2019

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

# ROS messages
from sensor_msgs.msg import Image
from sklearn.cluster import DBSCAN

# Local python modules
from utils import *
from erfnet.lane_validator import LaneValidator
from erfnet.lane_detection import ERFNetLaneDetector
from ipm.ipm import InversePerspectiveMapping

# Global objects
cmap = plt.get_cmap('tab10')
LANE_GT_COLOR = np.array([50, 230, 155])
PRED_COUNTER = 0
GT_COUNTER = 0

# Classes
lane_validator = LaneValidator()
lane_detector = ERFNetLaneDetector()
ipm = InversePerspectiveMapping()
lane_fps = FPSLogger('LaneNet')


########################### Functions ###########################


def callback(image_msg, image_gt_msg, image_pub, **kwargs):
    # Read image message
    global GT_COUNTER, PRED_COUNTER

    # Convert to cv2 images
    img = message_to_cv2(image_msg)
    img_gt = message_to_cv2(image_gt_msg)

    # Get slope intercepts of lanes
    output, lanes_det = lane_detector.run(img, rospy.Time.now())
    output, lanes_gt = validator.detect_lines(img, rospy.Time.now())

    # Convert to points
    points_det = validator.slope_intercept_to_points(lanes_det)
    points_gt = validator.slope_intercept_to_points(lanes_gt)

    # Computer error
    PRED_COUNTER += np.sum(np.int64(np.abs(points_det - points_gt)[:, 1] < 0.5))
    GT_COUNTER += 9

    cv2_to_message(img_gt, image_pub)


def shutdown_hook():
    print('\n\033[95m' + '*' * 30 + ' Lane Validation Shutdown ' + '*' * 30 + '\033[00m\n')
    lane_detector.close()
    # print('Lane Detection Accuracy: %.2f%%\n' % (PRED_COUNTER * 100.0 / GT_COUNTER))


def run(**kwargs):
    # Start node
    rospy.init_node('lane_validation', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    
    # Setup models
    lane_detector.setup()

    # Handle params and topics
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    image_gt = rospy.get_param('~image_gt', '/carla/ego_vehicle/camera/semantic_segmentation/front/image_segmentation')
    output_image = rospy.get_param('~output_image', '/delta/perception/lane_validation/image')

    # Publish output topic
    image_pub = rospy.Publisher(output_image, Image, queue_size=5)

    # Subscribe to topics
    image_sub = message_filters.Subscriber(image_color, Image)
    image_gt_sub = message_filters.Subscriber(image_gt, Image)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, image_gt_sub], queue_size=1, slop=0.5)
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
