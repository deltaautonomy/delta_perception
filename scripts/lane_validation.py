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
import PIL
from cStringIO import StringIO
import matplotlib
matplotlib.use('Agg')
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


def plot(points_det, points_gt):
    fig = plt.figure()

    plt.plot(points_gt[0:3, 0], points_gt[0:3, 1], '-r', linewidth=2.0, label='Ground Truth')
    plt.plot(points_gt[3:6, 0], points_gt[3:6, 1], '-r', linewidth=2.0, label='Ground Truth')
    plt.plot(points_gt[6:9, 0], points_gt[6:9, 1], '-r', linewidth=2.0, label='Ground Truth')

    plt.plot(points_det[0:3, 0], points_det[0:3, 1], '-g', linewidth=2.0, label='Detection')
    plt.plot(points_det[3:6, 0], points_det[3:6, 1], '-g', linewidth=2.0, label='Detection')
    plt.plot(points_det[6:9, 0], points_det[6:9, 1], '-g', linewidth=2.0, label='Detection')

    plt.legend()
    plt.grid()

    # convert canvas to image
    # plt.imshow(np.random.random((20,20)))
    buffer_ = StringIO()
    plt.savefig(buffer_, format="png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    graph_image = np.asarray(image)
    plt.close(fig)
    return graph_image


def callback(image_msg, image_gt_msg, image_pub, **kwargs):
    # Read image message
    global GT_COUNTER, PRED_COUNTER

    # Convert to cv2 images
    img = message_to_cv2(image_msg)
    img_gt = message_to_cv2(image_gt_msg)

    # Get slope intercepts of lanes
    output_det, lanes_det = lane_detector.run(img, rospy.Time.now())
    if output_det is None: return
    output_gt, lanes_gt = lane_validator.detect_lines(img_gt, rospy.Time.now())
    if output_gt is None: return

    # Convert to points
    points_det = lane_validator.slope_intercept_to_points(lanes_det)
    points_gt = lane_validator.slope_intercept_to_points(lanes_gt)

    # Plot
    output_image = plot(points_det, points_gt)
    # output_image = np.concatenate((output_det, output_gt), axis=1)

    # Computer error
    PRED_COUNTER += np.sum(np.int64(np.abs(points_det - points_gt)[:, 1] < 0.65))
    GT_COUNTER += 9
    text = '\r%03d / %03d | Accuracy %.2f%%  ' % (PRED_COUNTER, GT_COUNTER, PRED_COUNTER / GT_COUNTER)
    sys.stdout.write(text)
    sys.stdout.flush()

    # cv2_to_message(output, image_pub)
    cv2_to_message(output_image, image_pub, coding='bgra8')


def shutdown_hook():
    print('\n\033[95m' + '*' * 30 + ' Lane Validation Shutdown ' + '*' * 30 + '\033[00m\n')
    lane_detector.close()
    print('Lane Detection Accuracy: %.2f%%\n' % (PRED_COUNTER * 100.0 / GT_COUNTER))


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
