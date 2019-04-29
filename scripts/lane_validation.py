#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
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
from utils import *

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
from lanenet.lanenet import LaneNetModel

# Global objects
cmap = plt.get_cmap('tab10')
LANE_GT_COLOR = np.array([50, 230, 155])
PRED_COUNTER = 0
GT_COUNTER = 0
DELTA_FAR = 0.04 * 1280
DELTA_NEAR = 0.06 * 1280

# Perception models
lanenet = LaneNetModel()


########################### Functions ###########################


def perception_callback(image_msg, image_gt_msg, image_pub, **kwargs):
    # Read image message
    global GT_COUNTER, PRED_COUNTER

    # Convert to cv2 images
    img = message_to_cv2(image_msg)
    img_gt = message_to_cv2(image_gt_msg)

    # Rescale to camera image size (720 * 1280)
    img_gt = cv2.resize(img_gt, (0, 0), fx=1.6, fy=1.6)
    img_gt = img_gt[120:840, :, :]

    # Segment out lanes
    image1 = np.where(img_gt >= LANE_GT_COLOR - 10, 255, 0)
    image2 = np.where(img_gt <= LANE_GT_COLOR + 10, 255, 0)
    img_gt = (image1 & image2).astype('uint8')

    # Make the lines thick
    kernel = np.ones((7, 7), np.uint8)
    img_gt_lane = cv2.dilate(img_gt, kernel, iterations=3)
    ret, thresh = cv2.threshold(cv2.cvtColor(img_gt_lane, cv2.COLOR_BGR2GRAY), 127, 255, 0)

    # Find contours
    thresh[:200] = 0
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Keep only max 3 contours
    args = np.argsort([cv2.contourArea(cnt) for cnt in contours])[::-1]
    if len(args) < 3:
        print('Less than 3 lanes in ground truth visible')
        return
    contours = [contours[arg] for arg in args[:3]]

    # Sort them left to right
    args = np.argsort([int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00']) for cnt in contours])
    contours = [contours[arg] for arg in args[:3]]
    
    # Run lane detection
    lane_img, _, points = lanenet.run(img)
    if len(points) < 3:
        print('Less than 3 lanes in detected lanes visible')
        return

    # Fit lines on contours
    rows, cols = img_gt_lane.shape[:2]
    fy = img.shape[0] / float(lane_img.shape[0])
    fx = img.shape[1] / float(lane_img.shape[1])
    
    gt_points = []
    for point, cnt in zip(points, contours):
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)

        # Validation line 1
        gtx = int((((point[1] * fy) - y) * vx/vy) + x)
        gt_points.append([gtx, int(point[1] * fy)])
        cv2.circle(img, (int(gtx), int(point[1] * fy)), 5, (0, 255, 0), -1)

        # Validation line 2
        gtx = int((((point[3] * fy) - y) * vx/vy) + x)
        gt_points.append([gtx, int(point[3] * fy)])
        cv2.circle(img, (int(gtx), int(point[3] * fy)), 5, (0, 255, 0), -1)

    # Sort by top and bottom
    args = [1, 3, 5, 0, 2, 4]
    gt_points = np.asarray(gt_points)[args]

    # Scale predicted points
    points[:, 0] = points[:, 0] * fx
    points[:, 1] = points[:, 1] * fy
    points[:, 2] = points[:, 2] * fx
    points[:, 3] = points[:, 3] * fy
    points = points.reshape(-1, 2)[args]

    # Compute detections
    errors = np.abs(points[:, 0] - gt_points[:, 0])
    GT_COUNTER += 3
    if errors[0] < DELTA_FAR and errors[3] < DELTA_NEAR: PRED_COUNTER += 1
    if errors[1] < DELTA_FAR and errors[4] < DELTA_NEAR: PRED_COUNTER += 1
    if errors[2] < DELTA_FAR and errors[5] < DELTA_NEAR: PRED_COUNTER += 1

    # Display the detections
    for point in points: cv2.circle(img, tuple(point.tolist()), 5, (0, 0, 255), -1)
    cv2_to_message(img, image_pub)


def shutdown_hook():
    print('\n\033[95m' + '*' * 30 + ' Lane Validation Shutdown ' + '*' * 30 + '\033[00m\n')
    lanenet.close()
    print('Lane Detection Accuracy: %.2f%%\n' % (PRED_COUNTER * 100.0 / GT_COUNTER))


def run(**kwargs):
    # Start node
    rospy.init_node('main', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    
    # Setup models
    lanenet.setup()

    # Handle params and topics
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    image_gt = rospy.get_param('~image_gt', '/carla/ego_vehicle/camera/semantic_segmentation/front/image_segmentation')
    output_image = rospy.get_param('~output_image', '/delta/perception/lane_validation')

    # Publish output topic
    image_pub = rospy.Publisher(output_image, Image, queue_size=5)

    # Subscribe to topics
    image_sub = message_filters.Subscriber(image_color, Image)
    image_gt_sub = message_filters.Subscriber(image_gt, Image)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, image_gt_sub], queue_size=1, slop=0.5)
    ats.registerCallback(perception_callback, image_pub, **kwargs)

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
