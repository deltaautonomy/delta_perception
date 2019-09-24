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
from nav_msgs.msg import OccupancyGrid
from delta_perception.msg import LaneMarking, LaneMarkingArray

# Local python modules
from utils import *
from erfnet.lane_detection import ERFNetLaneDetector
from ipm.ipm import InversePerspectiveMapping
from occupancy_grid import OccupancyGridGenerator

# Frames
CAMERA_FRAME = 'ego_vehicle/camera/rgb/front'
EGO_VEHICLE_FRAME = 'ego_vehicle'

# Perception models
lane_detector = ERFNetLaneDetector()
ipm = InversePerspectiveMapping()
occupancy_grid = OccupancyGridGenerator(30, 100, EGO_VEHICLE_FRAME, resolution=0.2)

# FPS loggers
FRAME_COUNT = 0
lane_fps = FPSLogger('LaneNet')


########################### Functions ###########################


def lane_detection(image_msg, publishers, vis=True, **kwargs):
    # Read image message
    img = message_to_cv2(image_msg)
    if img is None: return

    # Lane detection
    lane_fps.lap()
    output, medians = lane_detector.run(img, rospy.Time.now())
    lane_fps.tick()

    # # Fit lines on contours
    # rows, cols = img.shape[:2]
    # fy = img.shape[0] / float(lane_img.shape[0])
    # fx = img.shape[1] / float(lane_img.shape[1])

    # # Scale predicted points
    # points[:, 0] = points[:, 0] * fx
    # points[:, 1] = points[:, 1] * fy
    # points[:, 2] = points[:, 2] * fx
    # points[:, 3] = points[:, 3] * fy

    # # Convert to lane marking array
    # lane_array = LaneMarkingArray()
    # lane_array.header.stamp = image_msg.header.stamp
    # lane_array.header.frame_id = CAMERA_FRAME
    # for xbot, ybot, xtop, ytop in points:
    #     lane = LaneMarking()
    #     lane.xtop, lane.ytop = xtop, ytop
    #     lane.xbot, lane.ybot = xbot, ybot
    #     lane_array.lanes.append(lane)

    # # Publish the lane data
    # publishers['lane_pub'].publish(lane_array)

    # Convert lane map image to x, y points
    # lane_img = cv2.resize(lane_img, (img.shape[1], img.shape[0]))
    # lane_map = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)
    # v, u = np.where(lane_map > 0)
    # uv_points = np.stack((u, v)).T
    # points_m = ipm.transform_points_to_m(uv_points)

    # # Convert x, y points to occupancy grid
    # grid = occupancy_grid.empty_grid()
    # for x, y in points_m: grid = occupancy_grid.place([y, -x], 100, grid)
    # grid_msg = occupancy_grid.refresh(grid, image_msg.header.stamp)
    # publishers['occupancy_grid_pub'].publish(grid_msg)

    # Visualize and publish image message
    # if vis:
    #     # Display lane markings
    #     overlay = cv2.resize(lane_img, (img.shape[1], img.shape[0]))
    #     img = cv2.addWeighted(img, 1.0, overlay, 1.0, 0)
    
    # ipm_img = ipm.transform_image(img)
    # output = lane_detector.hough_line_detector(lane_img, ipm_img)
    cv2_to_message(output, publishers['image_pub'])


def callback(image_msg, publishers, **kwargs):
    # Run the perception pipeline
    lane_detection(image_msg, publishers)


def shutdown_hook():
    print('\n\033[95m' + '*' * 30 + ' Lane Detection Shutdown ' + '*' * 30 + '\033[00m\n')
    lane_detector.close()


def run(**kwargs):
    # Start node
    rospy.init_node('main', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())

    # Wait for main perception to start first
    # rospy.wait_for_message('/delta/perception/object_detection_tracking/image', Image)
    # rospy.wait_for_message('/delta/perception/segmentation/image', Image)

    # Setup models
    lane_detector.setup()

    # Handle params and topics
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    lane_output = rospy.get_param('~lane_output', '/delta/perception/lane_detection/markings')
    output_image = rospy.get_param('~output_image', '/delta/perception/lane_detection/image')
    occupancy_grid_topic = rospy.get_param('~occupancy_grid', '/delta/perception/lane_detection/occupancy_grid')

    # Display params and topics
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('Lane marking topic: %s' % lane_output)
    rospy.loginfo('Output topic: %s' % output_image)
    rospy.loginfo('OccupancyGrid topic: %s' % occupancy_grid_topic)

    # Publish output topic
    publishers = {}
    publishers['lane_pub'] = rospy.Publisher(lane_output, LaneMarkingArray, queue_size=5)
    publishers['image_pub'] = rospy.Publisher(output_image, Image, queue_size=5)
    publishers['occupancy_grid_pub'] = rospy.Publisher(occupancy_grid_topic, OccupancyGrid, queue_size=5)

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
