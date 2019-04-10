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

# Built-in modules

# External modules
import matplotlib.pyplot as plt

# ROS modules
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from delta_perception.msg import VehicleGroundTruth, VehicleGroundTruthArray

# Local python modules
from utils import *
from sort.sort import Sort
from darknet.darknet_video import YOLO

# Global variables
cmap = plt.get_cmap('tab10')

# Models
yolov3 = YOLO()
tracker = Sort(max_age=50, min_hits=1, use_dlib=True)

# FPS loggers
all_fps = FPSLogger('Pipeline')
yolo_fps = FPSLogger('YOLOv3')
sort_fps = FPSLogger('Tracker')


########################### Functions ###########################


def visualize(img, tracked_targets, detections):
    # Draw visualizations
    # img = YOLO.cvDrawBoxes(detections, img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display tracked targets
    for tracked_target, detection in zip(tracked_targets, detections):
        label, score, bbox = detection
        x1, y1, x2, y2, tracker_id = tracked_target.astype('int')
        color = tuple(map(int, (np.asarray(cmap(tracker_id % 10))[:-1] * 255).astype('uint8')))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, '%s [%d%%] [ID: %d]' % (label.decode('utf-8').title(), score, tracker_id),
            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return img


def perception_pipeline(img, image_pub, vis=True):
    # Log pipeline FPS
    all_fps.lap()

    # Object detection
    yolo_fps.lap()
    detections, frame_resized = yolov3.run(img)
    yolo_fps.tick()

    # Object tracking
    sort_fps.lap()
    dets = np.asarray([[bbox[0], bbox[1], bbox[2], bbox[3], score] for label, score, bbox in detections])
    tracked_targets = tracker.update(dets, img)
    sort_fps.tick()

    # Display FPS logger status
    all_fps.tick()
    sys.stdout.write('\r%s | %s | %s ' % (all_fps.get_log(), yolo_fps.get_log(), sort_fps.get_log()))
    sys.stdout.flush()

    # Visualize and publish image message
    if vis:
        img = visualize(img.copy(), tracked_targets, detections)
        cv2_to_message(img, image_pub)


def callback(image_msg, camera_info, image_pub, **kwargs):
    # Read image message
    img = message_to_cv2(image_msg)
    if img is None:
        print('Error')
        sys.exit(1)

    # Run the perception pipeline
    perception_pipeline(img, image_pub)


def run(**kwargs):
    # Start node
    rospy.init_node('delta_perception', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    
    # Setup models
    yolov3.setup()

    # Handle params and topics
    camera_info = rospy.get_param('~camera_info_topic', '/carla/ego_vehicle/camera/rgb/front/camera_info')
    image_color = rospy.get_param('~image_color_topic', '/carla/ego_vehicle/camera/rgb/front/image_color')
    output_image = rospy.get_param('~output', '/delta_perception/output_image')

    # Display params and topics
    rospy.loginfo('CameraInfo topic: %s' % camera_info)
    rospy.loginfo('Image topic: %s' % image_color)

     # Publish output topic
    image_pub = rospy.Publisher(output_image, Image, queue_size=5)

    # Subscribe to topics
    info_sub = message_filters.Subscriber(camera_info, CameraInfo)
    image_sub = message_filters.Subscriber(image_color, Image)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, info_sub], queue_size=5, slop=0.2)
    ats.registerCallback(callback, image_pub, **kwargs)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    # Start perception node
    run()
