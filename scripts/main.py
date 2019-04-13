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
import tf
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError

# ROS messages
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from delta_perception.msg import MarkerArrayStamped
from derived_object_msgs.msg import Object, ObjectArray

# Local python modules
from utils import *
from sort.sort import Sort
from darknet.darknet_video import YOLO

# Global objects
cmap = plt.get_cmap('tab10')
tf_listener = None

# Global variables
CAMERA_INFO = None
CAMERA_FRAME = 'ego_vehicle/camera/rgb/front'
VEHICLE_FRAME = 'vehicle/%03d/autopilot'

# Perception models
# yolov3 = YOLO()
# tracker = Sort(max_age=200, min_hits=1, use_dlib=False)
# tracker = Sort(max_age=20, min_hits=1, use_dlib=True)

# FPS loggers
all_fps = FPSLogger('Pipeline')
yolo_fps = FPSLogger('YOLOv3')
sort_fps = FPSLogger('Tracker')


########################### Functions ###########################


def camera_info_callback(camera_info):
    global CAMERA_INFO
    if CAMERA_INFO is None:
        CAMERA_INFO = camera_info


def validator(image, objects, image_pub, **kwargs):
    # Check if camera info is available
    if CAMERA_INFO is None: return

    for obj in objects.objects:
        try:
            # Find the camera to vehicle extrinsics
            (trans, rot) = tf_listener.lookupTransform(CAMERA_FRAME, VEHICLE_FRAME % obj.id, rospy.Time(0))
            # print(trans, np.rad2deg(quaternion_to_rpy(rot)))
            camera_to_vehicle = pose_to_transformation(position=trans, orientation=rot)

            # Project 3D to 2D and filter bbox within image boundaries
            M = np.matrix(CAMERA_INFO.P).reshape(3, 4)
            bbox3D = get_bbox_vertices(camera_to_vehicle, obj.shape.dimensions)
            bbox2D = np.matmul(M, bbox3D.T).T
            
            # Ignore vehicles behind the camera view
            if bbox2D[0, 2] < 0: continue
            bbox2D = bbox2D / bbox2D[:, -1]
            bbox2D = bbox2D[:, :2].astype('int')#.tolist()

            # Display the 3D bbox vertices on image
            for point in bbox2D.tolist():
                # print(point)
                cv2.circle(image, tuple(point), 2, (255, 255, 0), -1)
            
            # Find the 2D bounding box coordinates
            top_left = (np.min(bbox2D[:, 0]), np.min(bbox2D[:, 1]))
            bot_right = (np.max(bbox2D[:, 0]), np.max(bbox2D[:, 1]))

            # Draw the rectangle
            cv2.rectangle(image, top_left, bot_right, (0, 255, 0), 1)
            cv2.putText(image, 'ID: %d [%.2fm]' % (obj.id, trans[2]), top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2_to_message(image, image_pub)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print(e)


def visualize(img, tracked_targets, detections, **kwargs):
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


def perception_pipeline(img, image_pub, vis=True, **kwargs):
    # Log pipeline FPS
    all_fps.lap()

    # Preprocess
    img = increase_brightness(img)

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


def perception_callback(image_msg, objects, image_pub, **kwargs):
    # Read image message
    img = message_to_cv2(image_msg)
    if img is None:
        print('Error')
        sys.exit(1)

    # Run the perception pipeline
    # perception_pipeline(img, image_pub)
    validator(img, objects, image_pub)


def run(**kwargs):
    global tf_listener

    # Start node
    rospy.init_node('main', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    tf_listener = tf.TransformListener()
    
    # Setup models
    # yolov3.setup()

    # Handle params and topics
    camera_info = rospy.get_param('~camera_info', '/carla/ego_vehicle/camera/rgb/front/camera_info')
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    object_array = rospy.get_param('~object_array', '/carla/objects')
    # vehicle_markers = rospy.get_param('~vehicle_markers', '/carla/vehicle_marker_array')
    output_image = rospy.get_param('~output_image', '/delta_perception/output_image')

    # Display params and topics
    rospy.loginfo('CameraInfo topic: %s' % camera_info)
    rospy.loginfo('Image topic: %s' % image_color)

     # Publish output topic
    image_pub = rospy.Publisher(output_image, Image, queue_size=5)

    # Subscribe to topics
    info_sub = rospy.Subscriber(camera_info, CameraInfo, camera_info_callback)
    image_sub = message_filters.Subscriber(image_color, Image)
    object_sub = message_filters.Subscriber(object_array, ObjectArray)
    # marker_sub = message_filters.Subscriber(vehicle_markers, MarkerArrayStamped)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, object_sub], queue_size=1, slop=0.1)
    ats.registerCallback(perception_callback, image_pub, **kwargs)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    # Start perception node
    run()
