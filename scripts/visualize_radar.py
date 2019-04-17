#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 17, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Handle paths and OpenCV import
from init_paths import *

# ROS modules
import tf
import rospy
import message_filters

# ROS messages
from sensor_msgs.msg import Image, CameraInfo
from radar_msgs.msg import RadarTrack, RadarTrackArray

# Local python modules
from utils import *

# Global variables
CAMERA_INFO = None
CAMERA_FRAME = 'ego_vehicle/camera/rgb/front'
EGO_VEHICLE_FRAME = 'ego_vehicle'
CAMERA_EXTRINSICS = None
CAMERA_PROJECTION_MATRIX = None


########################### Functions ###########################


def camera_info_callback(camera_info):
    global CAMERA_INFO, CAMERA_PROJECTION_MATRIX
    if CAMERA_INFO is None:
        CAMERA_INFO = camera_info
        CAMERA_PROJECTION_MATRIX = np.matmul(np.asarray(CAMERA_INFO.P).reshape(3, 4), CAMERA_EXTRINSICS)


def callback(image_msg, radar_msg, image_pub, **kwargs):
    # Read image message
    img = message_to_cv2(image_msg)
    if img is None:
        print('Error')
        sys.exit(1)

    # Project the radar points on image
    for track in radar_msg.tracks:
        if CAMERA_PROJECTION_MATRIX is not None:
            pos_msg = position_to_numpy(track.track_shape.points[0])
            pos = np.asarray([pos_msg[1], -pos_msg[0], 0])
            print(pos)
            pos = np.matrix(np.append(pos, 1)).T
            uv = np.matmul(CAMERA_PROJECTION_MATRIX, pos)
            uv = uv / uv[-1]
            uv = uv[:2].astype('int').tolist()
            uv = np.asarray(uv).flatten().tolist()
            cv2.circle(img, tuple(uv), 10, (0, 0, 255), -1)

    # Display image
    cv2_to_message(img, image_pub)


def run(**kwargs):
    global CAMERA_EXTRINSICS 

    # Start node
    rospy.init_node('visualize_radar', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    tf_listener = tf.TransformListener()

    # Find the camera to vehicle extrinsics
    tf_listener.waitForTransform(CAMERA_FRAME, EGO_VEHICLE_FRAME, rospy.Time(), rospy.Duration(4.0))
    (trans, rot) = tf_listener.lookupTransform(CAMERA_FRAME, EGO_VEHICLE_FRAME, rospy.Time(0))
    CAMERA_EXTRINSICS = pose_to_transformation(position=trans, orientation=rot)

    # Handle params and topics
    camera_info = rospy.get_param('~camera_info', '/carla/ego_vehicle/camera/rgb/front/camera_info')
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    radar = rospy.get_param('~radar', '/delta/radar/tracks')
    output_image = rospy.get_param('~output_image', '/delta_perception/camera_radar_image')

    # Display params and topics
    rospy.loginfo('CameraInfo topic: %s' % camera_info)
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('RADAR topic: %s' % radar)

    # Publish output topic
    image_pub = rospy.Publisher(output_image, Image, queue_size=5)

    # Subscribe to topics
    info_sub = rospy.Subscriber(camera_info, CameraInfo, camera_info_callback)
    image_sub = message_filters.Subscriber(image_color, Image)
    radar_sub = message_filters.Subscriber(radar, RadarTrackArray)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, radar_sub], queue_size=1, slop=0.1)
    ats.registerCallback(callback, image_pub, **kwargs)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    # Start perception node
    run()
