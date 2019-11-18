#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 12, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# ROS modules
import rospy

# ROS messages
from std_msgs.msg import Header
from delta_msgs.msg import MarkerArrayStamped
from visualization_msgs.msg import Marker, MarkerArray

# Globals
marker_pub = None
marker_array = None
marker_array_header = Header()


########################### Functions ###########################


def callback(marker, debug=False, **kwargs):
    global marker_array, marker_array_header

    # Skip ego vehicle marker
    if marker.header.frame_id == 'ego_vehicle': return
    
    # New timestamp batch
    if marker_array_header.stamp.nsecs != marker.header.stamp.nsecs:
        # Publish the old marker array
        if marker_array is not None:
            marker_array.header.stamp = marker_array_header.stamp
            marker_pub.publish(marker_array)
            if debug: print('Publishing marker array with stamp: %d and %d vehicles' % (marker_array.header.stamp.nsecs, len(marker_array.markers)))

        # Reset marker array
        marker_array = MarkerArrayStamped()
        marker_array_header.stamp = marker.header.stamp
        if debug: print('\nNew batch:', marker_array_header.stamp.nsecs)
    
    # Add markers to marker array
    marker_array.markers.append(marker)
    if debug: print(marker.header.frame_id, marker.header.stamp.nsecs)


def run(**kwargs):
    global marker_pub

    # Start node
    rospy.init_node('marker_array_publisher', anonymous=True)

    # Handle params and topics
    vehicle_markers = rospy.get_param('~vehicle_markers', '/carla/vehicle_marker')
    vehicle_marker_array = rospy.get_param('~vehicle_marker_array', '/delta/visualization/vehicle_marker_array')

    # Publish marker array
    marker_pub = rospy.Publisher(vehicle_marker_array, MarkerArrayStamped, queue_size=10)

    # Subscribe to topics
    marker_sub = rospy.Subscriber(vehicle_markers, Marker, callback)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    # Start perception node
    run()
