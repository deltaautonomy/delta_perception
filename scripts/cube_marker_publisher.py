#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 13, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# ROS modules
import rospy

# ROS messages
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from jsk_rviz_plugins.msg import Pictogram


########################### Functions ###########################


def make_label(text, position, frame_id='/map', marker_id=0,
    duration=0.5, color=[1.0, 1.0, 1.0]):
    """ 
    Helper function for generating visualization markers
    
    Args:
        text (str): Text string to be displayed.
        position (list): List containing [x,y,z] positions
        frame_id (str): ROS TF frame id
        marker_id (int): Integer identifying the label
        duration (rospy.Duration): How long the label will be displayed for
        color (list): List of label color floats from 0 to 1 [r,g,b]
    
    Returns: 
        Marker: A text view marker which can be published to RViz
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.id = marker_id
    marker.type = marker.TEXT_VIEW_FACING
    marker.text = text
    marker.action = marker.ADD
    marker.scale.x = 1.5
    marker.scale.y = 1.5
    marker.scale.z = 1.5
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.lifetime = rospy.Duration(duration)
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    return marker


def make_pictogram(character, position, frame_id='/map',
    duration=0.5, color=[1.0, 1.0, 1.0]):
    """ 
    Helper function for generating visualization markers
    
    Args:
        character (str): Character (icon) to be displayed.
        position (list): List containing [x,y,z] positions
        frame_id (str): ROS TF frame id
        duration (rospy.Duration): How long the label will be displayed for
        color (list): List of label color floats from 0 to 1 [r,g,b]
    
    Returns: 
        Pictogram: A jsk_rviz_plugins/Pictogram message which can be published to RViz
    """
    msg = Pictogram()
    msg.action = Pictogram.ADD
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    msg.mode = Pictogram.PICTOGRAM_MODE
    msg.character = character
    msg.speed = 1.0
    msg.ttl = duration
    msg.size = 3
    msg.color.r = color[0]
    msg.color.g = color[1]
    msg.color.b = color[2]
    msg.color.a = 1.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = -1.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0
    msg.pose.position.x = position[0]
    msg.pose.position.y = position[1]
    msg.pose.position.z = position[2]
    return msg


def make_trajectory(trajectory, frame_id='/map', marker_id=0,
    duration=0.5, color=[1.0, 1.0, 1.0]):
    """ 
    Helper function for generating visualization markers
    
    Args:
        trajectory (array-like): (n, 2) array-like trajectory data
        frame_id (str): ROS TF frame id
        marker_id (int): Integer identifying the trajectory
        duration (rospy.Duration): How long the trajectory will be displayed for
        color (list): List of color floats from 0 to 1 [r,g,b]
    
    Returns: 
        Marker: A trajectory marker message which can be published to RViz
    """
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = frame_id
    marker.id = marker_id
    marker.type = marker.LINE_STRIP
    marker.action = marker.ADD
    for x, y in trajectory:
        point = Point()
        point.x = x
        point.y = y
        point.z = 0.5
        marker.points.append(point)    
    marker.scale.x = 0.15
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration(duration)
    return marker


def make_cuboid(position, scale, frame_id='/map', marker_id=0,
    duration=0, color=[1.0, 1.0, 1.0]):
    """ 
    Helper function for generating visualization markers
    
    Args:
        position (list): List containing [x, y, z] positions
        scale (list): List containing [x, y, z] dimensions
        frame_id (str): ROS TF frame id
        marker_id (int): Integer identifying the label
        duration (rospy.Duration): How long the label will be displayed for
        color (list): List of label color floats from 0 to 1 [r, g, b]
    
    Returns: 
        Marker: A cube marker which can be published to RViz
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.id = marker_id
    marker.type = marker.CUBE
    marker.text = str(marker_id)
    marker.action = marker.ADD
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration(duration)
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    return marker


def publisher():
    # Setup node
    rospy.init_node('marker_publisher', anonymous=True)
    pub = rospy.Publisher('marker_publisher', Marker, queue_size=10)
    
    # Publish rate
    r = rospy.Rate(0.25)

    # Randomly publish some data
    while not rospy.is_shutdown():
        # Create the message array
        msg = make_cuboid([0, 0, 0], [0.05, 0.05, 0.05])
        
        # Header stamp and publish the message
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)

        # Sleep
        r.sleep()


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
