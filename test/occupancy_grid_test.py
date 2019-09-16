#!/usr/bin/env python

import rospy
import ros_numpy
from nav_msgs.msg import OccupancyGrid, MapMetaData

import random
import numpy as np

FRAME = 'ego_vehicle'

def demo():
    pub = rospy.Publisher("/occupancy_grid", OccupancyGrid, queue_size=1)
    rate = rospy.Rate(1)

    grid = np.ones((20, 20), dtype=np.int8)
    
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        occupancy_grid = ros_numpy.occupancy_grid.numpy_to_occupancy_grid(grid)
        occupancy_grid.info.resolution = 0.1
        occupancy_grid.header.stamp = now
        occupancy_grid.header.frame_id = FRAME
        pub.publish(occupancy_grid)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("occupancy_grid_test")
    demo()
