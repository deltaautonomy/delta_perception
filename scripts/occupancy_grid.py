#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Sep 10, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Handle paths and OpenCV import
from init_paths import *

# ROS modules
from ros_numpy.occupancy_grid import numpy_to_occupancy_grid

# ROS messages
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose

# Local python modules
from utils import *


class DeltaOccupancyGrid(object):
    
    def __init__(self, range_x, range_y, frame_id, resolution=1.5):
        self.info = MapMetaData()
        self.info.resolution = resolution  # Meters
        self.info.height = int(range_x / self.info.resolution)  # Meters
        self.info.width = int(range_y / self.info.resolution)  # Meters
        self.x_offset = range_x / 2  # Center left-right about ego vehicle
        self.info.origin = numpy_to_position([0, -self.x_offset, -0.5], Pose())

        self.occupancy_grid = OccupancyGrid()
        self.occupancy_grid.info = self.info
        self.occupancy_grid.header.frame_id = frame_id

        self.bins_x = np.linspace(-range_x / 2, range_x / 2, self.info.height)
        self.bins_y = np.linspace(0, range_y, self.info.width)

    def empty_grid(self):
        return np.zeros((self.info.height, self.info.width), dtype=np.int8)

    def clear(self):
        self.occupancy_grid.data = self.empty_grid().ravel()

    def update(self, grid, timestamp):
        self.occupancy_grid.header.stamp = timestamp
        self.occupancy_grid.data = grid.ravel()
        return self.occupancy_grid

    def refresh(self, grid, timestamp):
        self.clear()
        self.update(grid, timestamp)
        return self.occupancy_grid

    def place(self, position, value, grid):
        i = np.digitize(position[1], self.bins_x)
        j = np.digitize(position[0], self.bins_y)
        try: grid[i, j] = value
        except IndexError: pass
        return grid
