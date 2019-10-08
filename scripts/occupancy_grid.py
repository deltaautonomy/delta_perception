#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.1.0
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


class OccupancyGridGenerator(object):
    def __init__(self, range_x, range_y, frame_id, resolution=1.5):
        '''
        Set up the occupancy grid generator object.

        Args:
        - range_x (float): Total range of the occupancy grid width
                           (the x-origin is at the center of this range).
        - range_y (float): Total range of the occupancy grid height
                           (the y-origin is 0).
        - frame_id (str): The ROS frame ID at which the occupancy grid origin is aligned with.
        - resolution (float): The size of each cell in the occupancy grid (in meters).

        Return: None.
        '''
        self.info = MapMetaData()
        self.info.resolution = resolution  # Meters
        self.info.height = int(range_x / self.info.resolution)  # Meters
        self.info.width = int(range_y / self.info.resolution)  # Meters
        self.x_offset = range_x / 2  # Center left-right about ego vehicle
        self.info.origin = self.numpy_to_position([0, -self.x_offset, -0.5], Pose())

        self.occupancy_grid = OccupancyGrid()
        self.occupancy_grid.info = self.info
        self.occupancy_grid.header.frame_id = frame_id

        self.bins_x = np.linspace(-range_x / 2, range_x / 2, self.info.height)
        self.bins_y = np.linspace(0, range_y, self.info.width)

        # Meshgrid for Gaussians.
        Y = np.linspace(-range_x // 2, range_x // 2, self.info.height)
        X = np.linspace(0, range_y, self.info.width)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array.
        self.meshgrid = np.empty(X.shape + (2,))
        self.meshgrid[:, :, 0] = X
        self.meshgrid[:, :, 1] = Y

    def numpy_to_position(self, numpy_pos, position):
        '''
        Converts a numpy array to geometry_msgs.msg.Pose object.

        Args:
        - numpy_pos (ndarray): A (3,) numpy array.
        - position (geometry_msgs.msg.Pose): Destination pose object.

        Return:
        - position (geometry_msgs.msg.Pose): Destination pose object.
        '''
        position.position.x = numpy_pos[0]
        position.position.y = numpy_pos[1]
        position.position.z = numpy_pos[2]
        return position

    def empty_grid(self):
        '''
        Returns and empty numpy array representing the data that goes on
        to the occupancy grid.

        Args: None.

        Return:
        - grid (ndarray): A numpy array where each element represents a cell of the
                          occupancy grid. The dimensions depends on the range and the
                          resolution of the occupancy grid.
        '''
        return np.zeros((self.info.height, self.info.width), dtype=np.int8)

    def clear(self):
        '''
        Clears the internal grid data.
        This function is for private usage, use the refresh() method for public usage.

        Args: None.
        Return: None.
        '''
        self.guassian_grid = None
        self.occupancy_grid.data = self.empty_grid().ravel()

    def update(self, grid, timestamp):
        '''
        Updates the internal grid data and the header and returns a
        nav_msgs.OccupancyGrid message ready for publishing.
        This function is for private usage, use the refresh() method for public usage.

        Args:
        - grid (ndarray): A grid (numpy array) generated with the help of this class
                           (see OccupancyGridGenerator.empty_grid() above).
        - timestamp (rospy.Time): Timestamp for the message header.

        Return:
        - occupancy_grid (nav_msgs.OccupancyGrid) - OccupancyGrid message ready to be published.
        '''
        self.occupancy_grid.header.stamp = timestamp
        self.occupancy_grid.data = grid.ravel()
        return self.occupancy_grid

    def refresh(self, grid, timestamp):
        '''
        This functions clears all the internal grid data, updates the internal grid
        data and returns a nav_msgs.OccupancyGrid message ready for publishing.

        Args:
        - grid (ndarray): A grid (numpy array) generated with the help of this class
                           (see OccupancyGridGenerator.empty_grid() above).
        - timestamp (rospy.Time): Timestamp for the message header.

        Return:
        - occupancy_grid (nav_msgs.OccupancyGrid) - OccupancyGrid message ready to be published.
        '''
        self.clear()
        self.update(grid, timestamp)
        return self.occupancy_grid

    def place(self, position, value, grid):
        '''
        To place or update a single cell in the occupancy grid.

        Args:
        - position (ndarray): (2,) numpy array which is the position (in meters) of the object.
        - value (int8): The intensity value (0-100) of the cell in the grid.
        - grid (ndarray): A grid (numpy array) generated with the help of this class
                           (see OccupancyGridGenerator.empty_grid() above).

        Return:
        - grid (ndarray): A numpy array where each element represents a cell of the
                          occupancy grid. The dimensions depends on the range and the
                          resolution of the occupancy grid.
        '''
        i = np.digitize(position[1], self.bins_x)
        j = np.digitize(position[0], self.bins_y)
        try: grid[i, j] = np.clip(value, 0, 100)
        except IndexError: pass
        return grid

    def normalize(self, x):
        '''
        Normalizes a given array to the range (0-1).

        Args:
        - x (ndarray): Input numpy array.

        Return:
        - x (ndarray): Normalized numpy array of the same dimensions.
        '''
        if np.max(x) == np.min(x): return x
        return (x - np.min(x)) / (np.max(x) - np.min(x))
        # return cv2.normalize(x, None, 0, 100, cv2.NORM_MINMAX, cv2.CV_8U)

    def place_gaussian(self, mean, cov, value, grid):
        '''
        Places a multivariate Gaussian distribution on array meshgrid.
        Meshgrid is an array constructed by packing the meshed arrays of
        variables x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        Args:
        - mean (ndarray): (2,) numpy array which is the mean position (in meters) of the object.
        - cov (ndarray): (2, 2) numpy array which is the covariance matrix (in meters) of the object.
        - value (int8): The intensity value (0-100) of the cell in the grid.
        - grid (ndarray): A grid (numpy array) generated with the help of this class
                           (see OccupancyGridGenerator.empty_grid() above).

        Return:
        - grid (ndarray): A numpy array where each element represents a cell of the
                          occupancy grid. The dimensions depends on the range and the
                          resolution of the occupancy grid.
        '''
        mean, cov = np.flip(mean), np.flip(cov)
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        N = np.sqrt((2 * np.pi) ** mean.shape[0] * cov_det)
        diff = self.meshgrid - mean
        fac = np.einsum('...k,kl,...l->...', diff, cov_inv, diff)
        output = self.normalize(np.exp(-fac / 2) / N) * np.clip(value, 0, 100)
        grid = np.clip(grid + output, 0, 100)
        return grid


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    occupancy_grid = OccupancyGridGenerator(30, 100, 'ego_vehicle', 0.1)
    grid = occupancy_grid.empty_grid()
    grid = occupancy_grid.place([5, 5], 50, grid)
    grid = occupancy_grid.place_gaussian([-10, 5], np.eye(2), 120, grid)
    plt.imshow(grid)
    plt.show()
