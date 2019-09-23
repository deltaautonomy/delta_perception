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

        # Meshgrid for Gaussians.
        Y = np.linspace(-range_x // 2, range_x // 2, self.info.height)
        X = np.linspace(0, range_y, self.info.width)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array.
        self.meshgrid = np.empty(X.shape + (2,))
        self.meshgrid[:, :, 0] = X
        self.meshgrid[:, :, 1] = Y

    def empty_grid(self):
        return np.zeros((self.info.height, self.info.width), dtype=np.int8)

    def clear(self):
        self.guassian_grid = None
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
        try: grid[i, j] = np.clip(value, 0, 100)
        except IndexError: pass
        return grid

    def normalize(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def place_gaussian(self, mean, cov, value, grid):
        """
        Return the multivariate Gaussian distribution on array meshgrid.
        Meshgrid is an array constructed by packing the meshed arrays of
        variables x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        """
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
