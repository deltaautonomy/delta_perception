#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# External modules
import numpy as np


class LaneKalmanFilter():
    def __init__(self):
        """
        Initializes the kalman filter class 
        """
        self.state_dim = 12
        self.measurement_dim = 6

    def initialize_filter(self, first_call_time, measurement):
        """
        Internal function to initialize the filter
        """
        # Initial state
        m = measurement.flatten()
        self.x = np.c_[m, np.zeros_like(m)].flatten()
        assert len(self.x) == self.state_dim, 'State dim does not match'
        # State transition matrix
        self.F = np.zeros([12, 12])
        # H is how we go from state variable to measurement variable
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        self.H_lane_1 = self.H[:2, :]
        self.H_lane_2 = self.H[2:4, :]
        self.H_lane_3 = self.H[4:, :] 

        # R is the measurement noise        
        self.R = np.eye(6) * 0.5
        # P is the state transition noise
        self.P = np.eye(12) * 10.0
        # Q is the process covariance
        self.last_call_time = first_call_time

    def predict(self, current_time, sigma_acc=8.8):
        T = current_time - self.last_call_time
        G = np.array([0.5 * T ** 2, T, 0.5 * T ** 2, T])
        self.Q = np.matmul(G.T, G) * sigma_acc ** 2
        self.x = np.matmul(self.F,self.x)
        self.P = np.matmul(self.F, np.matmul(self.P, self.F.T)) + self.Q

    def set_F_matrix(self, T):
        self.F = np.array([[1, T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, T, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, T, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, T, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, T, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, T],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return self.F

    def predict_step(self, current_time):
        """
        @param time_step: time since last update has been called 
        return x: predicted state array of shape (12, 1) with values of:
        [y1, vy1, m1, vm1, y2, vy2, m2, vm2, y3, vy3, m3, vm3]
        """
        time_step = current_time - self.last_call_time
        self.set_F_matrix(time_step)
        self.predict(current_time)
        self.last_call_time = current_time

        return self.x

    def update(self, measurement, lane_id):
        """
        This function takes lane measurement and lane_id information
        """
        if lane_id == 1:
            H_curr = self.H_lane_1
            R_curr = self.R[:2, :2]
        if lane_id == 2:
            H_curr = self.H_lane_2
            R_curr = self.R[2:4, 2:4]
        if lane_id == 3:
            H_curr = self.H_lane_3
            R_curr = self.R[4:, 4:]

        Y = measurement - np.matmul(H_curr, self.x)
        covariance_sum = np.matmul(np.matmul(H_curr, self.P), H_curr.T) + R_curr  
        K = np.matmul(np.matmul(self.P, H_curr.T), np.linalg.pinv(covariance_sum))
        self.x = self.x + np.matmul(K, Y)
        KH = np.matmul(K, H_curr)
        self.P = np.matmul((np.eye(KH.shape[0]) - KH), self.P)   

    def update_step(self, z_lane_1=None, z_lane_2=None, z_lane_3=None):
        """
        @param measurements: np array of shape (2, 1) with values of:
        [y1, m1], [y2, m2], [y3, m3]
        return x: updated state array of shape (12, 1) with values of:
        [y1, vy1, m1, vm1, y2, vy2, m2, vm2, y3, vy3, m3, vm3]
        """
        if z_lane_1 is not None:
            self.update(z_lane_1, lane_id=1)
        if z_lane_2 is not None:
            self.update(z_lane_2, lane_id=2)
        if z_lane_3 is not None:
            self.update(z_lane_3, lane_id=3)

        return self.x
