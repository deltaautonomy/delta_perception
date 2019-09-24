# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

import numpy as np

class LaneKalmanFilter():
    def __init__(self, state_dim=12, measurement_dim=6, motion_model='velocity'):
        """
        Initializes the kalman filter class 
        @param: state_dim - number of states
        @param: measurement_dim - number of observations returned by erfnet
        @param: dt - timestep at which the vehicle state should update
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.motion_model = motion_model
    
    def initialize_filter(self, first_call_time, measurement):
        """
        Internal function to initialize the filter
        """
        # Initial state
        m = measurement.flatten()
        self.x = np.c_[m, np.zeros_like(m)].flatten()
        assert len(self.x) == self.state_dim, 'State dim does not match'
        # state transition matrix
        self.F = np.zeros([12, 12])
        # H is how we go from state variable to measurement variable
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        # R is the measurement noise        
        self.R = np.eye(6) * 100.0
        # P is the state transition noise
        self.P = np.eye(12) * 500.0
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
        y1, Vy1, m1, Vm1, y2, Vy2, m2, Vm2, y3, Vy3, m3, Vm3
        """
        time_step = current_time - self.last_call_time
        self.set_F_matrix(time_step)
        self.predict(current_time)
        #TODO: set T = timestep in state transition matrix
        # while(time_step > self.dt):
        #     time_step -= self.dt
        #     self.predict()
        # if (time_step%self.dt > 0):
        #     self.predict()
        self.last_call_time = current_time

        return self.x

    def update_lane1(self, measurement):
        """
        @param measurements: np array of shape (6, 1) with values of:
        y1, m1, y2, y3, m3
        return x: updated state array of shape (12, 1) with values of:
        y1, Vy1, m1, Vm1, y2, Vy2, m2, Vm2, y3, Vy3, m3, Vm3
        """
        Y = measurement - np.matmul(self.H, self.x)
        covariance_sum = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.pinv(covariance_sum))
        self.x = self.x + np.matmul(K, Y)
        KH = np.matmul(K, self.H)
        self.P = np.matmul((np.eye(KH.shape[0]) - KH), self.P)
        return self.x

if __name__ == '__main__':
    first_call_time = 0.1
    lane_tracker = LaneKalmanFilter()
    lane_tracker.initialize_filter(first_call_time)
    lane_tracker.predict_step(0.2)
    test_measurement = np.array([[3.1],[1.3],[1.5],[2.5], [1.1],[4.3]])
    print(lane_tracker.update_step(test_measurement))
