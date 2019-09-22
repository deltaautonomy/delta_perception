'''
Notes:
States: y1, Vy1, m1, Vm1, y2, Vy2, m2, Vm2, y3, Vy3, m3, Vm3
Measurement:  y1, m1, y2, m2, y3, m3


Predictict step  ->Xt+1 = F * Xt
update step  -> 
'''


import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

tracker = KalmanFilter(dim_x=6, dim_z=6)
dt = .1   # time step 1 second


class LaneKalmanFilter():
    def __init__(self, state_dim = 12, measurement_dim = 6, motion_model = 'velocity', dt = 0.1):
        """
        Initializes the kalman filter class 
        @param: state_dim - number of states
        @param: measurement_dim - number of observations returned by erfnet
        @param: dt - timestep at which the vehicle state should update
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.dt = dt
        self.motion_model = motion_model
        self.initialize_filter()
    
    def initialize_filter(self):
        """
        Internal function to initialize the filter
        """

        T = self.dt
        # state transition matrix
        self.F = np.array([[1, dt, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
                          [0,  1, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0],
                          [0,  0, 1, dt, 0,  0, 0,  0, 0,  0, 0,  0],
                          [0,  0, 0,  1, 0,  0, 0,  0, 0,  0, 0,  0],
                          [0,  0, 0,  0, 1, dt, 0,  0, 0,  0, 0,  0],
                          [0,  0, 0,  0, 0,  1, 0,  0, 0,  0, 0,  0],
                          [0,  0, 0,  0, 0,  0, 1, dt, 0,  0, 0,  0],
                          [0,  0, 0,  0, 0,  0, 0,  1, 0,  0, 0,  0],
                          [0,  0, 0,  0, 0,  0, 0,  0, 1, dt, 0,  0],
                          [0,  0, 0,  0, 0,  0, 0,  0, 0,  1, 0,  0],
                          [0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 1, dt],
                          [0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  1]])

        # H is how we go from state variable to measurement variable
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

        # R is the measurement noise        
        self.R = np.eye(6, dtype=int)* 5

        self.P = np.eye(12) * 500
    
    def predict(self, dt):

    def update(self, measurement):
        





# TODO: Q has to be replaced since maximum dimension for this noise is 4 (8) only. We need it to be 6. 
q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
tracker.Q = block_diag(q, q)







# Initial condition 
tracker.x = np.zeros([12, 1]) # initial state
tracker.P = np.eye(12) * 500 # initial state covariance

