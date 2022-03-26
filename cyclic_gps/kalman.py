import numpy as np
import numpy.random as npr
from filterpy.kalman import KalmanFilter

def init_kalman_filter(leg_model, time_step=1):
    leg_model.register_model_matrices_from_params()
    kf = KalmanFilter(dim_x=leg_model.rank, dim_z=leg_model.obs_dim)
    x_0 = npr.rand(leg_model.rank, 1)
    kf.x = x_0
    kf.P = np.eye(leg_model.rank) #initial state covariance I think
    A = np.eye(leg_model.rank, leg_model.rank) - 0.5 * time_step * leg_model.G.detach().numpy() #first order taylor approximation of mean of p(z_t | z_{t-1}) (which would be x in their notation), assumes equal time steps of a given magnitude
    kf.F = A
    kf.Q = np.eye(leg_model.rank, leg_model.rank) - (A @ A.T) #first order taylor approximation of covariance of 
    #p(z_t | z_{t-1})
    kf.H = leg_model.B.detach().numpy() #measurement function
    kf.R = leg_model.calc_Lambda_Lambda_T(leg_model.Lambda).detach().numpy() #measurement covariance
    #kf.B (control transition matrix) not being used I believe
    return kf

def get_state_estimates(kf, data):
    #kf: pre-initialized kalman filter
    #data: (num_obs x data_dim)
    state_estimates = np.empty((data.shape[0], kf.x.shape[0], 1))
    for i in range(data.shape[0]):
        kf.predict()
        kf.update(data[i])
        state_estimates[i] = kf.x
    return state_estimates



