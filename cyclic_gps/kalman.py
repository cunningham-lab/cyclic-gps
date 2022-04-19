import numpy as np
import numpy.random as npr
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import expm

def init_kalman_filter(leg_model, time_step=1, use_approximation=True):
    leg_model.register_model_matrices_from_params()
    kf = KalmanFilter(dim_x=leg_model.rank, dim_z=leg_model.obs_dim)
    x_0 = np.zeros((leg_model.rank, 1))
    #x_0 = npr.multivariate_normal(mean=np.zeros(leg_model.rank), cov=np.eye(leg_model.rank)) #*2000?
    kf.x = x_0 
    #kf.P = np.eye(leg_model.rank) #initial state covariance I think
    if use_approximation:
        A = np.eye(leg_model.rank, leg_model.rank) - 0.5 * time_step * leg_model.G.detach().numpy() #first order taylor approximation of mean of p(z_t | z_{t-1}) (which would be x in their notation), assumes equal time steps of a given magnitude
        Q = time_step * (leg_model.N @ leg_model.N.T).detach().numpy()
    elif not use_approximation:
        A = expm(-0.5 * time_step * leg_model.G.detach().numpy())
        Q = np.eye(leg_model.rank, leg_model.rank) - (A @ A.T)
    kf.F = A
    kf.Q = Q
    #p(z_t | z_{t-1})
    kf.H = leg_model.B.detach().numpy() #measurement function
    kf.R = leg_model.calc_Lambda_Lambda_T(leg_model.Lambda).detach().numpy() #measurement covariance
    #kf.B (control transition matrix) not being used I believe

    return kf

def generate_states_from_kalman(kf, ts):
    #kf: pre-initialized kalman filter
    #ts: timepoints to make predictions at (assuming uniform discrete time for now)
    states = np.empty(shape=(ts.shape[0], kf.dim_x)) #x in filterpy notation is the z we want
    for i in range(len(ts)):
        process_noise = npr.multivariate_normal(mean=np.zeros(kf.dim_x), cov=kf.Q)
        kf.predict()
        kf.x += np.expand_dims(process_noise, axis=-1)
        states[i] = kf.x.squeeze()
    
    return states


def get_state_estimates(kf, data):
    #kf: pre-initialized kalman filter
    #data: (num_obs x data_dim)
    (state_estimates, state_covariances, _, _) = kf.batch_filter(data)
    (state_estimates_smoothed, _, _, _) = kf.rts_smoother(state_estimates, state_covariances)
    #state_estimates = np.empty((data.shape[0], kf.x.shape[0], 1))
    # for i in range(data.shape[0]):
    #     kf.predict()
    #     kf.update(data[i])
    #     state_estimates[i] = kf.x
    return state_estimates_smoothed

def kf_log_marginal_likelihood(kf, data):
    ll = 0
    for i in range(data.shape[0]):
        kf.predict()
        kf.update(data[i])
        ll += kf.log_likelihood
    return ll

def zero_filter(kf, rank):
    kf.x_0 = np.zeros((rank, 1))
    kf.P = np.eye(rank)
    return kf





