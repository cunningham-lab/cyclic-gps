import torch
import numpy as np
import matplotlib.pyplot as plt
from cyclic_gps.models import LEGFamily
from cyclic_gps.kalman import init_kalman_filter, generate_states_from_kalman, get_state_estimates
import filterpy
from filterpy.kalman import KalmanFilter


num_datapoints = 200
RANK = 2
OBS_DIM = 2
TIME_STEP = 0.5
ts = torch.arange(start=0, end=TIME_STEP * (num_datapoints), step=TIME_STEP).numpy()
print(ts.shape)
fig, axs = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        leg_model = LEGFamily(rank=RANK, obs_dim=OBS_DIM, train=False, prior_process_noise_level=0.5 * (i + 1), prior_length_scale=j*4)
        kf = init_kalman_filter(leg_model, TIME_STEP, use_approximation=False)
        zs = generate_states_from_kalman(kf, ts)
        xs = zs @ kf.H.T + np.random.multivariate_normal(mean=np.zeros(shape=(2)), cov=leg_model.calc_Lambda_Lambda_T(leg_model.Lambda).numpy(), size=num_datapoints)
        kf = init_kalman_filter(leg_model, TIME_STEP, use_approximation=False) 
        kf_state_ests = get_state_estimates(kf, xs).squeeze(-1)
        leg_state_ests, _ = leg_model.compute_insample_posterior(ts=torch.from_numpy(ts).float(), xs=torch.from_numpy(xs).float())

        axs[i,j].plot(ts, zs[:,0], label='generated z', c='green')
        axs[i,j].set_title("N: {}I, R: {}J".format(0.5 * (i + 1), j*4))
        #axs[i,j].axis('off')
        axs[i,j].plot(ts, kf_state_ests[:,0], label='kf predicted states', c='orange')
        axs[i,j].plot(ts, leg_state_ests.numpy()[:,0], label='leg posterior predictions', c='blue')

plt.subplots_adjust(
                    bottom=0.1, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.legend()
plt.show()