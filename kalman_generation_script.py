import torch
import numpy as np
import matplotlib.pyplot as plt
from cyclic_gps.models import LEGFamily
from cyclic_gps.kalman import init_kalman_filter, generate_states_from_kalman, get_state_estimates
import filterpy
from filterpy.kalman import KalmanFilter


num_datapoints = 50
RANK = 2
OBS_DIM = 2
TIME_STEP = 2

leg_model = LEGFamily(rank=RANK, obs_dim=OBS_DIM, train=False)
kf = init_kalman_filter(leg_model, TIME_STEP, use_approximation=False)

ts = torch.arange(start=0, end=TIME_STEP * (num_datapoints), step=TIME_STEP).numpy()
print(ts.shape)
zs = generate_states_from_kalman(kf, ts)
xs = zs @ kf.H.T + np.random.multivariate_normal(mean=np.zeros(shape=(2)), cov=leg_model.calc_Lambda_Lambda_T(leg_model.Lambda).numpy(), size=num_datapoints) 
kf_state_ests = get_state_estimates(kf, xs).squeeze(-1)
leg_state_ests, _ = leg_model.compute_insample_posterior(ts=torch.from_numpy(ts).float(), xs=torch.from_numpy(xs).float())

print(zs.shape)
plt.scatter(ts, zs[:,0], label='original states', c='green')
plt.scatter(ts, kf_state_ests[:,0], label='kf predicted states', c='orange')
plt.scatter(ts, leg_state_ests.numpy()[:,0], label='leg posterior predictions', c='blue')

plt.legend()
plt.show()