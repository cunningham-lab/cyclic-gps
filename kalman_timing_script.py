import torch
import numpy as np
import matplotlib.pyplot as plt
from cyclic_gps.models import LEGFamily
from cyclic_gps.kalman import init_kalman_filter, generate_states_from_kalman, get_state_estimates, log_marginal_likelihood
import filterpy
from filterpy.kalman import KalmanFilter
from time import time


RANK = 2
OBS_DIM = 2
TIME_STEP = 0.5
NOISE_LEVEL = 1
LENGTH_SCALE = 4

leg_model = LEGFamily(rank=RANK, obs_dim=OBS_DIM, train=False, prior_process_noise_level=NOISE_LEVEL, prior_length_scale=LENGTH_SCALE)
kf = init_kalman_filter(leg_model, TIME_STEP, use_approximation=False)

START_N = 200
END_N = 400000
STEP = 20000
#ns = range(START_N, END_N, STEP)
pows = range(1, 7)
ns = [10 ** i for i in pows]
print(ns)
leg_post_times = np.empty(len(ns))
kf_post_times = np.empty(len(ns))
leg_ll_times = np.empty(len(ns))
kf_ll_times = np.empty(len(ns))
i = 0
total_percent_post_diff = 0
total_percent_ll_diff = 0
for n in ns:
    print(n)
    ts = torch.arange(start=0, end=TIME_STEP * n, step=TIME_STEP).float()
    zs = generate_states_from_kalman(kf, ts)
    xs_np = zs @ kf.H.T + np.random.multivariate_normal(mean=np.zeros(shape=(2)), cov=leg_model.calc_Lambda_Lambda_T(leg_model.Lambda).numpy(), size=n) 
    xs = torch.from_numpy(xs_np).float()
    kf_t_i = time() 
    kf_state_ests = get_state_estimates(kf, xs_np).squeeze(-1)
    kf_t_f = time()
    kf_post_times[i] = kf_t_f - kf_t_i
    leg_t_i = time() 
    leg_state_ests, _ = leg_model.compute_insample_posterior(ts=ts, xs=xs)
    leg_t_f = time() 
    leg_post_times[i] = leg_t_f - leg_t_i
    percent_post_diff = np.sum(np.abs((np.array(leg_state_ests) -  kf_state_ests)/kf_state_ests)) / (zs.shape[0] * zs.shape[1]) * 100
    total_percent_post_diff += percent_post_diff
    kf_t_i = time() 
    kf_ll = log_marginal_likelihood(kf, xs_np)
    kf_t_f = time()
    kf_ll_times[i] = kf_t_f - kf_t_i
    leg_t_i = time()
    leg_ll = leg_model.log_likelihood(ts=ts, xs=xs)
    leg_t_f = time() 
    leg_ll_times[i] = leg_t_f - leg_t_i
    percent_ll_diff = np.abs((np.array(leg_ll) - kf_ll)/kf_ll)*100
    total_percent_ll_diff += percent_ll_diff
    i += 1

print("average posterior estimates percentage difference per state estimate coordinate: {}".format(total_percent_post_diff/len(ns)))
print("average log-likelihood percentage difference: {}".format(total_percent_ll_diff/len(ns)))
ns = np.array(list(ns), dtype=float)
fig, (axs1, axs2) = plt.subplots(nrows=1, ncols=2)
axs1.scatter(ns, kf_post_times, label="kf posterior times")
axs1.scatter(ns, leg_post_times, label="leg posterior times")
axs1.set_xlabel("num datapoints")
axs1.set_ylabel("seconds")
axs1.set_xscale("log")
axs1.set_yscale("log")
axs1.legend()
axs2.scatter(ns, kf_ll_times, label="kf log likelihood times")
axs2.scatter(ns, leg_ll_times, label="leg log likelihood times")
axs2.set_xlabel("num datapoints")
axs2.set_ylabel("seconds")
axs2.set_xscale("log")
axs2.set_yscale("log")
axs2.legend()
plt.show()

