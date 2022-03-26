import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import scipy as sp
import scipy.ndimage
from cyclic_gps.models import LEGFamily
from cyclic_gps.data_utils import time_series_dataset
import matplotlib.pyplot as plt


from cyclic_gps.kalman import init_kalman_filter, get_state_estimates
import filterpy
from filterpy.kalman import KalmanFilter
import numpy.random as npr

print("OK")

num_datapoints = 100
sample1_ts = torch.empty(num_datapoints)
#sample1_ts = torch.cumsum(sample1_ts.exponential_(lambd=1) + 0.01, dim=0)
sample1_ts = torch.cumsum(torch.ones(num_datapoints), dim=0) #starting with uniform time steps
sample1_vals = torch.tensor(
    sp.ndimage.gaussian_filter1d(torch.randn(num_datapoints), 10, axis=0)[:, None]
)
sample2_vals = torch.tensor(
    sp.ndimage.gaussian_filter1d(torch.randn(num_datapoints), 10, axis=0)[:, None]
)
vals = torch.cat([sample1_vals, sample2_vals], dim=-1)

assert vals.shape == (num_datapoints, 2)
assert sample1_ts.shape == (num_datapoints,)

# plt.scatter(sample1_ts, vals[:, 0])
# plt.scatter(sample1_ts, vals[:, 1])
# plt.show()

dataset = time_series_dataset(sample1_ts.unsqueeze(0), vals.unsqueeze(0))
example = dataset[0]
assert torch.allclose(example[0], sample1_ts.unsqueeze(0)) # we're getting our data plus one batch element
dl = DataLoader(dataset=dataset, batch_size=1)

RANK = 5
MAX_EPOCHS = 100 #for testing purposes
LEG_model = LEGFamily(rank=RANK, obs_dim=vals.shape[1], train=True)

logger = pl.loggers.TensorBoardLogger("tb_logs", name="first_pass_model")
trainer = pl.Trainer(max_epochs=MAX_EPOCHS, logger=logger, log_every_n_steps=1)
trainer.fit(model=LEG_model, train_dataloaders=dl)

TIME_STEP = 1 #time step for kalman predictions
kf = init_kalman_filter(LEG_model, TIME_STEP)
kf_state_ests = torch.from_numpy(get_state_estimates(kf, vals.numpy())).float().squeeze(-1)
leg_state_ests, _ = LEG_model.compute_insample_posterior(ts=sample1_ts, xs=vals)
print(kf_state_ests[0], leg_state_ests[0])
print(kf_state_ests.shape, leg_state_ests.shape)
print(kf_state_ests.dtype, leg_state_ests.dtype)
assert torch.allclose(kf_state_ests, leg_state_ests)






