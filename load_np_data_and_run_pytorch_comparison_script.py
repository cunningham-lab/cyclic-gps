import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import scipy as sp
import numpy as np
import scipy.ndimage
from cyclic_gps.models import LEGFamily
from cyclic_gps.data_utils import time_series_dataset
import matplotlib.pyplot as plt


num_datapoints = 1000
DTYPE = torch.double

RANK = 5
MAX_EPOCHS = 800
OPTIMIZER = "ADAM" #or "ADAM" || "BFGS"

with open("../numpy_arrays/all_ts_2.npy", "rb") as f:
    all_ts = np.load(f)
with open("../numpy_arrays/all_vals_2.npy", "rb") as f:
    all_vals = np.load(f)

all_ts = torch.from_numpy(all_ts)
all_vals = torch.from_numpy(all_vals)
print(all_ts.shape)
print(all_vals.shape)


# create a torch dataset, and add a batch dim of zero
dataset = time_series_dataset(all_ts, all_vals)
example = dataset[0]

#print("example datatype: {}".format(example[0].dtype))
assert torch.allclose(example[0], all_ts.unsqueeze(0))

dl = DataLoader(dataset=dataset, batch_size=1)
leg_model = LEGFamily(rank=RANK, obs_dim=all_vals.shape[2], train=True, optimizer=OPTIMIZER, data_type=DTYPE)
leg_model.double()
trainer = pl.Trainer(max_epochs=MAX_EPOCHS)
trainer.fit(model=leg_model, train_dataloaders=dl)
#print(leg_model.G)
leg_model.register_model_matrices_from_params()
#print(leg_model.G)

PATH_TO_NPY = "../numpy_arrays/"
with open(PATH_TO_NPY + "sample3_ts_2.npy", "rb") as f:
    sample3_ts = np.load(f)
with open(PATH_TO_NPY + "sample3_vals_2.npy", "rb") as f:
    sample3_vals = np.load(f)

sample3_ts = torch.from_numpy(sample3_ts)
sample3_vals = torch.from_numpy(sample3_vals)

# sample3_ts_chopped = sample3_ts[:200]
# sample3_vals_chopped = sample3_vals[:200]
# forecast_times = sample3_ts[200:300]
sample3_ts_chopped = torch.cat([sample3_ts[:200], sample3_ts[-200:]], dim=0)
sample3_vals_chopped = torch.cat([sample3_vals[:200], sample3_vals[-200:]], dim=0)
print("sample_3 shapes: ts:{}, vals:{}".format(sample3_ts_chopped.shape, sample3_vals_chopped.shape))
with open(PATH_TO_NPY + "forecast_times_2.npy", "rb") as f:
    forecast_times = np.load(f)
forecast_times = torch.from_numpy(forecast_times)

pred_means, pred_variances = leg_model.make_predictions(sample3_ts_chopped, sample3_vals_chopped, forecast_times)
#print("data type precision:{}".format(pred_means.dtype))
pred_means = pred_means.detach().numpy()
pred_variances = pred_variances.detach().numpy()

plt.scatter(sample3_ts_chopped, sample3_vals_chopped[:, 0], label='observed data')
plt.scatter(sample3_ts[200:-200], sample3_vals[200:-200][:, 0],label='censored data')
plt.plot(forecast_times, pred_means[:,0], 'C1', label='interpolation/forecasting')
plt.fill_between(forecast_times,
                 pred_means[:,0]+2*np.sqrt(pred_variances[:,0,0]),
                 pred_means[:,0]-2*np.sqrt(pred_variances[:,0,0]),
                color='black',alpha=.5,label='Uncertainty')
plt.legend() #bbox_to_anchor=[1,1],fontsize=20
plt.show()


