from statistics import variance
import pandas as pd
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader
from cyclic_gps.models import LEGFamily
from cyclic_gps.data_utils import time_series_dataset, calc_per_element_percentage_diff
from cyclic_gps.plotting_utils import plot_predictions
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pickle

LOAD_PARAMS = False
PATH_TO_NPY = "../numpy_arrays/"

data_np = pd.read_csv("../data/co2_mm_mlo.csv", comment='#',names=['year','month','decimal date','average','interpolated','trend','mysterycolumn1','mysterycolumn2'],header=0).to_numpy().astype(np.double)
co2_data = torch.from_numpy(data_np)
all_ts = co2_data[:, 2]
all_xs = co2_data[:, 3].unsqueeze(-1)

ts_standardized = 12*(all_ts-all_ts.min()) # one unit of time = one sample on average
xs_standardized = all_xs-torch.mean(all_xs)
xs_standardized = xs_standardized/torch.std(xs_standardized)

all_ts = ts_standardized
all_xs = xs_standardized

train_ts = torch.cat([all_ts[:262], all_ts[502:-28]], dim=0) 
train_xs = torch.cat([all_xs[:262], all_xs[502:-28]], dim=0) 

dataset = time_series_dataset(train_ts.unsqueeze(0), train_xs.unsqueeze(0))
example = dataset[0]
assert torch.allclose(example[0], train_ts.unsqueeze(0)) # we're getting our data plus one batch element
dl = DataLoader(dataset=dataset, batch_size=1)

RANK = 5
MAX_EPOCHS = 5000
DTYPE = torch.double
OPTIMIZER = "ADAM" #or "ADAM" || "BFGS"
leg_model = LEGFamily(rank=RANK, obs_dim=all_xs.shape[-1], train=True, optimizer=OPTIMIZER, data_type=DTYPE, lr=1e-3)
leg_model.double()

if not LOAD_PARAMS:
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS)
    trainer.fit(model=leg_model, train_dataloaders=dl)
    leg_model.register_model_matrices_from_params()
else:
    with open(PATH_TO_NPY + "good_params.pkl", "rb") as f:
        params = pickle.loads(f.read())
        N = params["N"]
        R = params["R"]
        B = params["B"]
        L = params["Lambda"]
    # with open(PATH_TO_NPY + "N_3.npy", "rb") as f:
    #     N = np.load(f)
    # with open(PATH_TO_NPY + "R_3.npy", "rb") as f:
    #     R = np.load(f)
    # with open(PATH_TO_NPY + "B_3.npy", "rb") as f:
    #     B = np.load(f)
    # with open(PATH_TO_NPY + "Lambda_3.npy", "rb") as f:
    #     L = np.load(f)
    N = torch.from_numpy(N)
    R = torch.from_numpy(R)
    B = torch.from_numpy(B)
    L = torch.from_numpy(L)
    leg_model.N = Parameter(N)
    leg_model.R = Parameter(R)
    leg_model.B = Parameter(B)
    leg_model.Lambda = Parameter(L)
    leg_model.calc_G()

interpolate_ts = all_ts[262:501]
forecast_ts = all_ts[501:-28] + 12 * 20
test_ts = torch.cat([interpolate_ts, forecast_ts], dim=0)

pred_means, pred_variances = leg_model.make_predictions(train_ts, train_xs, test_ts)

# with open(PATH_TO_NPY + "jackson_good_params_preds.pkl", "rb") as f:
#     jackson_preds = pickle.loads(f.read())
# jackson_mean = torch.from_numpy(jackson_preds["means"])
# jackson_variances = torch.from_numpy(jackson_preds["variances"])
# mean_diff = calc_per_element_percentage_diff(pred_means, jackson_mean)
# print("mean percentage diff: {}".format(mean_diff))
# print(pred_means[:5], jackson_mean[:5])
# #assert(torch.allclose(pred_means, jackson_mean))
# variance_diff = calc_per_element_percentage_diff(pred_variances, jackson_variances)
# print("variance percentage diff: {}".format(variance_diff))
# #print(pred_variances[:5], jackson_variances[:5])
# #assert(torch.allclose(pred_variances, jackson_variances))

pred_means = pred_means.detach().numpy()
pred_variances = pred_variances.detach().numpy()
plot_predictions(all_ts[:-28], all_xs[:-28], [interpolate_ts, forecast_ts], [pred_means[:len(interpolate_ts)],pred_means[len(interpolate_ts):]], pred_variances=[pred_variances[:len(interpolate_ts)], pred_variances[len(interpolate_ts):]])
plt.legend()
plt.show()




