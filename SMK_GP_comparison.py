from cyclic_gps.other_gps import SpectralMixtureGPModel, train_gp, test_gp
from cyclic_gps.plotting_utils import plot_predictions
import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt


data_np = pd.read_csv("../data/co2_mm_mlo.csv", comment='#',names=['year','month','decimal date','average','interpolated','trend','mysterycolumn1','mysterycolumn2'],header=0).to_numpy().astype(np.double)
co2_data = torch.from_numpy(data_np)
all_ts = co2_data[:, 2]
all_xs = co2_data[:, 3]#.unsqueeze(-1)

all_ts = 12*(all_ts-all_ts.min()) # one unit of time = one sample on average
all_xs = all_xs-torch.mean(all_xs)
all_xs = all_xs/torch.std(all_xs)

train_ts = torch.cat([all_ts[:262], all_ts[502:-28]], dim=0) 
train_xs = torch.cat([all_xs[:262], all_xs[502:-28]], dim=0) 

RANK = 5
MAX_EPOCHS = 5000

likelihood = gpytorch.likelihoods.GaussianLikelihood()
smk_model = SpectralMixtureGPModel(train_x=train_ts,train_y=train_xs,likelihood=likelihood,num_mixtures=RANK)
trained_model = train_gp(smk_model, train_ts, train_xs, num_training_iters=MAX_EPOCHS)

interpolate_ts = all_ts[262:501]
forecast_ts = all_ts[501:-28] + 12 * 20
test_ts = torch.cat([interpolate_ts, forecast_ts], dim=0)

preds = test_gp(trained_model, test_ts=test_ts)
pred_means = preds.mean.numpy()[...,None]
lower, upper = preds.confidence_region()
lower_confidence = lower.detach().numpy()
upper_confidence = upper.detach().numpy()

plot_predictions(
    all_ts[:-28], 
    all_xs[:-28, None], 
    [interpolate_ts, forecast_ts], 
    [pred_means[:len(interpolate_ts)],pred_means[len(interpolate_ts):]],
    lower_confidence=[lower_confidence[:len(interpolate_ts)],lower_confidence[len(interpolate_ts):]],
    upper_confidence=[upper_confidence[:len(interpolate_ts)],upper_confidence[len(interpolate_ts):]]
)
plt.show()


