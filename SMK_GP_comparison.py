from cProfile import label
from cyclic_gps.other_gps import SpectralMixtureGPModel, train_gp, test_gp
from cyclic_gps.plotting_utils import plot_predictions
from cyclic_gps.dataset_process_utils import load_BART
import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt

DATA_PATH = "../data/date-hour-soo-dest-2011.csv"
all_ts, all_xs, train_ts, train_xs = load_BART(DATA_PATH, dtype=torch.float32, load_tensor=True)

all_xs = all_xs.squeeze(-1) #needed for GPytorch
train_xs = train_xs.squeeze(-1)

RANK = 5
MAX_EPOCHS = 20000

likelihood = gpytorch.likelihoods.GaussianLikelihood()
smk_model = SpectralMixtureGPModel(train_x=train_ts,train_y=train_xs,likelihood=likelihood,num_mixtures=RANK, lr=1e-3)
trained_model = train_gp(smk_model, train_ts, train_xs, num_training_iters=MAX_EPOCHS)

# interpolate_ts = all_ts[262:501]
# forecast_ts = all_ts[501:-28] + 12 * 20
interpolate_ts = train_ts + 5
forecast_ts = all_ts[len(train_ts):]
test_ts = torch.cat([interpolate_ts, forecast_ts], dim=0)


preds = test_gp(trained_model, test_ts=test_ts)
pred_means = preds.mean.numpy()[...,None]
lower, upper = preds.confidence_region()
lower_confidence = lower.detach().numpy()
upper_confidence = upper.detach().numpy()

# plot_predictions(
#     all_ts[:-28], 
#     all_xs[:-28, None], 
#     [interpolate_ts, forecast_ts], 
#     [pred_means[:len(interpolate_ts)],pred_means[len(interpolate_ts):]],
#     lower_confidence=[lower_confidence[:len(interpolate_ts)],lower_confidence[len(interpolate_ts):]],
#     upper_confidence=[upper_confidence[:len(interpolate_ts)],upper_confidence[len(interpolate_ts):]]
# )
plot_predictions(
    observation_ts=all_ts, 
    observation_xs=all_xs[:, None], 
    test_ts=[interpolate_ts, forecast_ts], 
    pred_means=[pred_means[:len(interpolate_ts)],pred_means[len(interpolate_ts):]],
    lower_confidence=[lower_confidence[:len(interpolate_ts)],lower_confidence[len(interpolate_ts):]],
    upper_confidence=[upper_confidence[:len(interpolate_ts)],upper_confidence[len(interpolate_ts):]],
    labels=["interpolation", "forecasting"]
)
plt.legend()
plt.show()


