import numpy as np
import torch
from torch.nn import Parameter
from cyclic_gps.models import LEGFamily
import matplotlib.pyplot as plt

DTYPE = torch.double
RANK = 5

PATH_TO_NPY = "../numpy_arrays/"
with open(PATH_TO_NPY + "sample3_ts.npy", "rb") as f:
    sample3_ts = np.load(f)
with open(PATH_TO_NPY + "sample3_vals.npy", "rb") as f:
    sample3_vals = np.load(f)
with open(PATH_TO_NPY + "N.npy", "rb") as f:
    N = np.load(f)
with open(PATH_TO_NPY + "R.npy", "rb") as f:
    R = np.load(f)
with open(PATH_TO_NPY + "B.npy", "rb") as f:
    B = np.load(f)
with open(PATH_TO_NPY + "Lambda.npy", "rb") as f:
    L = np.load(f)

sample3_ts = torch.from_numpy(sample3_ts)
sample3_vals = torch.from_numpy(sample3_vals)
N = torch.from_numpy(N)
R = torch.from_numpy(R)
B = torch.from_numpy(B)
L = torch.from_numpy(L)

print(sample3_ts.shape)
print(sample3_vals.shape)
print(N.shape)

leg_model = LEGFamily(rank=RANK, obs_dim=sample3_vals.shape[1], data_type=DTYPE)
leg_model.N = Parameter(N)
leg_model.R = Parameter(R)
leg_model.B = Parameter(B)
leg_model.Lambda = Parameter(L)
leg_model.calc_G()

# print(leg_model.N)
# print(leg_model.R)
# print(leg_model.B)
# print(leg_model.Lambda)


sample3_ts_chopped = sample3_ts[:200]
sample3_vals_chopped = sample3_vals[:200]


test_ll = leg_model.log_likelihood(sample3_ts, sample3_vals)
print("test log_likelihood with jackson's params: {}".format(test_ll))


forecast_times = sample3_ts[200:300]

# plt.scatter(sample3_ts,sample3_vals[:,0],color='C1',alpha=.2)
# plt.scatter(sample3_ts_chopped,sample3_vals_chopped[:,0],color='C0')
# plt.show()

#print("N check before preds: {}".format(leg_model.N))
pred_means, pred_variances = leg_model.make_predictions(sample3_ts_chopped, sample3_vals_chopped, forecast_times)
#print("N check after preds: {}".format(leg_model.N))

pred_means = pred_means.detach().numpy()
pred_variances = pred_variances.detach().numpy()

plt.scatter(sample3_ts_chopped, sample3_vals_chopped[:, 0], label='observed data')
plt.scatter(sample3_ts[200:300], sample3_vals[200:300][:, 0],label='censored data')
plt.plot(forecast_times, pred_means[:,0], 'C1', label='forecasting')
plt.fill_between(forecast_times,
                 pred_means[:,0]+2*np.sqrt(pred_variances[:,0,0]),
                 pred_means[:,0]-2*np.sqrt(pred_variances[:,0,0]),
                color='black',alpha=.5,label='Uncertainty')
plt.legend() #bbox_to_anchor=[1,1],fontsize=20
plt.show()





