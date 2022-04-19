import torch
from torch.utils.data import Dataset
from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import scipy as sp
import scipy.ndimage

patch_typeguard()

# might not be neccessary for our purposes, takes into account multiple observations per time point (Note that this is different than having multidimensional observations)
@typechecked
def threshold_timesteps(
    ts_vec: TensorType["num_obs"], thresh=1e-10, check=True
) -> Tuple[TensorType["num_thresholded_obs"], TensorType["num_obs"]]:
    """
	inputs:
		all_ts: tensor of observation times for a sample
		thresh: minimum difference between observation times
		check: whether to make sure subseqent observation times are not constant
	output:
		ts

	"""

    diff = ts_vec[1:] - ts_vec[:-1]
    if check:
        assert (diff >= 0).all()

    # only taking the time points which differ from previous time one by greater than the threshold
    good = torch.cat([torch.tensor(True), diff > thresh], axis=0)
    ts = ts_vec[good]

    # find index
    # true, true, false, true, false, false, true
    # 1, 2, 2, 3, 3, 3, 4
    # 0, 1, 1, 2, 2, 2, 3

    # indexes that go from original timestep indices to the new timestep indices
    idxs = torch.cum_sum(good.int()) - 1

    return ts, idxs  # might need to change ts data type

def generate_data(num_datapoints, data_dim, data_type, spacing: str = "irregular") -> Tuple[TensorType["num_datapoints"], TensorType["num_datapoints", "data_dim"]]:
    if spacing == "irregular":
        ts = torch.empty(num_datapoints, dtype=data_type)
        ts = torch.cumsum(ts.exponential_(lambd=1) + 0.01, dim=0)
    else:
        ts = torch.cumsum(torch.ones(num_datapoints), dim=0)
    vals = []
    for _ in range(data_dim):
        oned_vals = torch.tensor(
            sp.ndimage.gaussian_filter1d(torch.randn(num_datapoints, dtype=data_type), 10, axis=0)[:, None]
        )
        vals.append(oned_vals)
    vals = torch.cat(vals, dim=-1)
    return ts, vals


@typechecked
class time_series_dataset(Dataset):
    def __init__(
        self,
        ts: TensorType["batch", "num_observations"],
        xs: TensorType["batch", "num_observations", "observation_dim"],
    ):
        self.ts = ts
        self.xs = xs

    def __len__(self):
        return self.ts.shape[0]

    def __getitem__(self, idx):
        # TODO: for now, just take the first element of the batch
        return self.ts[0, :], self.xs[0, :, :]

def calc_per_element_percentage_diff(tensor1, tensor2):
    return (torch.sum(torch.abs((tensor2 - tensor1)/tensor1))/(torch.numel(tensor1))) * 100

