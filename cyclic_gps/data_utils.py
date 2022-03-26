import torch
from torch.utils.data import Dataset
from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

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

