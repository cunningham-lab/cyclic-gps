import torch
from typing import List

def threshold_timesteps(ts_vec: TensorType["num_obs"], thresh=1e-10, check=True):
	"""
	inputs:
		all_ts: tensor of observation times for a sample
		thresh: minimum difference between observation times
		check: whether to make sure subseqent observation times are not constant
	output:

	"""


	diff=ts_vec[1:]-ts_vec[:-1]
	if check:
		assert (diff >= 0).all()

	#only taking the time points which differ from previous time one by greater than the threshold
	good = torch.cat([torch.tensor(True), diff>thresh], axis=0)
	ts = ts_vec[good]

	#find 
	idxs = torch.cum_sum(good.int()) - 1

	return ts, idxs #might need to change ts data type


