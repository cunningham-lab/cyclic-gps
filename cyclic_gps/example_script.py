import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import scipy as sp
import scipy.ndimage
from cyclic_gps.models import LEGFamily
from cyclic_gps.data_utils import time_series_dataloader
import matplotlib.pyplot as plt


num_datapoints = 100
sample1_ts = torch.empty(num_datapoints)
sample1_ts = torch.cumsum(sample1_ts.exponential_(lambd=1) + 0.01, dim=0)
sample1_vals = torch.tensor(
    sp.ndimage.gaussian_filter1d(torch.randn(num_datapoints), 10, axis=0)[:, None]
)
sample2_vals = torch.tensor(
    sp.ndimage.gaussian_filter1d(torch.randn(num_datapoints), 10, axis=0)[:, None]
)
vals = torch.cat([sample1_vals, sample2_vals], dim=-1)

assert vals.shape == (num_datapoints, 2)
assert sample1_ts.shape == (num_datapoints,)

plt.scatter(sample1_ts, vals[:, 0])
plt.scatter(sample1_ts, vals[:, 1])

plt.show()

RANK = 5
MAX_EPOCHS = 100

# create a torch dataset, and add a batch dim of zero
dataset = time_series_dataloader(sample1_ts.unsqueeze(0), vals.unsqueeze(0))
example = dataset[0]
assert torch.allclose(example[0], sample1_ts.unsqueeze(0))

dl = DataLoader(dataset=dataset, batch_size=1)

LEG_model = LEGFamily(rank=RANK, obs_dim=vals.shape[1], train=True)

trainer = pl.Trainer(max_epochs=MAX_EPOCHS)
trainer.fit(model=LEG_model, train_dataloaders=dl)
