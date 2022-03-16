import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import scipy as sp
import scipy.ndimage
from cyclic_gps.models import LEGFamily
from cyclic_gps.data_utils import time_series_dataloader
import matplotlib.pyplot as plt

sample1_ts = torch.empty(1000)
sample1_ts = torch.cumsum(sample1_ts.exponential_(lambd=1) + 0.01, dim=0)

sample1_vals = torch.tensor(sp.ndimage.gaussian_filter1d(torch.randn(1000),10,axis=0)[:,None])

print(sample1_ts.shape)
print(sample1_vals.shape)

plt.scatter(sample1_ts, sample1_vals)
#plt.show()

RANK = 5
MAX_EPOCHS = 100

dataset = time_series_dataloader(sample1_ts.unsqueeze(0), sample1_vals.unsqueeze(0))

dl = DataLoader(dataset=dataset, batch_size=1)

LEG_model = LEGFamily(rank=RANK, n=sample1_ts.shape[0], train=True)

trainer = pl.Trainer(max_epochs=MAX_EPOCHS)
trainer.fit(model=LEG_model, train_dataloaders=dl)






