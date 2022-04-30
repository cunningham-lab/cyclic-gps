import torch
from torch.optim import Adam
import gpytorch

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures, lr):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)
        self.lr = lr

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, mll, t, x, batch_idx=0):
        nobs = torch.numel(x)
        output = self(t)
        loss = -mll(output, x) #/ nobs
        return loss


def train_gp(model, train_ts, train_xs, num_training_iters):
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    model.likelihood.train()
    optimizer = model.configure_optimizers()
    print(train_ts.shape, train_xs.shape)
    for i in range(num_training_iters):
        optimizer.zero_grad()
        loss = model.training_step(mll=mll,t=train_ts, x=train_xs)
        loss.backward()
        if i % 100 == 0:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, num_training_iters, loss.item()))
        optimizer.step()
    return model

def test_gp(model, test_ts):
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model.likelihood(model(test_ts))
    return preds




    

