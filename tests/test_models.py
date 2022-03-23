import torch
import numpy as np
from cyclic_gps.models import LEGFamily


def test_leg_family():

    leg_family = LEGFamily(rank=3, obs_dim=2)
    assert len(leg_family.N_params) == 6

    assert len(leg_family.R_params) == 3

    leg_family.set_initial_N()
    # N is the identity, and we're taking it's lower triangular part
    assert torch.allclose(
        leg_family.N_params, torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    )

    assert leg_family.parameter_count == 18  # true for ell=3, n=2

    leg_family.set_initial_R()
    # assert(torch.allclose(R, -R.T))

    leg_family.set_initial_Lambda()

    leg_family.set_initial_B()

    zeros = torch.zeros(leg_family.rank, leg_family.rank)
    zeros[leg_family.N_idxs] = leg_family.N_params
    assert torch.allclose(
        zeros, torch.eye(leg_family.rank)
    )  # holds for the current init

    zeros_R = torch.zeros(leg_family.rank, leg_family.rank)
    zeros_R[leg_family.R_idxs] = leg_family.R_params
    # print(zeros.scatter_(dim=0, index=inds, src=leg_family.R_params))

    # our B is similar to Jackson's
    B = np.ones((leg_family.obs_dim, leg_family.rank))
    B_numpy = 0.5 * B / np.sqrt(np.sum(B**2, axis=1, keepdims=True))
    assert torch.allclose(leg_family.B, torch.tensor(B_numpy).float())

    leg_family.Lambda_from_params()  # fill in the matrix
    lambda_lambda_t = leg_family.calc_Lambda_Lambda_T(leg_family.Lambda)
    assert lambda_lambda_t.shape == (2, 2)  # observation ddims
    assert torch.allclose(lambda_lambda_t, lambda_lambda_t.T)  # symmetry

    leg_family.R_from_params()
    leg_family.N_from_params()
    print(leg_family.calc_G())


def test_exp_mult():
    G = torch.ones(size=(4, 4))
    t = torch.arange(start=0.0, end=1.0, step=0.1)
    out = G.unsqueeze(0) * t.reshape(-1, 1, 1)
    # assert that the shape is timepoints, G[0], G[1]
    assert out.shape == (len(t), G.shape[0], G.shape[1])
    # assert that we have a matrix per time point, that is multiplied by that time point
    for i in range(out.shape[0]):
        assert torch.allclose(out[i, :, :], t[i])

    # matrix exp doesn't change  shapes
    assert torch.matrix_exp(out).shape == out.shape


def test_BTop_LambdaLambdaTop_inv_B():
    rank = 3
    leg_family = LEGFamily(rank=rank, obs_dim=2)
    # compute Lambda Lambda^{\top}
    LambdaLambdaTop = leg_family.calc_Lambda_Lambda_T(leg_family.Lambda)
    # instead of explicitly inverting and computing (\Lambda\Lambda^{\top})^{-1} B:
    LambdaLambdaTop_inv_B = torch.linalg.solve(LambdaLambdaTop, leg_family.B)
    BTop_LambdaLambdaTop_inv_B = leg_family.B.T @ LambdaLambdaTop_inv_B
    assert BTop_LambdaLambdaTop_inv_B.shape == (rank, rank)


def test_compute_PEG_precision():
    rank = 3
    num_obs = 100
    leg_family = LEGFamily(rank=rank, obs_dim=2)
    out = leg_family.compute_PEG_precision(
        torch.linspace(start=0.0, end=10.0, steps=num_obs)
    )
    assert out[0].shape == (num_obs, rank, rank)
    assert out[1].shape == (num_obs - 1, rank, rank)


def test_posterior():
    rank = 3
    num_obs = 100
    leg_family = LEGFamily(rank=rank, obs_dim=2)
    posterior_mean, posterior_cov, posterior_precision = leg_family.posterior(
        torch.linspace(start=0.0, end=10.0, steps=num_obs)
    )

    # assert shapes
    assert posterior_precision["diag_blocks"].shape == (num_obs, rank, rank)
    assert posterior_precision["off_diag_blocks"].shape == (num_obs - 1, rank, rank)
    assert posterior_mean.shape == (num_obs, rank)
    assert posterior_cov.shape == (num_obs, rank, rank)

    # assert that the precision, posterior, are positive semi-definite

    # make_up_data from that process and do posterior inference. can you fit?
