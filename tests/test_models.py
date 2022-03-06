import torch
import numpy as np


def test_leg_family():
    from cyclic_gps.models import LEGFamily

    leg_family = LEGFamily(ell=3, n=2)
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

    zeros = torch.zeros(leg_family.ell, leg_family.ell)
    zeros[leg_family.N_idxs] = leg_family.N_params
    assert torch.allclose(
        zeros, torch.eye(leg_family.ell)
    )  # holds for the current init

    zeros_R = torch.zeros(leg_family.ell, leg_family.ell)
    zeros_R[leg_family.R_idxs] = leg_family.R_params
    # print(zeros.scatter_(dim=0, index=inds, src=leg_family.R_params))

    # our B is similar to Jackson's
    B = np.ones((leg_family.n, leg_family.ell))
    B_numpy = 0.5 * B / np.sqrt(np.sum(B ** 2, axis=1, keepdims=True))
    assert torch.allclose(leg_family.B, torch.tensor(B_numpy).float())

    leg_family.Lambda_from_params()  # fill in the matrix
    lambda_lambda_t = leg_family.calc_Lambda_Lambda_T(leg_family.Lambda)
    assert lambda_lambda_t.shape == (2, 2)  # observation ddims
    assert torch.allclose(lambda_lambda_t, lambda_lambda_t.T)  # symmetry


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
