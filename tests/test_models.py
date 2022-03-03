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

    leg_family.set_initial_R()

    print(leg_family.R_params)
    print(leg_family.Lambda_params)

    leg_family.set_initial_Lambda()
    print(leg_family.Lambda_params)

    leg_family.set_initial_B()

    # our B is similar to Jackson's
    B = np.ones((leg_family.n, leg_family.ell))
    B_numpy = 0.5 * B / np.sqrt(np.sum(B ** 2, axis=1, keepdims=True))
    assert torch.allclose(leg_family.B, torch.tensor(B_numpy).float())

