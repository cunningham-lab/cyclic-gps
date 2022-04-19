import torch
from cyclic_gps.models import LEGFamily
from cyclic_gps.model_utils import compute_log_marginal_likelihood
from cyclic_gps.data_utils import generate_data
from cyclic_gps.kalman import init_kalman_filter, kf_log_marginal_likelihood
import numpy as np


def test_log_marginal_likelihood():
    RANK = 5
    DTYPE = torch.double
    for spacing in ["regular", "irregular"]:
        for n in [10, 33, 50, 100, 150]:
            for d in [1, 2, 3]:
                print(n, d)
                ts, xs = generate_data(num_datapoints=n, data_dim=d, data_type=DTYPE, spacing = spacing)
                leg_model = LEGFamily(rank=RANK, obs_dim=xs.shape[-1], train=False, data_type=DTYPE)
                leg_model.double()
                naive_ll = compute_log_marginal_likelihood(N = leg_model.N, R = leg_model.R, B = leg_model.B, Lambda=leg_model.calc_Lambda_Lambda_T(leg_model.Lambda), ts=ts, xs=xs)
                leg_ll = leg_model.log_likelihood(ts=ts, xs=xs)
                if spacing == "regular":
                    kf = init_kalman_filter(leg_model=leg_model, use_approximation=False)
                    kf_ll = kf_log_marginal_likelihood(kf, xs)
                    assert(torch.allclose(leg_ll, torch.from_numpy(np.array(kf_ll))))
                    print("naive ll: {}, leg ll: {}, kf ll: {}".format(naive_ll, leg_ll, kf_ll))
                assert(torch.allclose(leg_ll, naive_ll))





