from matplotlib.pyplot import axis
import torch as torch
import numpy as np
from typing import List, Dict, Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
class LEGFamily:
    # TODO: inherit from torch.nn.module? or lightning module?

    """
    z \sim PEG(N, R)
    x(t) \sim \mathcal{N}(Bz(t), \Lambda \Lambda^{\top})
    """

    def __init__(self, ell: int, n: int, train: bool = False) -> None:
        # TODO: change ell -> rank, and n -> obs dim ?
        self.ell = ell
        self.n = n

        # everything below and including diagonal in (self.ell, self.ell) mat
        self.N_idxs = self.inds_to_tuple(
            torch.tril_indices(row=self.ell, col=self.ell, offset=0)
        )
        self.N_params = torch.empty(len(self.N_idxs[0]), requires_grad=train)
        torch.nn.init.uniform_(self.N_params)  # initialize (later override)

        # everything below and including lower off-diagonal in (self.ell, self.ell) mat
        self.R_idxs = self.inds_to_tuple(
            torch.tril_indices(row=self.ell, col=self.ell, offset=-1)
        )
        self.R_params = torch.empty(len(self.R_idxs[0]), requires_grad=train)
        torch.nn.init.uniform_(self.R_params)  # initialize (later override)

        # convert to tuple
        # inds = (leg_family.N_idxs[0], leg_family.N_idxs[1])
        # fill in parameters inside mats value
        # zeros.index_put_(indices=inds, values=torch.arange(inds[0].shape).float())

        # everything below and including diagonal in (self.n, self.n) mat
        self.Lambda_idxs = self.inds_to_tuple(
            torch.tril_indices(row=self.n, col=self.n, offset=0)
        )
        self.Lambda_params = torch.empty(len(self.Lambda_idxs[0]), requires_grad=train)
        torch.nn.init.uniform_(self.Lambda_params)  # initialize (later override)

        self.B = torch.empty((self.n, self.ell))
        torch.nn.init.uniform_(self.B)  # initialize (later override)

    @staticmethod
    def inds_to_tuple(
        raw_inds: TensorType[2, "num_entries"]
    ) -> Tuple[TensorType["num_entries"], TensorType["num_entries"]]:
        return (raw_inds[0], raw_inds[1])

    def get_initial_guess(self):
        pass

    def set_initial_N(self) -> None:
        """modify the initial data of self.N_params"""
        I = torch.eye(self.ell)
        # Jackson had the below as torch.linalg.cholesky(N@N.T) but Cholesky of an identity is the identity. https://github.com/jacksonloper/leg-gps/blob/c160f13440d67e1041b5b13cdab9dab253569ee7/leggps/training.py#L189
        N = I @ I.T
        self.N_params.data = N[self.N_idxs]

    def set_initial_R(self) -> None:
        """modify the initial data of self.N_params"""
        R = torch.randn((self.ell, self.ell)) * 0.2
        R = R - R.T  # makes R anti-symetric
        # Jackson had it in two steps depending on whether R is provided or not, but I decided to simplify and get an identical result (hope ot not run into over/underflow issues). see https://github.com/jacksonloper/leg-gps/blob/c160f13440d67e1041b5b13cdab9dab253569ee7/leggps/training.py#L175
        self.R_params.data = R[self.R_idxs]

    def set_initial_Lambda(self) -> None:
        Lambda = 0.1 * torch.eye(self.n)
        # Jackson had the below for the case that params are provided. but L = cholesky(LL.T)
        # Lambda = torch.linalg.cholesky(Lambda @ Lambda.T)
        self.Lambda_params.data = Lambda[self.Lambda_idxs]

    def set_initial_B(self) -> None:
        B = torch.ones((self.n, self.ell))
        B_torch = 0.5 * B / torch.sqrt(torch.sum(B ** 2, dim=1, keepdim=True))
        self.B.data = B_torch  # dense, no special indexing like above

    @property
    def parameter_count(self) -> int:
        """number of total params"""
        return (
            self.N_idxs.shape[1]  # N params from PEG over z
            + self.R_idxs.shape[1]  # R params from PEG over z
            + self.ell * self.n  # B params, project z->x
            + self.Lambda_idxs.shape[1]  # Lambda params, x's covariance
        )

    # def p2NRBL(self, p):
    #     i = 0

    #     # N!
    #     sz = self.N_idxs.shape[0]
    #     N = tf.scatter_nd(self.N_idxs, p[i : i + sz], (self.ell, self.ell))
    #     i += sz

    #     # R!
    #     sz = self.R_idxs.shape[0]
    #     R = tf.scatter_nd(self.R_idxs, p[i : i + sz], (self.ell, self.ell))
    #     i += sz

    #     # B!
    #     sz = self.ell * self.n
    #     B = tf.reshape(p[i : i + sz], (self.n, self.ell))
    #     i += sz

    #     # Lambda!
    #     sz = self.Lambda_idxs.shape[0]
    #     Lambda = tf.scatter_nd(self.Lambda_idxs, p[i : i + sz], (self.n, self.n))
    #     i += sz

    #     return N, R, B, Lambda

    # @tf.function(autograph=False)
    # def informant(self, ts, x, idxs, p):
    #     """
    #     gradient of log likelihood w.r.t. p
    #     """
    #     with tf.GradientTape() as g:
    #         g.watch(p)
    #         N, R, B, Lambda = self.p2NRBL(p)
    #         nats = legops.leg_log_likelihood_tensorflow(ts, x, idxs, N, R, B, Lambda)
    #     return g.gradient(nats, p)

    # @tf.function(autograph=False)
    # def log_likelihood(self, ts, x, idxs, p):
    #     """
    #     log likelihood
    #     """
    #     N, R, B, Lambda = self.p2NRBL(p)
    #     return legops.leg_log_likelihood_tensorflow(ts, x, idxs, N, R, B, Lambda)

    # def get_initial_guess(self, ts, xs, N=None, R=None, B=None, Lambda=None):
    #     # make up values when nothing is provided
    #     if N is None:
    #         N = np.eye(self.ell)
    #     if R is None:
    #         R = npr.randn(self.ell, self.ell) * 0.2
    #         R = 0.5 * (R - R.T)
    #     if B is None:
    #         B = np.ones((self.n, self.ell))
    #         B = 0.5 * B / np.sqrt(np.sum(B ** 2, axis=1, keepdims=True))
    #     if Lambda is None:
    #         Lambda = 0.1 * np.eye(self.n)

    #     # make 'em nice for us
    #     N = tf.linalg.cholesky(N @ tf.transpose(N))
    #     R = R - tf.transpose(R)
    #     Lambda = tf.linalg.cholesky(Lambda @ tf.transpose(Lambda))

    #     # put it all together
    #     pN = tf.gather_nd(N, self.N_idxs)
    #     pR = tf.gather_nd(R, self.R_idxs)
    #     pB = tf.reshape(B, (self.n * self.ell,))
    #     pL = tf.gather_nd(Lambda, self.Lambda_idxs)
    #     return tf.concat([pN, pR, pB, pL], axis=0)


class CeleriteFamily(LEGFamily):
    def __init__(self, nblocks, n):
        self.nblocks = nblocks
        self.ell = nblocks * 2
        self.n = n

        msk = np.eye(self.ell, dtype=np.bool) + np.diag(
            np.tile([True, False], self.nblocks)[:-1], -1
        )
        self.N_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

        msk = np.diag(np.tile([True, False], self.nblocks)[:-1], -1)
        self.R_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

        msk = np.tril(np.ones((self.n, self.n)))
        self.Lambda_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

        self.psize = (
            self.N_idxs.shape[0]
            + self.R_idxs.shape[0]
            + self.ell * self.n
            + self.Lambda_idxs.shape[0]
        )

    def get_initial_guess(self, ts, xs):
        N = np.eye(self.ell)
        R = npr.randn(self.ell, self.ell) * 0.2
        B = np.ones((self.n, self.ell))
        B = 0.5 * B / np.sqrt(np.sum(B ** 2, axis=1, keepdims=True))
        Lambda = 0.1 * np.eye(self.n)
        N = tf.linalg.cholesky(N @ tf.transpose(N))
        R = R - tf.transpose(R)
        Lambda = tf.linalg.cholesky(Lambda @ tf.transpose(Lambda))

        # put it all together
        pN = tf.gather_nd(N, self.N_idxs)
        pR = tf.gather_nd(R, self.R_idxs)
        pB = tf.reshape(B, (self.n * self.ell,))
        pL = tf.gather_nd(Lambda, self.Lambda_idxs)
        return tf.concat([pN, pR, pB, pL], axis=0)

