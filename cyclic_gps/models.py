# from matplotlib.pyplot import axis
import torch as torch
from torch.optim import Adam
import pytorch_lightning as pl

# import numpy as np
from typing import List, Dict, Tuple, Optional
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from cyclic_gps.cyclic_reduction import mahal_and_det, mahal, decompose, det
import math
import pytorch_lightning as pl

patch_typeguard()


@typechecked
class LEGFamily(pl.LightningModule):

    """
    z \sim PEG(N, R)
    x(t) \sim \mathcal{N}(Bz(t), \Lambda \Lambda^{\top})
    """

    def __init__(self, rank: int, n: int, train: bool = False) -> None:
        # TODO: change n -> obs dim ?
        super().__init__()
        self.rank = rank
        self.n = n

        # everything below and including diagonal in (self.rank, self.rank) mat
        self.N_idxs = self.inds_to_tuple(
            torch.tril_indices(row=self.rank, col=self.rank, offset=0)
        )
        self.N_params = torch.empty(len(self.N_idxs[0]), requires_grad=train)
        torch.nn.init.uniform_(self.N_params)  # initialize (later override)

        # everything below and including lower off-diagonal in (self.rank, self.rank) mat
        self.R_idxs = self.inds_to_tuple(
            torch.tril_indices(row=self.rank, col=self.rank, offset=-1)
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

        self.B = torch.empty((self.n, self.rank))
        torch.nn.init.uniform_(self.B)  # initialize (later override)

        #########################################################################
        self.get_initial_guess()

        # self.peg_precision = compute_PEG_precision()
        # self.

    @staticmethod
    def inds_to_tuple(
        raw_inds: TensorType[2, "num_entries"]
    ) -> Tuple[TensorType["num_entries"], TensorType["num_entries"]]:
        # convert a tensor to a tuple of tensors, to support natural indexing.
        return (raw_inds[0], raw_inds[1])

    def get_initial_guess(self):
        """initialize all the parameters."""
        # latent PEG process params
        self.set_initial_N()
        self.set_initial_R()
        # observation model params
        self.set_initial_B()
        self.set_initial_Lambda()

    def set_initial_N(self) -> None:
        """modify the initial data of self.N_params"""
        I = torch.eye(self.rank)
        # Jackson had the below as torch.linalg.cholesky(N@N.T) but Cholesky of an identity is the identity. https://github.com/jacksonloper/leg-gps/blob/c160f13440d67e1041b5b13cdab9dab253569ee7/leggps/training.py#L189
        N = I @ I.T
        self.N_params.data = N[self.N_idxs]

    def set_initial_R(self) -> None:
        """modify the initial data of self.N_params"""
        R = torch.randn((self.rank, self.rank)) * 0.2
        R = R - R.T  # makes R anti-symetric
        # Jackson had it in two steps depending on whether R is provided or not, but I decided to simplify and get an identical result (hope ot not run into over/underflow issues). see https://github.com/jacksonloper/leg-gps/blob/c160f13440d67e1041b5b13cdab9dab253569ee7/leggps/training.py#L175
        self.R_params.data = R[self.R_idxs]

    def set_initial_Lambda(self) -> None:
        Lambda = 0.1 * torch.eye(self.n)
        # Jackson had the below for the case that params are provided. but L = cholesky(LL.T)
        # Lambda = torch.linalg.cholesky(Lambda @ Lambda.T)
        self.Lambda_params.data = Lambda[self.Lambda_idxs]

    def set_initial_B(self) -> None:
        B = torch.ones((self.n, self.rank))
        B_torch = 0.5 * B / torch.sqrt(torch.sum(B ** 2, dim=1, keepdim=True))
        self.B.data = B_torch  # dense, no special indexing like above

    @property
    def parameter_count(self) -> int:
        """number of total params"""

        count = (
            len(self.N_params)  # N params from PEG over z
            + len(self.R_params)  # R params from PEG over z
            + len(self.Lambda_params)  # B params, project z->x
            + torch.numel(self.B)  # Lambda params, x's covariance
        )
        return count

    def N_from_params(self) -> None:
        N = torch.zeros(self.rank, self.rank, dtype=self.N_params.dtype)
        N[self.N_idxs] = self.N_params
        self.register_buffer("N", N)

    def R_from_params(self) -> None:
        R = torch.zeros(self.rank, self.rank, dtype=self.R_params.dtype)
        R[self.R_idxs] = self.R_params
        self.register_buffer("R", R)

    def Lambda_from_params(self) -> None:
        Lambda = torch.zeros(self.n, self.n, dtype=self.Lambda_params.dtype)
        Lambda[self.Lambda_idxs] = self.Lambda_params
        self.register_buffer("Lambda", Lambda)

    def calc_G(self) -> None:
        """describe G here. it's an important quantity of LEG, needed for cov/precision"""
        G: TensorType[
            "latent_dim", "latent_dim"
        ] = self.N @ self.N.T + self.R - self.R.T
        # add small diag noise
        G += torch.eye(self.N.shape[0], dtype=self.N.dtype) * (1e-5)
        self.register_buffer("G", G)

    @staticmethod
    def calc_Lambda_Lambda_T(Lambda: TensorType["n", "n"]) -> TensorType["n", "n"]:
        if len(Lambda.shape) == 2:
            Lambda_Lambda_T = Lambda @ Lambda.T
            Lambda_Lambda_T += 1e-9 * torch.eye(
                Lambda_Lambda_T.shape[0], dtype=Lambda_Lambda_T.dtype
            )
        else:  # TODO: check what this condition covers
            Lambda_Lambda_T = torch.diag(Lambda ** 2 + 1e-9)
        return Lambda_Lambda_T

    # def distribution(self, inputs, outputs, indices, Sig):
    #     # note: Sig = LambdaLambdaT, have to call the above function for that.
    #     # TODO: type assertions
    #     # get Jd
    #     J_dblocks, J_offblocks = self.compute_PEG_precision(inputs=inputs)

    #     # get JT (TODO: verify)
    #     # Sig v = x -> v = Sig^{-1} x, call it "Sig_inv_x"
    #     Sig_inv_x = torch.transpose(torch.linalg.solve(Sig, inputs.T))  # <-- m x n
    #     Sig_inv_b = torch.linalg.solve(Sig, self.B)  # <-- n x rank
    #     # TODO: check how this adds offset?
    #     offset_adder = torch.einsum("nl,mn->ml", self.B, Sig_inv_x)  # <-- m x rank
    #     # TODO: continute here.
    @typechecked
    def compute_PEG_precision(self, ts: TensorType["num_obs"]):
        """computes the diagonal and offdiag blocks of J precision matrix corresponding to the PEG process """

        diffs = ts[1:] - ts[:-1]  # be careful when extending to input_dim>1
        # exponentiate the diffs, TODO: move to Jackson's efficient method?
        # comes from Definition 1 (diffs are tau in the paper)

        expd = torch.matrix_exp(-0.5 * self.G.unsqueeze(0) * diffs.reshape(-1, 1, 1))
        expdT = torch.transpose(expd, 1, 2)  # exp(X^T) = (exp X)^T
        eye = torch.eye(self.G.shape[0], dtype=expd.dtype)

        # from here on, expdp is called G, and expdT is called G^T in the comments

        # Ax = b --> x = A^{-1}b, therefore
        # (I - G^T G) x = G^T --> x = (I - G^T G)^{-1} G^T
        imgtginvgt = torch.linalg.solve(A=eye.unsqueeze(0) - expdT @ expd, b=expdT)
        imggtinvg = torch.linalg.solve(A=eye.unsqueeze(0) - expd @ expdT, b=expd)

        # clarify how to go from Q in JPC notes to imgtginvgt/imggtinvg

        offdiag_blocks = -imggtinvg

        Dcontrib1 = expd @ imgtginvgt
        Dcontrib2 = expdT @ imggtinvg

        diag_blocks_inner = eye.unsqueeze(0) + Dcontrib1[:-1] + Dcontrib2[1:]
        diag_blocks = torch.cat(
            [
                (eye + Dcontrib2[0]).unsqueeze(0),  # b/c d_0 = \inf
                diag_blocks_inner,
                (eye + Dcontrib1[-1]).unsqueeze(0),  # b/c d_m = \inf
            ],
            dim=0,
        )

        return diag_blocks, offdiag_blocks

        # Get J (perfectly described by its diagonal and off-diagonal blocks)

    @typechecked
    def log_likelihood(
        self,
        ts: TensorType["num_obs"],
        xs: TensorType["num_obs", "obs_dim"],
        idxs: Optional[TensorType["num_obs"]] = None,
    ):
        # Notation:
        # v := B^T LLT^{-1} x
        # Sigma := PEG covariance
        # K := Sigma^{-1} + B^T(LLT)^{-1}B
        # LLT := Lambda(Lambda^T)

        LLT = self.calc_Lambda_Lambda_T(self.Lambda)  # (obs_dim x obs_dim)

        # LL^T is symmetric
        # LL^{T} v = xs^{T} -> v = ((LL^{T})^{-1} xs^{T})^{T} = xs L^{-1} L^{T}
        x_LLT_inv = torch.linalg.solve(A=LLT, B=xs.T).T  # (num_obs x obs_dim)
        LLT_mahal = x_LLT_inv * xs

        # could make this more efficient as lambda is triangular?
        LLT_det = (
            torch.logdet(LLT) * xs.shape[0]
        )  # "real" shape of LLT is block diagional with num_obs blocks each of size (obs_dim x obs_dim)

        # we want the matrix multiplication of B.T @ x_LLT_inv.T = (x_LLT_inv @ B).T
        # so not sure why he doesn't have a transpose here
        v = x_LLT_inv @ self.B
        # v_entries = torch.einsum('nl,mn->ml', self.B, x_LLT_inv) #(m x rank) b/c equal to x LLT_inv B (which is equal to (B^T LLT_inv x^T)^T)

        # v = torch.zeros(ts.shape[0], self.G.shape[0])
        # v = v.scatter_(dim)

        # compute the PEG precision matrix from its parameters and ts:
        Sigma_inv_Rs, Sigma_inv_Os = self.compute_PEG_precision(ts)
        # cyclic reduction decomposition of the PEG precision:
        Sigma_inv_decomp = decompose(Sigma_inv_Rs, Sigma_inv_Os)
        Sigma_inv_det = det(Sigma_inv_decomp)

        LLT_inv_B = torch.linalg.solve(A=LLT, B=self.B)
        B_T_LLT_inv_B = self.B.T @ LLT_inv_B  # trying without einsum

        # note that the "real" B is block diagional with diagional blocks given by B, and same with the "real" LLT
        # so the "real" matrix B_T_LLT_inv_B we are concerned with has diagional blocks equal to B_T_LLT_inv_B
        K_Rs = Sigma_inv_Rs + B_T_LLT_inv_B.unsqueeze(0)
        K_Os = Sigma_inv_Os

        K_mahal, K_det = mahal_and_det(Rs=K_Rs, Os=K_Os, x=v)
        mahal = LLT_mahal - K_mahal

        det = LLT_det + K_det - Sigma_inv_det  # log rules

        return -0.5 * (mahal + det)

    def training_step(self, train_batch, batch_idx):
        t, x = train_batch
        nobs = x.shape[0] * x.shape[1]
        loss = self.log_likelihood(t, x)
        loss = -loss / nobs
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer

    def gradient_log_likelihood(self):
        raise NotImplementedError

    # we have fit as an class method instead of an external function
    # def fit(self, ts, xs):
    #     get_initial_guess() #intializes N, R, B, and Lambda
    #     fit_model_family(self, ts, xs)


# TODO: add implementation for CeleriteFamily
# class CeleriteFamily(LEGFamily):
#     def __init__(self, nblocks: int, n: int) -> None:
#         self.nblocks = nblocks
#         self.rank = nblocks * 2
#         self.n = n

#         msk = np.eye(self.rank, dtype=np.bool) + np.diag(
#             np.tile([True, False], self.nblocks)[:-1], -1
#         )
#         self.N_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

#         msk = np.diag(np.tile([True, False], self.nblocks)[:-1], -1)
#         self.R_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

#         msk = np.tril(np.ones((self.n, self.n)))
#         self.Lambda_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

#         self.psize = (
#             self.N_idxs.shape[0]
#             + self.R_idxs.shape[0]
#             + self.ell * self.n
#             + self.Lambda_idxs.shape[0]
#         )

#     def get_initial_guess(self, ts, xs):
#         N = np.eye(self.ell)
#         R = npr.randn(self.ell, self.ell) * 0.2
#         B = np.ones((self.n, self.ell))
#         B = 0.5 * B / np.sqrt(np.sum(B ** 2, axis=1, keepdims=True))
#         Lambda = 0.1 * np.eye(self.n)
#         N = tf.linalg.cholesky(N @ tf.transpose(N))
#         R = R - tf.transpose(R)
#         Lambda = tf.linalg.cholesky(Lambda @ tf.transpose(Lambda))

#         # put it all together
#         pN = tf.gather_nd(N, self.N_idxs)
#         pR = tf.gather_nd(R, self.R_idxs)
#         pB = tf.reshape(B, (self.n * self.ell,))
#         pL = tf.gather_nd(Lambda, self.Lambda_idxs)
#         return tf.concat([pN, pR, pB, pL], axis=0)

