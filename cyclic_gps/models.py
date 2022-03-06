from matplotlib.pyplot import axis
import torch as torch
import numpy as np
from typing import List, Dict, Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
class LEGFamily(torch.nn.Module):
    # TODO: inherit from torch.nn.module? or lightning module?

    """
    z \sim PEG(N, R)
    x(t) \sim \mathcal{N}(Bz(t), \Lambda \Lambda^{\top})
    """

    def __init__(self, ell: int, n: int, train: bool = False) -> None:
        # TODO: change ell -> rank, and n -> obs dim ?
        super().__init__()
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

        count = (
            len(self.N_params)  # N params from PEG over z
            + len(self.R_params)  # R params from PEG over z
            + len(self.Lambda_params)  # B params, project z->x
            + torch.numel(self.B)  # Lambda params, x's covariance
        )
        return count

    def N_from_params(self) -> None:
        N = torch.zeros(self.ell, self.ell, dtype=self.N_params.dtype)
        N[self.N_idxs] = self.N_params
        self.register_buffer("N", N)

    def R_from_params(self) -> None:
        R = torch.zeros(self.ell, self.ell, dtype=self.R_params.dtype)
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

    def distribution(self, inputs, outputs, indices, Sig):
        # note: Sig = LambdaLambdaT, have to call the above function for that.
        # TODO: type assertions
        # get Jd
        J_dblocks, J_offblocks = self.exponentiate_generator(inputs=inputs)

        # get JT (TODO: verify)
        # Sig v = x -> v = Sig^{-1} x, call it "Sig_inv_x"
        Sig_inv_x = torch.transpose(torch.linalg.solve(Sig, inputs.T))  # <-- m x n
        Sig_inv_b = torch.linalg.solve(Sig, self.B)  # <-- n x ell
        # TODO: check how this adds offset?
        offset_adder = torch.einsum("nl,mn->ml", self.B, Sig_inv_x)  # <-- m x ell
        # TODO: continute here.

    def exponentiate_generator(self, inputs):
        """computes the diagonal and offdiag blocks of J precision matrix corresponding to the PEG process """
        # TODO: educate ourselves on the computations here
        diffs = inputs[1:] - inputs[:-1]  # be careful when extending to input_dim>1
        # exponentiate the diffs, TODO: move to Jackson's efficient method?
        # comes from Definition 1 (diffs are tau in the paper)
        expd = torch.matrix_exp(-0.5 * self.G.unsqueeze(0) * diffs.reshape(-1, 1, 1))
        expdT = torch.transpose(expd, 1, 2)  # exp(X^T) = (exp X)^T
        eye = torch.eye(self.G.shape[0], dtype=expd.dtype)

        # TODO: replace names after better understanding
        # Ax = b --> x = A^{-1}b, therefore
        # (I - G^T G) x = G^T --> x = (I - G^T G)^{-1} G^T
        imgtgigt = torch.linalg.solve(A=eye.unsqueeze(0) - expdT @ expd, b=expdT)
        imggtig = torch.linalg.solve(A=eye.unsqueeze(0) - expd @ expdT, b=expd)

        offdiag_blocks = -imggtig
        # Dcontrib1[-1] connects ts[-2] to ts[-1], and isn't applic to 0
        Dcontrib1 = expd @ imgtgigt
        # Dcontrib2[0] connects ts[0] to ts[1], and isn't applicable to -1
        Dcontrib2 = expdT @ imggtig

        # I + everything but Dcontrib1[-1] + everything but Dcontrib2[0]
        diag_blocks_inner = eye.unsqueeze(0) + Dcontrib1[:-1] + Dcontrib2[1:]
        diag_blocks = torch.cat(
            [
                (eye + Dcontrib2[0]).unsqueeze(0),
                diag_blocks_inner,
                (eye + Dcontrib1[-1]).unsqueeze(0),
            ],
            dim=0,
        )

        return diag_blocks, offdiag_blocks

        # Get J (perfectly described by its diagonal and off-diagonal blocks)

    def log_likelihood(self):
        raise NotImplementedError

    def gradient_log_likelihood(self):
        raise NotImplementedError


# TODO: add implementation for CeleriteFamily
# class CeleriteFamily(LEGFamily):
#     def __init__(self, nblocks: int, n: int) -> None:
#         self.nblocks = nblocks
#         self.ell = nblocks * 2
#         self.n = n

#         msk = np.eye(self.ell, dtype=np.bool) + np.diag(
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

