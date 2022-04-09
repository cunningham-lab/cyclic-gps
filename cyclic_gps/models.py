# from matplotlib.pyplot import axis
import torch as torch
from torch.optim import Adam, LBFGS
import pytorch_lightning as pl

# import numpy as np
from typing import List, Dict, Tuple, Optional
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from cyclic_gps.cyclic_reduction import inverse_blocks, mahal_and_det, decompose, det, solve
from cyclic_gps.model_utils import compute_eG, build_2x2_block, build_3x3_block, gaussian_stitch
import math
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

patch_typeguard()


@typechecked
class LEGFamily(pl.LightningModule):

    """
    z \sim PEG(N, R)
    x(t) \sim \mathcal{N}(Bz(t), \Lambda \Lambda^{\top})
    """

    def __init__(self, rank: int, obs_dim: int, prior_process_noise_level: float = 1.0, prior_length_scale: float = 0.2, train: bool = False, optimizer: str = "ADAM", data_type=torch.float32) -> None:
        # TODO: change n -> obs dim ?
        super().__init__()
        self.rank = rank
        self.obs_dim = obs_dim
        self.prior_process_noise_level = prior_process_noise_level
        self.prior_length_scale = prior_length_scale
        self.optimizer = optimizer
        self.data_type = data_type

        # everything below and including diagonal in (self.rank, self.rank) mat
        self.N_idxs = self.inds_to_tuple(
            torch.tril_indices(row=self.rank, col=self.rank, offset=0)
        )
        self.N_params = torch.nn.Parameter(
            data=torch.zeros(len(self.N_idxs[0]), dtype=data_type), requires_grad=train
        )

        # everything below and including lower off-diagonal in (self.rank, self.rank) mat
        self.R_idxs = self.inds_to_tuple(
            torch.tril_indices(row=self.rank, col=self.rank, offset=-1)
        )
        self.R_params = torch.nn.Parameter(
            data=torch.zeros(len(self.R_idxs[0]), dtype=data_type), requires_grad=train
        )

        # convert to tuple
        # inds = (leg_family.N_idxs[0], leg_family.N_idxs[1])
        # fill in parameters inside mats value
        # zeros.index_put_(indices=inds, values=torch.arange(inds[0].shape).float())

        # everything below and including diagonal in (self.obs_dim, self.obs_dim) mat
        self.Lambda_idxs = self.inds_to_tuple(
            torch.tril_indices(row=self.obs_dim, col=self.obs_dim, offset=0)
        )
        self.Lambda_params = torch.nn.Parameter(
            data=torch.zeros(len(self.Lambda_idxs[0]), dtype=data_type), requires_grad=train
        )

        self.B = torch.nn.Parameter(
            data=torch.zeros((self.obs_dim, self.rank), dtype=data_type), requires_grad=train
        )

        #########################################################################
    
        self.get_initial_guess()

        self.register_model_matrices_from_params()

        # self.peg_precision = compute_PEG_precision()

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
        # TODO: if needed do something like
        # torch.nn.init.uniform_(self.Lambda_params)

    def set_initial_N(self) -> None:
        """modify the initial data of self.N_params"""
        N = torch.eye(self.rank, dtype=self.data_type) * self.prior_process_noise_level
        # Jackson had the below as torch.linalg.cholesky(N@N.T) but Cholesky of an identity is the identity. https://github.com/jacksonloper/leg-gps/blob/c160f13440d67e1041b5b13cdab9dab253569ee7/leggps/training.py#L189
        N = torch.linalg.cholesky(N @ N.T)
        self.N_params.data = N[self.N_idxs]

    def set_initial_R(self) -> None:
        """modify the initial data of self.N_params"""
        R = torch.randn((self.rank, self.rank), dtype=self.data_type)
        R = .5 * (R - R.T) * self.prior_length_scale  # makes R anti-symetric
        R = R - R.T #he does this twice for some reason
        # Jackson had it in two steps depending on whether R is provided or not, but I decided to simplify and get an identical result (hope ot not run into over/underflow issues). see https://github.com/jacksonloper/leg-gps/blob/c160f13440d67e1041b5b13cdab9dab253569ee7/leggps/training.py#L175
        self.R_params.data = R[self.R_idxs]

    def set_initial_Lambda(self) -> None:
        Lambda = 0.1 * torch.eye(self.obs_dim, dtype=self.data_type)
        # Jackson had the below for the case that params are provided. but L = cholesky(LL.T)
        Lambda = torch.linalg.cholesky(Lambda @ Lambda.T)
        self.Lambda_params.data = Lambda[self.Lambda_idxs]

    def set_initial_B(self) -> None:
        B = torch.ones((self.obs_dim, self.rank), dtype=self.data_type)
        B_torch = 0.5 * B / torch.sqrt(torch.sum(B**2, dim=1, keepdim=True))
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
        Lambda = torch.zeros(self.obs_dim, self.obs_dim, dtype=self.Lambda_params.dtype)
        # enforce positive
        Lambda_params = torch.nn.functional.softplus(self.Lambda_params)
        Lambda[self.Lambda_idxs] = Lambda_params
        self.register_buffer("Lambda", Lambda)

    def calc_G(self) -> None:
        """describe G here. it's an important quantity of LEG, needed for cov/precision"""
        G: TensorType["rank", "rank"] = (
            self.N @ self.N.T + self.R - self.R.T
        )
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
            Lambda_Lambda_T = torch.diag(Lambda**2 + 1e-9)
        return Lambda_Lambda_T

    def register_model_matrices_from_params(self) -> None:
        """creates matrices using the current params, registered in buffer."""
        self.Lambda_from_params()
        self.N_from_params()
        self.R_from_params()
        # self.B is already computed since it's dense.
        self.calc_G()  # from N and R

    @typechecked
    def compute_PEG_precision(
        self, ts: TensorType["num_obs"]
    ) -> Tuple[
        TensorType["num_obs", "rank", "rank"],
        TensorType["num_obs_minus_one", "rank", "rank"],
    ]:
        """Computes the OU process' precision matrix. 
        Since it is block-tridiagonal, we only need to compute its diagonal and offdiag blocks.
        Let 
        r$\begin{align*}
        d_{i}= & \begin{cases}
        \infty & i=0\\
        t_{i+1}-t_{i} & i\in\left\{ 1, \dots, m \right\} \\
        \infty & i=m+1
        \end{cases}$
        The first "time difference" is infinity, the last "time difference" is \infty, and the middle one is a simple time difference.
        We treat these three cases separately.  """
        # compute time difference, \tau in JMLR, Definition 1
        diffs = ts[1:] - ts[:-1]  # be careful when extending to input_dim>1

        # TODO: move to Jackson's efficient method?
        # expd:= \exp(-\frac{1}{2} d_{i} G)
        expd = torch.matrix_exp(-0.5 * self.G.unsqueeze(0) * diffs.reshape(-1, 1, 1))
        # for matrix exponentials, \exp(X^T) = (\exp (X))^T
        expdT = torch.transpose(expd, 1, 2)  # \exp(-\frac{1}{2} d_{i} G^{\top})
        eye = torch.eye(self.G.shape[0], dtype=expd.dtype)

        # from here on, expdp is called G, and expdT is called G^T in the comments

        # want to compute
        # Ax = b --> x = A^{-1}b, therefore
        # (I - expd^T expd) x = expd^T --> x = (I - G^T G)^{-1} G^T
        """
        O_{i} & = - (I - e^{-\frac{1}{2}d_{i}G^{T}} e^{-\frac{1}{2}d_{i}G})^{-1} e^{-\frac{1}{2}d_{i}G^{T}}
        we compute 
        -O_{i} = (I - e^{-\frac{1}{2}d_{i}G^{T}} e^{-\frac{1}{2}d_{i}G})^{-1} e^{-\frac{1}{2}d_{i}G^{T}}
        """

        imgtginvgt = torch.linalg.solve(eye.unsqueeze(0) - expdT @ expd, expdT)
        imggtinvg = torch.linalg.solve(eye.unsqueeze(0) - expd @ expdT, expd)

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

    def sample_from_prior(
        self, ts: TensorType["num_obs"]
    ) -> TensorType["num_samples", "obs_dim"]:
        """sample from the prior"""

        # compute the prior covariance matrix, mean is zero.
        diffs = ts[1:] - ts[:-1]  # be careful when extending to input_dim>1
        expd = torch.matrix_exp(-0.5 * self.G.unsqueeze(0) * diffs.reshape(-1, 1, 1))
        expdT = torch.transpose(expd, 1, 2)  # exp(X^T) = (exp X)^T
        eye = torch.eye(self.G.shape[0], dtype=expd.dtype)

    def compute_posterior_precision(self, ts):
        LLT = self.calc_Lambda_Lambda_T(
            self.Lambda
        )
        LLT_inv_B = torch.linalg.solve(LLT, self.B)
        # (obs_dim X obs_dim) * (obs_dim X rank) ->(obs_dim X rank)
        B_T_LLT_inv_B = self.B.T @ LLT_inv_B
        # (rank x obs_dim) (obs_dim X rank) -> (rank X rank) trying without einsum
        Sigma_inv_Rs, Sigma_inv_Os = self.compute_PEG_precision(ts)
        # note that the "real" \tilde{B} is block-diagional with diagional blocks given by B, and same with the "real" \tilde{\Lambda}
        # so the "real" matrix B_T_LLT_inv_B we are concerned with has diagional blocks equal to B_T_LLT_inv_B
        K_Rs = Sigma_inv_Rs + B_T_LLT_inv_B.unsqueeze(0) # add to the diag blocks of the PEG precision
        K_Os = Sigma_inv_Os 

        return K_Rs, K_Os

    def compute_v(self, xs):
        LLT = self.calc_Lambda_Lambda_T( #could be moved to input to the function
            self.Lambda
        )
        x_LLT_inv = torch.linalg.solve(
            LLT, xs.T #effectively broadcasting over blocks of the large tilde lambda matrix
        ).T
        v = (
            x_LLT_inv @ self.B
        )  # -> (num_obs X obs_dim) X (obs_dim X rank) -> (num_obs X rank)
        return v

    def compute_insample_posterior(
        self,
        ts: TensorType["num_obs"],
        xs: TensorType["num_obs", "obs_dim"],
    ):
        #self.register_model_matrices_from_params()
        posterior_precision = {}
        posterior_precision["Rs"], posterior_precision["Os"] = self.compute_posterior_precision(ts)
        posterior_precision_cr = decompose(**posterior_precision)
        v = self.compute_v(xs)
        posterior_mean = solve(posterior_precision_cr, v) 
        posterior_cov = {}
        #Question: is posterior covariance tridiagional or close?
        posterior_cov["Rs"], posterior_cov["Os"] = inverse_blocks(
            posterior_precision_cr
        )
        return posterior_mean, posterior_cov

    @typechecked
    def log_likelihood(
        self,
        ts: TensorType["num_obs"],
        xs: TensorType["num_obs", "obs_dim"]
    ) -> TensorType([]):
        # Notation:
        # v := B^T LLT^{-1} x
        # Sigma := PEG covariance
        # K := Sigma^{-1} + B^T(LLT)^{-1}B
        # LLT := Lambda(Lambda^T)
        # tildeL := num_obs * obs_dim X num_obs * obs_dim matrix, with diagonal blocks = LLT

        # grab params and put these into matrices accessible as self.Lambda, self.N ...
        self.register_model_matrices_from_params()
        LLT = self.calc_Lambda_Lambda_T(
            self.Lambda
        )  # per-measurment observation noise, shape(obs_dim x obs_dim)
        # LL^T is symmetric ->
        # LL^{T} v = xs^{T} -> v = ((LL^{T})^{-1} xs^{T})^{T} = xs  (LL^{T})^{-1} --> (num_obs X obs_dim) times (obs_dim X obs_dim)
        x_LLT_inv = torch.linalg.solve(
            LLT, xs.T
        ).T  # (obs_dim x obs_dim) * (num_obs x obs_dim)^{T} -> (obs_dim x num_obs)^{T} -> (num_obs X obs_dim)
        # torch.linalg.solve standard  A= (2X2), B= (2X1) -> v = (2X1). if you have 100 in the last dimension of b, you solve 100 systems.
        LLT_mahal = torch.sum(
            x_LLT_inv * xs
        )  # (num_obs X obs_dim) * (num_obs X obs_dim) reduce to scalar
        # effective dot product. first elemntewise multiplication, then sum. TODO: think again how this works out.

        # final result: x^t (LL^T)^{-1} x -> typically: (1 X obs_dim) X (obs_dim X obs_dim) X (obs_dim X 1) -> (1 X 1)
        # (v^{T} K^{-1} v) but we're computing v K^{-1} v^{T}

        # could make this more efficient as lambda is triangular?
        # =|\tilde{Lambda}|. take the logdet of each block, and multiply by the number of blocks. equivalent to summing
        LLT_det = (
            torch.logdet(2 * math.pi * LLT) * xs.shape[0]
        )  # "real" shape of LLT is block diagional with num_obs blocks each of size (obs_dim x obs_dim)

        # we want the matrix multiplication of B.T @ x_LLT_inv.T = (x_LLT_inv @ B).T
        # so not sure why he doesn't have a transpose here
        v = (
            x_LLT_inv @ self.B
        )  # -> (num_obs X obs_dim) X (obs_dim X rank) -> (num_obs X rank)
        # v_entries = torch.einsum('nl,mn->ml', self.B, x_LLT_inv) #(m x rank) b/c equal to x LLT_inv B (which is equal to (B^T LLT_inv x^T)^T)

        # v = torch.zeros(ts.shape[0], self.G.shape[0])
        # v = v.scatter_(dim)

        # compute the PEG precision matrix from its parameters and ts:
        Sigma_inv_Rs, Sigma_inv_Os = self.compute_PEG_precision(ts)
        # cyclic reduction decomposition of the PEG precision:
        Sigma_inv_decomp = decompose(Sigma_inv_Rs, Sigma_inv_Os)
        # compute determinant using the decomposition results:
        Sigma_inv_det = det(Sigma_inv_decomp)  # = |\Sigma^{-1}|

        LLT_inv_B = torch.linalg.solve(LLT, self.B)
        # (obs_dim X obs_dim) * (obs_dim X rank) ->(obs_dim X rank)
        B_T_LLT_inv_B = self.B.T @ LLT_inv_B
        # (rank x obs_dim) (obs_dim X rank) -> (rank X rank) trying without einsum

        # note that the "real" \tilde{B} is block-diagional with diagional blocks given by B, and same with the "real" \tilde{\Lambda}
        # so the "real" matrix B_T_LLT_inv_B we are concerned with has diagional blocks equal to B_T_LLT_inv_B
        K_Rs = Sigma_inv_Rs + B_T_LLT_inv_B.unsqueeze(0)
        # add to the diag blocks of the PEG precision
        K_Os = Sigma_inv_Os  # don't touch off-diags.

        # now we have a block-tridiagonal matrix K which we can operate on using cyclic reduction
        K_mahal, K_det = mahal_and_det(Rs=K_Rs, Os=K_Os, x=v)  # giving us two scalars.
        mahal = LLT_mahal - K_mahal
        #print("LLT mahal: {}".format(LLT_mahal))
        #print("LLT det: {}, Post det: {}, PEG det: {} ".format(LLT_det, K_det, Sigma_inv_det))

        log_det = LLT_det + K_det - Sigma_inv_det  # log rules

        return -0.5 * (mahal + log_det)

    def training_step(self, train_batch, batch_idx):
        t, x = train_batch
        nobs = x.shape[0] * x.shape[1] * x.shape[2]
        # eliminate the batch dimension of 1
        loss = self.log_likelihood(t.squeeze(0), x.squeeze(0))
        loss = -loss / nobs  # TODO: why divide by n_obs?
        self.log("NLL", loss)
        return loss

    
    def configure_optimizers(self):
        if self.optimizer == "ADAM":
            optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-2)
        elif self.optimizer == "BFGS":
            optimizer = LBFGS(filter(lambda p: p.requires_grad, self.parameters()), lr=0.1, max_iter=20)
        scheduler = ReduceLROnPlateau(optimizer, "min")

        # optimizer = Adam([self.N_params, self.R_params, self.Lambda_params, self.B])
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "NLL"}

    def forecast(
        self, 
        eG: TensorType["rank", "rank"], 
        ip_mean: TensorType["rank"],
        ip_cov: TensorType["rank", "rank"]
    ):
        #eG: prior covariance
        #ip_mean: insample posterior mean
        #ip_cov: insample posterior covariance (diagional and off diagional blocks of it)
        nones=(None,)*(len(eG.shape)-2)
        I = torch.eye(self.rank)[nones]   #.expand_as(eG)
        joint_mean = torch.zeros(self.rank*2)[nones] #
        joint_cov = build_2x2_block(I, eG.T, eG, I)
        return gaussian_stitch(joint_mean, joint_cov, ip_mean, ip_cov)

    def interpolate(
        eG1: TensorType["rank", "rank"],
        eG2: TensorType["rank", "rank"],
        prev_ip_mean: TensorType["rank"],
        prev_ip_cov_diag: TensorType["rank", "rank"],
        prev_ip_cov_offdiag,
        next_ip_mean: TensorType["rank"],
        next_ip_cov_diag: TensorType["rank", "rank"]
    ):
        """
        p(z_1, z_3|x) = N([prev_ip_mean, next_ip_mean], 
                        [[prev_ip_cov_diag, ])

        """
        nones = (None,)*(len(eG1.shape)-2)
        I = torch.eye(self.rank)[nones]

        joint_latent_mean = torch.zeroes(self.rank*3)[nones]

        eG3 = eG1@eG2 #computes exp{-t_3 - t_1}G
        joint_latent_cov = build_3x3_block( #Why is different than what Jackson has in his code? It alligns with his notes
            I, eG1.T, eG3.T,
            eG1, I, eG2.T,
            eG3, eG2, I
        )
        
        joint_ip_mean = torch.cat([prev_ip_mean, next_ip_mean], dim=0)

        joint_ip_cov = build_2x2_block(
            prev_ip_cov_diag, prev_ip_cov_offdiag.T, #note that the offdiagional blocks of the insample posterior give the cross covariance between one time step and the next timestep, and since this is the lower off diagonal we want the transpose to get the forwards direction
            prev_ip_cov_offdiag, next_ip_cov_diag
        )

        return gaussian_stitch(joint_latent_mean, joint_latent_cov, joint_ip_mean, joint_ip_cov)
   
    
    def intercast(
        self, 
        ip_mean, 
        ip_cov, 
        ts, 
        target_ts, 
        thresh=1e-10
    ):
        '''
        ip_mean: (num_obs x rank) insample_posterior_mean
        ip_cov: (num_obs x rank x rank) insample_posterior_cov
        ts: (num_obs) time steps correponding to observations
        target_ts: (num_preds) time steps to make predictions at
        '''
        G_val, G_vec = torch.linalg.eig(self.G)
        G_vec_inv = torch.linalg.inv(G_vec)
        I = torch.eye(self.rank)
        assert (target_ts[1:]-target_ts[:-1]>0).all() # make sure target_ts is sorted
        new_loc_idxs = torch.searchsorted(ts, target_ts)
        means, variances = [], []
        for i, idx in enumerate(new_loc_idxs):
            if idx==0: # forecasting backwards
                #search sorted will return 0 if the item we are looking for is before the 
                #first element of target_ts or equal to it, so we need to check for 2nd case
                if torch.allclose(target_ts[i], ts[0]):
                    m, v = ip_mean[0], ip_cov["Rs"][0]
                else:
                    diff = (ts[0] - target_ts[i])
                    eG = compute_eG(G_val, G_vec, G_vec_inv, torch.tensor(diff))[0]
                    print("eG shape")
                    print(eG.shape)
                    eG = eG.T
                    print(eG.shape)
                    m, v = self.forecast(eG, ip_mean[0], ip_cov["Rs"][0])
            
            elif idx==ts.shape[0]: #forecasting forwards
                if torch.allclose(target_ts[i], ts[-1]): #last value
                    m, v = ip_mean[-1], ip_cov["Rs"][-1]
                else:
                    diff = (target_ts[i] - ts[-1])
                    eG = compute_eG(G_val, G_vec, G_vec_inv, torch.tensor(diff))[0]
                    m, v = self.forecast(eG, ip_mean[-1], ip_cov["Rs"][-1])
            else:
                #interpolating
                if torch.allclose(target_ts[i], ts[-1]): #checks the case where idx == ts.shape[0] - 1 and we just want the last value exactly
                    m, v = ip_mean[-1], ip_cov["Rs"][-1]
                else:
                    diff1 = target_ts[i] - ts[idx-1] #diff between target and previous train ts
                    diff2 = ts[idx] - target_ts[i] #diff between next train ts and target
                    eg1 = compute_eG(G_val, G_vec, G_vec_inv, torch.tensor(diff1))[0]
                    eg2= compute_eG(G_val, G_vec, G_vec_inv, torch.tensor(diff2))[0]
                    m, v = self.interpolate(
                        eg1, 
                        eg2,
                        ip_mean[idx-1],
                        ip_cov["Rs"][idx-1],
                        ip_cov["Os"][idx-1],
                        ip_mean[idx],
                        ip_cov["Rs"][idx]
                    )    
            means.append(m)
            variances.append(v)

        return torch.stack(means, dim=0), torch.stack(variances, dim=0)
    
    def predictive_posterior(self, ts, xs, target_ts):
        '''
        ts: (num_obs) time steps correponding to observations
        xs : (num_obs x obs_dim) observations
        target_ts: (num_preds) time steps to make predictions at 
        
        Output:
        - means: (num_preds x rank) E[Z(target_ts[i]) |X] for each i
        - variance: (num_preds x rank x rank) (I think) Cov(Z(target_ts[i]) |X) for each i
        '''
        insample_posterior_mean, insample_posterior_cov = self.compute_insample_posterior(ts, xs)
        predictive_posterior_means, predicted_posterior_variances = self.intercast(insample_posterior_mean, insample_posterior_cov, ts, target_ts)
        return predictive_posterior_means, predicted_posterior_variances

    def make_predictions(self, ts, xs, target_ts):
        '''
        ts: (num_obs) time steps correponding to observations
        xs : (num_obs x obs_dim) observations
        target_ts: (num_preds) time steps to make predictions at 
        
        Output:
        - mean: (num_preds x obs_dim) predicted datapoint for each target_ts
        - variance: (num_preds x obs_dim x obs_dim) covariance for each predicted datapoint
        '''
        #self.register_model_matrices_from_params()
        self.calc_G()
        predictive_posterior_means, predicted_posterior_variances = self.predictive_posterior(ts, xs, target_ts)
        prediction_means = predictive_posterior_means @ self.B.T
        prediction_covariances = self.B.unsqueeze(0) @ predicted_posterior_variances @ self.B.T.unsqueeze(0)

        return prediction_means, prediction_covariances

    def gradient_log_likelihood(self):
        raise NotImplementedError

   
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
