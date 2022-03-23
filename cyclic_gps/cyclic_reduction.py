from math import ceil, floor
import torch
from torchtyping import TensorType, patch_typeguard
import numpy as np
from typing import Tuple, List, Union
from typeguard import typechecked
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import NotPSDError

patch_typeguard()


def UU_T(diags, offdiags):
    n = diags.shape[0]
    m = offdiags.shape[0]

    # if n == m we have a non-square matrix
    if n == m:
        # matrix multiplication between each block and its transpose
        tq = torch.einsum("ijk,ilk->ijl", diags, diags)
        tq += torch.einsum("ijk,ilk->ijl", offdiags, offdiags)
        # matrix multiplication between offdiags and diag transpose
        offdiags = torch.einsum("ijk,ilk->ilj", offdiags[:-1], diags[1:])
        return tq, offdiags

    else:
        # matrix multiplication between each block and its transpose
        leaf1 = torch.einsum("ijk,ilk->ijl", diags, diags)
        leaf2 = torch.einsum("ijk,ilk->ijl", offdiags, offdiags)
        # adding the "squares" on the diagional offset by one
        temp = leaf1[:-1] + leaf2
        tq = torch.cat([leaf1[:-1] + leaf2, leaf1[-1].unsqueeze(0)], dim=0)
        # matrix multiplication between offdiags and diag transpose
        offdiags = torch.einsum("ijk,ilk->ilj", offdiags, diags[1:])
        return tq, offdiags


def Ux(diags, offdiags, x):
    """
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags
    We would like to compute U@x
    """
    n = diags.shape[0]
    m = offdiags.shape[0]

    # non-square matrix
    if n == m:
        return torch.einsum("ijk,ik...->ij", diags, x[:-1]) + torch.einsum(
            "ijk,ik...->ij", offdiags, x[1:]
        )
    else:
        leaf1 = torch.einsum("ijk,ik...->ij", diags, x)
        leaf2 = torch.einsum("ijk,ik...->ij", offdiags, x[1:])
        leaf_sum = leaf1[:-1] + leaf2
        # remembering the final row of the matrix which just has a diagional element and not an upper diagional element
        return torch.cat([leaf_sum, leaf1[-1].unsqueeze(0)], dim=0)


def U_Tx(diags, offdiags, x):
    """
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags
    We would like to compute U.T@x
    """

    n = diags.shape[0]
    m = offdiags.shape[0]

    # non-square matrix
    if n == m:
        leaf1 = torch.einsum("ikj,ik...->ij", diags, x)
        leaf2 = torch.einsum("ikj,ik...->ij", offdiags, x)
        return torch.cat(
            [leaf1[0].unsqueeze(0), leaf1[1:] + leaf2[:-1], leaf2[-1].unsqueeze(0)],
            dim=0,
        )
    else:
        leaf1 = torch.einsum("ikj,ik...->ij", diags, x)  # double check ellipsis
        leaf2 = torch.einsum("ikj,ik...->ij", offdiags, x[:-1])
        return torch.cat(
            [leaf1[0].unsqueeze(0), leaf1[1:] + leaf2], dim=0
        )  # unsqueezing again


def SigU(sig_dblocks, sig_offdblocks, u_dblocks, u_offdblocks):
    """
    Let Sig be a symmetric block-tridiagonal matrix whose
    - diagonal blocks are sig_dblocks
    - lower off-diagonals are sig_offdblocks
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by u_dblocks
    - upper off-diagonals are given by u_offdblocks
    We would like to compute block-tridiagonal blocks of Sig @ U
    """

    # square matrix
    if u_dblocks.shape[0] == u_offdblocks.shape[0] + 1:
        main_diagional = torch.cat(
            [
                (sig_dblocks[0] @ u_dblocks[0]).unsqueeze(0),
                torch.matmul(sig_dblocks[1:], u_dblocks[1:])
                + torch.matmul(sig_offdblocks, u_offdblocks),
            ],
            axis=0,
        )

        upper_diagional = torch.matmul(sig_dblocks[:-1], u_offdblocks) + torch.matmul(
            torch.transpose(sig_offdblocks, 1, 2), u_dblocks[1:]
        )  # make sure transposing right dimensions

    # non-square matrix
    else:
        main_diagional = torch.cat(
            [
                (sig_dblocks[0] @ u_dblocks[0]).unsqueeze(0),
                torch.matmul(sig_dblocks[1:], u_dblocks[1:])
                + torch.matmul(sig_offdblocks, u_offdblocks[:-1]),
            ],
            axis=0,
        )

        upper_diagional = torch.cat(
            [
                torch.matmul(sig_dblocks[:-1], u_offdblocks[:-1])
                + torch.matmul(torch.transpose(sig_offdblocks, 1, 2), u_dblocks[1:]),
                (sig_dblocks[-1] @ u_offdblocks[-1]).unsqueeze(0),
            ],
            axis=0,
        )

    return main_diagional, upper_diagional


def UtV_diags(
    u_dblocks: TensorType["num_blocks", "block_dim", "block_dim"],
    u_offdblocks: TensorType["num_off_blocks", "block_dim", "block_dim"],
    v_dblocks: TensorType["num_blocks", "block_dim", "block_dim"],
    v_offdblocks: TensorType["num_off_blocks", "block_dim", "block_dim"],
):
    """
    U is upper didiagonal with
    - diagional blocks are u_dblocks
    - upper off-diagionals are u_offdblocks
    V is upper didiagonal with
    - diagional blocks are v_dblocks
    - upper off-diationals are v_offdblocks
    We want the diagonal blocks of U.T @ V
    """

    # square matrix
    if u_dblocks.shape[0] == u_offdblocks.shape[0] + 1:
        return torch.cat(
            [
                (u_dblocks[0].T @ v_dblocks[0]).unsqueeze(0),
                torch.matmul(torch.transpose(u_dblocks[1:], 1, 2), v_dblocks[1:])
                + torch.matmul(torch.transpose(u_offdblocks, 1, 2), v_offdblocks),
            ],
            axis=0,
        )

    # non-square matrix
    else:
        return torch.cat(
            [
                (u_dblocks[0].T @ v_dblocks[0]).unsqueeze(0),
                torch.matmul(torch.transpose(u_dblocks[1:], 1, 2), v_dblocks[1:])
                + torch.matmul(
                    torch.transpose(u_offdblocks[:-1], 1, 2), v_offdblocks[:-1]
                ),
                (u_offdblocks[-1].T @ v_offdblocks[-1]).unsqueeze(0),
            ],
            axis=0,
        )


def interleave(a, b):
    """
    V=np.zeros((a.shape[0]+b.shape[0],)+a.shape[1:])
    V[::2]=a
    V[1::2]=b
    """
    a_shape = a.shape
    b_shape = b.shape
    n = a_shape[0]
    m = b_shape[0]
    if n < m:
        shp = (n * 2,) + a_shape[1:]
        first_part = torch.reshape(torch.stack([a, b[:n]], dim=1), shape=shp)
        last_bit = b[n:]
        return torch.cat([first_part, last_bit], dim=0)
    else:
        shp = (m * 2,) + b_shape[1:]
        first_part = torch.reshape(torch.stack([a[:m], b], dim=1), shape=shp)
        last_bit = a[m:]
        return torch.cat([first_part, last_bit], dim=0)


@typechecked
def decompose_step(
    Rs: TensorType["num_dblocks", "block_dim", "block_dim"],
    Os: TensorType["num_offdblocks", "block_dim", "block_dim"],
) -> Tuple[
    Tuple[
        int,
        TensorType[-1, "block_dim", "block_dim"],
        TensorType[-1, "block_dim", "block_dim"],
        TensorType[-1, "block_dim", "block_dim"],
    ],
    Tuple[
        TensorType[-1, "block_dim", "block_dim"],
        TensorType[-1, "block_dim", "block_dim"],
    ],
]:
    # We're not strongly asserting the number of blocks, because it depends on whether we we have an even number of diagonal blocks.
    # upper and lower off-diags should have one less block
    num_dblocks = Rs.shape[0]
    num_offdblocks = Os.shape[0]
    assert num_dblocks == num_offdblocks + 1
    num_dblocks = Rs.shape[0]
    Rs_even = Rs[::2]  # R_0, R_2, ...

    Ks_even = psd_safe_cholesky(Rs_even)  # Cholesky per diag block

    # try:
    #     Ks_even = psd_safe_cholesky(Rs_even)  # Cholesky per diag block
    # except:
    #     print(Rs_even)

    Os_even = Os[::2]  # O_0, O_2, ...
    Os_odd = Os[1::2]  # O_1, O_3, ...

    Os_even_T = Os_even.transpose(1, 2)

    # Could have fewer off diagional blocks than diagional blocks if overall matrix is square
    N2 = Os_even.shape[0]
    # we use lower triangular solves since Ks_even (D in the paper) is lower triangular -- Cholesky on R blocks
    F = torch.transpose(
        torch.triangular_solve(input=Os_even_T, A=Ks_even[:N2], upper=False)[0], 1, 2
    )
    # Why is is Os_odd not transposed?
    G = torch.transpose(
        torch.triangular_solve(input=Os_odd, A=Ks_even[1::][:N2], upper=False)[0], 1, 2
    )

    UU_T_diags, UU_T_offdiags = UU_T(F, G)

    # Constructing J~
    Rs = Rs[1::2] - UU_T_diags
    Os = -UU_T_offdiags

    # Check that output batch dims make sense
    check_decompose_loop_outputs(num_dblocks, Ks_even, F, G, Rs, Os)

    return (num_dblocks, Ks_even, F, G), (Rs, Os)


def check_decompose_loop_outputs(num_dblocks, Ks_even, F, G, Rs, Os):
    # if even number of diagonal blocks:
    if num_dblocks % 2 == 0:
        assert (
            Ks_even.shape[0] == num_dblocks // 2
            and F.shape[0] == num_dblocks // 2
            and G.shape[0] == num_dblocks // 2 - 1
            and Rs.shape[0] == num_dblocks // 2
            and Os.shape[0] == num_dblocks // 2 - 1
        )
    # if odd number of diagonal blocks, and more than one such block:
    elif num_dblocks % 2 != 0 & num_dblocks > 1:
        assert (
            Ks_even.shape[0] == ceil(num_dblocks // 2)
            and F.shape[0] == floor(num_dblocks // 2)
            and G.shape[0] == floor(num_dblocks // 2)
            and Rs.shape[0] == floor(num_dblocks // 2)
            and Os.shape[0] == floor(num_dblocks // 2) - 1
        )


# J is a symmetric block tridiagional matrix with R blocks on the main diagional, and O blocks below it
# diagonal block matrices must be invertible


@typechecked
def decompose(
    Rs: TensorType["num_dblocks", "block_dim", "block_dim"],
    Os: TensorType["num_offdblocks", "block_dim", "block_dim"],
):
    num_dblocks = Rs.shape[0]  # of the original matrix
    block_dim = Rs.shape[1]  # for leggp precision mat, this int is set by model rank
    Ds = []
    Fs = []
    Gs = []
    ms = []
    while len(Rs) > 1:
        # we keep splitting to evens and odds
        (num_dblocks, Ks_even, F, G), (Rs, Os) = decompose_step(Rs, Os)
        ms += [num_dblocks]  # overwritten, for the submatrix we're looking at
        Ds += [Ks_even]
        Fs += [F]
        Gs += [G]

    Ds += [psd_safe_cholesky(Rs)]
    ms += [1]

    return torch.tensor(ms), Ds, Fs, Gs


def halfsolve(decomp, y: TensorType["num_blocks", "block_dim"]):
    ms, Ds, Fs, Gs = decomp

    # ytilde is our effective y for each L~ as we recurr
    ytilde = y
    xs = []
    for i in range(ms.shape[0]):
        # effectively doing: y = P_m \tilde{y}, i.e., extract odd indices.
        y = ytilde[::2]
        # taking the ith diagonal block of D, which is a lower triangular matrix (from Cholesky),
        # solving D x_1 = y. TODO: this is done in a loop, but the diag blocks are independent. can we parallelize?
        xs.append(
            torch.triangular_solve(input=y.unsqueeze(-1), A=Ds[i], upper=False)[0][
                ..., 0
            ]
        )
        # if we have more than one block in y
        if ytilde.shape[0] > 1:
            # plug in x_1 you obtained above to the re-arranged equation for x_2.
            # i.e., \tilde{L} x_2 = Q_m b - U x_1
            # ytilde[1::2] = Q_m y, selects the even entries.
            ytilde = ytilde[1::2] - Ux(Fs[i], Gs[i], xs[-1])
            # we again need to solve L^{\top}x= \tilde{y}, so \tilde{y} will go into the for loop above.
        else:
            break

    return xs


def backhalfsolve(decomp, ycrr):
    """
    Input:
    - decomp, the cyclic reduction representation of a block tridiagonal matrix
      with nchain diagonal blocks, each of size nblock
    - ycrr, the cyclic reduction representation of a vector y:
    returns the value of T_n^T L^-T y //My Question: What does T_n mean here?
    """
    ms, Ds, Fs, Gs = decomp

    # bottom up recursion: start with the base case, corresponds to the final elements of D and b
    ytilde = ycrr[-1]
    # solving L~^T(Q_m)x = b_2
    # base case is L~ = D[-1], and this corresponds to the final D block, and it's lower triangular b/c its just a cholesky decomp
    xs = torch.triangular_solve(
        input=ytilde.unsqueeze(-1), A=Ds[-1], transpose=True, upper=False
    )[0][
        ..., 0
    ]  # double check tf.linalg.triangular_solve documentation for adjoint

    # up the tower of blocks
    for i in range(1, ms.shape[0] + 1):
        # makes sure we haven't reached the top of the D-blocks
        if len(ycrr) - i - 1 >= 0:
            # next two lines correspond to equation D^T(P_m)x = b_1 - U^T(Q_m)x
            ytilde = ycrr[-i - 1] - U_Tx(Fs[-i], Gs[-i], xs)
            x_even = torch.triangular_solve(
                input=ytilde.unsqueeze(-1), A=Ds[-i - 1], transpose=True, upper=False
            )[0][..., 0]

            # Note that xs corresponds to (Q_m)x, which is x_odd for all solutions of x below this point in the recurrsion,
            # so in order to get the right x vector for this point in the recurrsion we need to interleave the even and odd solutions
            xs = interleave(x_even, xs)  # unsqueeze(0)
        else:
            break

    return xs


def mahal_and_det(
    Rs: TensorType["num_dblocks", "block_dim", "block_dim"],
    Os: TensorType["num_offdblocks", "block_dim", "block_dim"],
    x: TensorType["num_dblocks", "block_dim"],
):
    """
    Let J denote a symmetric positive-definite block-tridiagonal matrix whose
    - diagonal blocks are given by Rs
    - lower off-diagonal blocks are given by Os
    returns
    - mahal: x^T J^-1 x
    - det: log |J|
    We here obtain a cyclic reduction representation of J which can be used
    for further analysis.
    """
    ms = []
    Ds = []
    Fs = []
    Gs = []

    det = 0
    mahal = 0

    ytilde = x

    # What is the idea of this "mahal" function?
    # Compute ||L^{-1}x||^2 for a given J
    # So we show that this is equal to ||D^{-1}(P_m x)||^2 + ||L~^{-1}Q_m y - UD^{-1}(P_m y)||^2
    # Since for a given decompose_step we compute D, U, and J~, we can compute this recursively
    # by first computing ||D^{-1}(P_m x)||^2 and then calling the function recursively
    # with J~ and v = Q_m y - UD^{-1}(P_m y) which will give us ||L~^{-1}v||^2

    while Rs.shape[0] > 1:
        # get the decomposition of D and U for this state of the cyclic reduction recursion
        (numblocks, Ks_even, F, G), (Rs, Os) = decompose_step(Rs, Os)

        # det
        det += torch.sum(torch.log(torch.diagonal(Ks_even, dim1=1, dim2=2)))

        # computes D^{-1}(P_m y)
        y = ytilde[::2]
        newx = torch.triangular_solve(input=y.unsqueeze(-1), A=Ks_even, upper=False)[0][
            ..., 0
        ]
        mahal += torch.sum(newx**2)

        # computes Q_m y - UD^{-1}(P_m y)
        ytilde = ytilde[1::2] - Ux(F, G, newx)

    Ks_even = psd_safe_cholesky(Rs)
    det += torch.sum(torch.log(torch.diagonal(Ks_even, dim1=1, dim2=2)))

    y = ytilde[::2]
    newx = torch.triangular_solve(input=y.unsqueeze(-1), A=Ks_even, upper=False)[0][
        ..., 0
    ]
    mahal += torch.sum(newx**2)

    return mahal, 2 * det


def solve(decomp, y: TensorType["num_blocks", "block_dim"]):
    v = halfsolve(decomp, y)
    w = backhalfsolve(decomp, v)
    return w


def det(decomp):
    # TODO: change name? this is logdet.
    # computes log |J| given L which is a cyclic reduction decomposition of J
    """Our CR implementation effectively performs a Block Cholesky decompositon of J:
    LL^{\top} = J.
    The determinant in this case is -> det =  \prod diag(D_i)^2
    The log determinant -> logdet =  \sum 2* log (diag(D_i))
    """
    ms, Ds, Fs, Gs = decomp
    # concat together all the diagonal elements from all the diagonal blocks
    diags = torch.cat([torch.cat([torch.diag(x) for x in y]) for y in Ds])
    return 2 * torch.sum(torch.log(diags))


def mahal(decomp, y):
    """
    get y^T J^-1 y = y^T (LL^T)^-1 y = (y^T L^-T)(L^-1 y) =  ||(L^-1 y)||^2
    """

    v = halfsolve(decomp, y)
    return torch.sum(torch.cat(v, dim=0) ** 2)


def inverse_blocks(decomp):
    """
    returns diagonal and lower off-diagonal blocks of J^-1
    """
    ms, Ds, Fs, Gs = decomp
    Sig_diag = torch.linalg.inv(Ds[-1])
    Sig_diag = torch.matmul(torch.transpose(Sig_diag, 1, 2), Sig_diag)
    Sig_off = torch.zeros((0,) + Sig_diag.shape[1:], dtype=Sig_diag.dtype)
    for i in range(1, len(Ds)):
        D = Ds[-i - 1]
        F = Fs[-i]
        G = Gs[-i]

        # invert D
        Di = torch.linalg.inv(D)
        DtiDi = torch.matmul(torch.transpose(Di, 1, 2), Di)

        # compute U D^-1
        # FDi and GDi are the diagional and lower diational blocks of U D^-1 respectively
        FDi = torch.matmul(F, Di[: len(F)])  # batch matrix multiplication
        GDi = torch.matmul(G, Di[1:])

        # compute the diagonal and upper-diagonal parts of Sig UD^-1
        SUDi_diag, SUDi_off = SigU(-Sig_diag, -Sig_off, FDi, GDi)

        # compute the diagonal parts of D^-T U^T Sig UD^-1
        UtSUDi_diag = -UtV_diags(FDi, GDi, SUDi_diag, SUDi_off) + DtiDi

        Sig_diag, Sig_off = (
            interleave(UtSUDi_diag, Sig_diag),
            interleave(SUDi_diag, torch.transpose(SUDi_off, 1, 2)),
        )

    return Sig_diag, Sig_off
