import torch
from cyclic_gps.cyclic_reduction import *
from known_matrices_full import bab_matrix, bab_determinant, bab_inverse, schur_block_matrix, schur_block_determinant, schur_block_inverse, \
                                neumann_matrix, neumann_determinant


def eo(n):
    import numpy as np

    guys1 = np.r_[0:n:2]
    guys2 = np.r_[1:n:2]
    return np.concatenate([guys1, guys2])


def recursive_eo(n):
    import numpy as np

    if n == 1:
        return np.array([0])
    elif n == 2:
        return np.array([0, 1])
    else:
        guys1 = np.r_[0:n:2]
        guys2 = np.r_[1:n:2]
        return np.concatenate([guys1, guys2[recursive_eo(len(guys2))]])


def perm2P(perm):
    import numpy as np

    n = len(perm)
    v = np.zeros((n, n))
    for i in range(n):
        v[i, perm[i]] = 1
    return v


def make_U(diags, offdiags):
    """
    Let U be an upper bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags
    We return U
    """
    import numpy as np

    n = diags.shape[0]
    m = offdiags.shape[0]
    k = diags.shape[1]

    if n == m:
        V = np.zeros((n, k, n + 1, k))
        for i in range(n):
            V[i, :, i] = diags[i]
        for i in range(n):
            V[i, :, i + 1] = offdiags[i]
        return V.reshape((n * k, (n + 1) * k))
    else:
        V = np.zeros((n, k, n, k))
        for i in range(n):
            V[i, :, i] = diags[i]
        for i in range(n - 1):
            V[i, :, i + 1] = offdiags[i]
        return V.reshape((n * k, n * k))


def _test_efficient_tridiag_matrix_mult(block_dim, num_dblocks, square):
    import numpy as np
    import numpy.random as npr

    def fl(A):
        return A.reshape((A.shape[0] * A.shape[1], A.shape[2] * A.shape[3]))

    def unfl(A):
        return A.reshape((A.shape[0] // block_dim, block_dim, A.shape[1] // block_dim, block_dim))

    A = npr.randn(num_dblocks, block_dim, block_dim)
    if square:
        B = npr.randn(num_dblocks, block_dim, block_dim)
        x = npr.randn(num_dblocks + 1, block_dim)
    else:
        B = npr.randn(num_dblocks - 1, block_dim, block_dim)
        x = npr.randn(num_dblocks, block_dim)
    y = npr.randn(num_dblocks, block_dim)
    U = make_U(A, B)

    Sig = npr.randn(num_dblocks * block_dim, num_dblocks * block_dim)
    Sig = unfl(Sig @ Sig.T)
    dblocks = np.array([Sig[i, :, i] for i in range(len(Sig))])
    offblocks = np.array([Sig[i + 1, :, i] for i in range(len(Sig) - 1)])

    # is UUt right?
    UUt_d, UUt_o = UU_T(torch.from_numpy(A), torch.from_numpy(B))
    UUt_full = unfl(U @ U.T)
    for i in range(num_dblocks):
        assert np.allclose(UUt_d[i], UUt_full[i, :, i])
    for i in range(num_dblocks - 1):
        assert np.allclose(UUt_o[i], UUt_full[i + 1, :, i])

    # is Ux right?
    assert np.allclose(
        U @ x.ravel(),
        Ux(torch.from_numpy(A), torch.from_numpy(B), torch.from_numpy(x))
        .numpy()
        .ravel(),
    )

    # is UTx right?
    assert np.allclose(
        U.T @ y.ravel(),
        U_Tx(torch.from_numpy(A), torch.from_numpy(B), torch.from_numpy(y))
        .numpy()
        .ravel(),
    )

    # is tridi right?
    SigU_full = unfl(fl(Sig) @ U)
    SigU_mid, SigU_hi = SigU(
        torch.from_numpy(dblocks),
        torch.from_numpy(offblocks),
        torch.from_numpy(A),
        torch.from_numpy(B),
    )
    assert np.allclose(
        SigU_mid, np.array([SigU_full[i, :, i] for i in range(SigU_full.shape[0])])
    )
    assert np.allclose(
        SigU_hi, np.array([SigU_full[i, :, i + 1] for i in range(SigU_hi.shape[0])])
    )

    # finally, we need to look at U^T SigU
    UTSigU = unfl(U.T @ fl(SigU_full))
    centrals = np.array([UTSigU[i, :, i] for i in range(UTSigU.shape[0])])
    centrals_guess = UtV_diags(
        torch.from_numpy(A), torch.from_numpy(B), SigU_mid, SigU_hi
    )
    assert np.allclose(centrals.ravel(), centrals_guess.numpy().ravel())


def test_efficient_tridiag_matrix_mult():
    _test_efficient_tridiag_matrix_mult(block_dim=1, num_dblocks=4, square=True)
    _test_efficient_tridiag_matrix_mult(block_dim=1, num_dblocks=4, square=False)
    _test_efficient_tridiag_matrix_mult(block_dim=2, num_dblocks=3, square=True)
    _test_efficient_tridiag_matrix_mult(block_dim=2, num_dblocks=3, square=False)


def test_random_tridiag_matrices():
    import numpy as np
    import numpy.random as npr

    #npr.seed(10)

    for block_dim in [1, 3]:
        print(block_dim)
        for num_dblocks in [2, 6, 30, 31, 32, 33]:  # 2
            print(num_dblocks)
            sh1 = (num_dblocks, block_dim, num_dblocks, block_dim)
            sh2 = (num_dblocks * block_dim, num_dblocks * block_dim)
            Ld = [npr.randn(block_dim, block_dim) for i in range(num_dblocks)]
            Lo = [npr.randn(block_dim, block_dim) for i in range(num_dblocks - 1)]
            L = np.zeros(sh1)
            for i in range(num_dblocks):
                L[i, :, i] = Ld[i] + np.eye(block_dim) * 3
            for i in range(1, num_dblocks):
                L[i, :, i - 1] = Lo[i - 1]
            L = L.reshape(sh2)
            J = L @ L.T
            J = J.reshape(sh1)

            # the slow analysis
            Tm = np.kron(perm2P(recursive_eo(num_dblocks)), np.eye(block_dim))
            L = np.linalg.cholesky(Tm @ J.reshape(sh2) @ Tm.T)

            # the fast analysis
            Rs = np.array([J[i, :, i] for i in range(num_dblocks)])
            Os = np.array([J[i, :, i - 1] for i in range(1, num_dblocks)])
            decomp = decompose(torch.from_numpy(Rs), torch.from_numpy(Os))
            ms, Ds, Fs, Gs = decomp

            # check mahalanobis and halfsolve
            v = npr.randn(num_dblocks, block_dim)

            mahal1 = np.sum(v.ravel() * np.linalg.solve(J.reshape(sh2), v.ravel()))
            mahal2 = mahal(decomp, torch.from_numpy(v))
            # mahal2=np.mean(np.concatenate(halfsolve(decomp,torch.from_numpy(v)))**2)
            assert np.allclose(mahal1, mahal2)

            assert np.allclose(
                np.linalg.solve(L, Tm @ v.ravel()),
                np.concatenate(halfsolve(decomp, torch.from_numpy(v))).ravel(),
            )

            # check determinant
            # diags= np.concatenate([[np.diag(x) for x in y] for y in Ds])
            # det1=2*np.sum(np.log(diags))
            det1 = np.linalg.slogdet(J.reshape(sh2))[1]
            det2 = det(decomp)

            assert np.allclose(det1, det2)

            # check mahal and det
            mahal3, det3 = mahal_and_det(
                torch.from_numpy(Rs), torch.from_numpy(Os), torch.from_numpy(v)
            )

            assert np.allclose(mahal1, mahal3)
            assert np.allclose(det1, det3)

            # check backhalfsolve
            vrep = [torch.tensor(npr.randn(x, block_dim)) for x in (np.array(ms) + 1) // 2]
            v = np.concatenate(vrep)
            rez = np.linalg.solve(L.T @ Tm, v.ravel())
            assert np.allclose(backhalfsolve(decomp, vrep).numpy().ravel(), rez)

            # check inverse
            Sig = np.linalg.inv(J.reshape(sh2)).reshape(sh1)
            Sig_diag = np.array([Sig[i, :, i] for i in range(J.shape[0])])
            Sig_off = np.array([Sig[i + 1, :, i] for i in range(J.shape[0] - 1)])
            guess_diag, guess_off = inverse_blocks(decomp)
            a = Sig_diag.ravel()
            b = guess_diag.numpy().ravel()
            assert np.allclose(guess_diag.numpy().ravel(), Sig_diag.ravel())
            assert np.allclose(guess_off.numpy().ravel(), Sig_off.ravel())


#M is our matrix which we want to compute the
#bs is our block size
def compute_block_diagonal_representation(M, block_size, square=True):
    assert(M.shape[0] % block_size == 0) #might want to add functionality for matrices where this is not true?
    num_dblocks = M.shape[0] // block_size
    Rs = torch.empty(size=(num_dblocks, block_size, block_size))
    Os = torch.empty(size=(num_dblocks-1, block_size, block_size))
    for i in range(num_dblocks):
        m_idx = i * block_size
        Rs[i] = M[m_idx:m_idx+block_size, m_idx:m_idx+block_size]
    for i in range(num_dblocks-1):
        m_idx = i * block_size
        Os[i] = M[m_idx+block_size:m_idx+2*block_size, m_idx:m_idx+block_size]
    return Rs, Os



def test_known_matrices():
    x = torch.rand(size=(10, 1))    

    BAB = torch.from_numpy(bab_matrix(n=10, alpha=5, beta=2))
    BAB_dblocks, BAB_offdblocks = compute_block_diagonal_representation(BAB, 1)
    BAB_decomp = decompose(BAB_dblocks, BAB_offdblocks)
    
    gt_det = np.log(bab_determinant(10, 5, 2)) 
    cr_det = det(BAB_decomp)
    cr_mahal, cr_det2 = mahal_and_det(BAB_dblocks, BAB_offdblocks, x=x)
    assert np.allclose(gt_det, cr_det)
    assert np.allclose(gt_det, cr_det2)

    gt_inverse = torch.from_numpy(bab_inverse(10, 5, 2)).float()
    gt_inverse_Rs, gt_inverse_Os = compute_block_diagonal_representation(gt_inverse, 1)

    print(gt_inverse.dtype, x.dtype)
    gt_mahal = x.T @ gt_inverse @ x

    cr_inverse = inverse_blocks(BAB_decomp)
    cr_inverse_Rs, cr_inverse_Os = cr_inverse
    assert np.allclose(gt_inverse_Rs, cr_inverse_Rs)
    assert np.allclose(gt_inverse_Os, cr_inverse_Os)

    assert np.allclose(gt_mahal, cr_mahal)

    schur = torch.from_numpy(schur_block_matrix(n=10, x=[1] * 10, y=[2] * 9))
    schur = schur.T @ schur #taking the gram matrix so it is positive semi-definite
    schur_dblocks, schur_offdblocks = compute_block_diagonal_representation(schur, 2)
    schur_decomp = decompose(schur_dblocks, schur_offdblocks)
    
    gt_det = np.log(schur_block_determinant(n=10, x=[1] * 10, y=[2] * 9)**2)
    cr_det = det(schur_decomp)
    cr_mahal, cr_det2 = mahal_and_det(schur_dblocks, schur_offdblocks, x=x.reshape(5, 2))
    assert np.allclose(gt_det, cr_det)
    assert np.allclose(gt_det, cr_det2)

    gt_inverse = torch.from_numpy(schur_block_inverse(n=10, x=[1] * 10, y=[2] * 9)).float()
    gt_inverse = gt_inverse @ gt_inverse.T

    gt_mahal = x.T @ gt_inverse @ x

    gt_inverse_Rs, gt_inverse_Os = compute_block_diagonal_representation(gt_inverse, 2)
    cr_inverse = inverse_blocks(schur_decomp)
    cr_inverse_Rs, cr_inverse_Os = cr_inverse
    assert np.allclose(gt_inverse_Rs, cr_inverse_Rs)
    assert np.allclose(gt_inverse_Os, cr_inverse_Os)

    assert np.allclose(gt_mahal, cr_mahal)

    #neumann = neumann_matrix






    




