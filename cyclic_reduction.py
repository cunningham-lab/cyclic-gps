import torch
from torchtyping import TensorType, patch_typeguard
import inspect


#def schur_complement():


def UU_T(diags, offdiags):
    n = diags.shape[0]
    m = offdiags.shape[0]

    #if n == m we have a non-square matrix
    if n == m:
        #matrix multiplication between each block and its transpose
        tq = torch.einsum('ijk,ilk->ijl',diags,diags)
        tq += torch.einsum('ijk,ilk->ijl',offdiags,offdiags)
        #matrix multiplication between offdiags and diag transpose
        offdiags = torch.einsum('ijk,ilk->ilj',offdiags[:-1],diags[1:])
        return tq,offdiags

    else:
        #matrix multiplication between each block and its transpose
        leaf1 = torch.einsum('ijk,ilk->ijl',diags,diags)
        leaf2 = torch.einsum('ijk,ilk->ijl',offdiags,offdiags)
        #adding the "squares" on the diagional offset by one
        tq = torch.cat([leaf1[:-1]+leaf2,[leaf1[-1]]],dim=0)
        #matrix multiplication between offdiags and diag transpose
        offdiags = torch.einsum('ijk,ilk->ilj',offdiags,diags[1:])
        return tq,offdiags

def Ux(diags, offdiags, x):
    '''
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags
    We would like to compute U@x
    '''
    n=diags.shape[0]
    m=offdiags.shape[0]

    #non-square matrix
    if n == m:
        return torch.einsum('ijk,ik->ij',diags,x[:-1]) + torch.einsum('ijk,ik->ij',offdiags,x[1:])
    else:
        leaf1 = torch.einsum('ijk,ik->ij',diags,x)
        leaf2 = torch.einsum('ijk,ik->ij',offdiags,x[1:])
        leaf_sum = leaf1[:-1] + leaf2
        #remembering the final row of the matrix which just has a diagional element and not an upper diagional element
        return torch.cat([leaf_sum, leaf1[-1]], dim=0)

def Utx(diags,offdiags,x):
    '''
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags
    We would like to compute U.T@x
    '''

    n=diags.shape[0]
    m=offdiags.shape[0]
    
    #non-square matrix
    if n==m:
        leaf1= torch.einsum('ikj,ik->ij',diags,x)
        leaf2= torch.einsum('ikj,ik->ij',offdiags,x)
        return torch.cat([[leaf1[0]],leaf1[1:]+leaf2[:-1],[leaf2[-1]]], dim=0)
    else:
        leaf1= torch.einsum('ikj,ik->ij',diags,x)
        leaf2= torch.einsum('ikj,ik->ij',offdiags,x[:-1])
        return torch.cat([[leaf1[0]],leaf1[1:]+leaf2], dim=0)

def interleave(a, b):
    '''
    V=np.zeros((a.shape[0]+b.shape[0],)+a.shape[1:])
    V[::2]=a
    V[1::2]=b
    '''
    a_shape=a.shape
    b_shape=b.shape
    n=a_shape[0]
    m=b_shape[0]
    if n<m:
        shp = (n*2,)+a_shape[1:]
        first_part = torch.reshape(torch.stack([a,b[:n]], dim=1), shape=shp)
        last_bit = b[n:]
        return torch.cat([first_part,last_bit], dim=0)
    else:
        shp = (m*2,)+b_shape[1:]
        first_part = torch.reshape(torch.stack([a[:m],b], dim=1), shape=shp)
        last_bit = a[m:]
        return torch.cat([first_part,last_bit], dim=0)


#J is a symmetric block tridiagional matrix with R blocks on the main diagional, and O blocks below it
#diagonal block matrices must be invertible
def decompose(Rs: TensorType["num_blocks", "block_dim", "block_dim"], Os: TensorType["num_blocks", "block_dim", "block_dim"]):
    num_blocks = Rs.shape[0]
    block_dim = Rs.shape[1]
    Ds = []
    Fs = []
    Gs = []
    ms = []

    while num_blocks > 1:
        ms += [num_blocks]
        Rs_even = Rs[::2]
        Ks_even = torch.cholesky(Rs_even)

        Os_even = Os[::2]
        Os_odd = Os[1::2]

        Os_even_T = Os_even.transpose(1, 2)

        #Os could be different block size than Ks
        N2 = Os_even.shape[0]
        #NOTE: HAVING TROUBLE WITH torch.trangular_solve function, DON'T TRY TO RUN THE CODE YET, DEBUGGING STILL IN PROGRESS
        F = torch.transpose(torch.triangular_solve(b=Os_even_T, A=Ks_even[:N2], upper=False), 1, 2)
        #Why is is Os_odd not transposed?
        G = torch.transpose(torch.triangular_solve(b=Os_odd, A=Ks_even[1::][:N2], upper=False), 1, 2)

        UU_T_diags, UU_T_offdiags = UU_T(F, G)

        #Constructing J~
        Rs = Rs[1::2] - UU_T_diags
        Os = -UU_T_offdiags
        
        num_blocks = Rs.shape[0]

        Ds += [Ks_even]
        Fs += [F]
        Gs += [G]

    Ds += [torch.cholesky(Rs)]
    ms += [1]

    return ms, Ds, Fs, Gs

def halfsolve(decomp, y):
    ms, Ds, Fs, Gs = decomp

    #ytilde is our effective y for each L~ as we recurr
    ytilde=y
    xs=[]
    for i in range(ms.shape[0]):
        #effect of permutation matrix P_m (takes even b)
        y = ytilde[::2]
        #taking the ith matrix of Ds from cyclic reduction
        xs.append(torch.triangular_solve(b = y.unsqueeze(-1), A = Ds[i], upper=False)[...,0])

        if ytilde.shape[0] > 1:
            #From the equation (L~)x_2 = (Q_m)b - U(x_1)
            ytilde = ytilde[1::2] - Ux(Fs[i], Gs[i], xs[-1])
        else:
            break

    return xs

def backhalfsolve(decomp, ycurr):
    '''
    Input:
    - decomp, the cyclic reduction representation of a block tridiagonal matrix
      with nchain diagonal blocks, each of size nblock
    - ycrr, the cyclic reduction representation of a vector y:
    returns the value of T_n^T L^-T y //My Question: What does T_n mean here?
    '''
    ms, Ds, Fs, Gs = decomp

    #bottom up recursion: start with the base case, corresponds to the final elements of D and b
    ytilde = ycurr[-1]
    #solving L~^T(Q_m)x = b_2
    #base case is L~ = D[-1], and this corresponds to the final D block, and it's lower triangular b/c its just a cholesky decomp
    xs=torch.triangular_solve(b = ytilde, A = Ds[-1], transpose=True, upper=False) #double check tf.linalg.triangular_solve documentation for adjoint

    #up the tower of blocks
    for i in range(1, ms.shape[0] + 1):
        #makes sure we haven't reached the top of the D-blocks
        if len(ycrr)-i-1>=0:
            #next two lines correspond to equation D^T(P_m)x = b_1 - U^T(Q_m)x 
            ytilde = ycurr[-i-1] - U_Tx(Fs[-i], Gs[-i], xs)
            x_even = torch.triangular_solve(b = ytilde.unsqueeze(-1), A = Ds[-i-1], transpose=True, upper=False)[...,0]

            #Note that xs corresponds to (Q_m)x, which is x_odd for all solutions of x below this point in the recurrsion, 
            #so in order to get the right x_vector for this point in the recurrsion we need to interleave the even and odd solutions
            xs = interleave(x_even,xs)
        else:
            break

    return xs


def solve(decomp, y):
    v = halfsolve(decomp,y)
    w = backhalfsolve(decomp,v)
    return w


###############################################################
############################Tests##############################
###############################################################

def recursive_eo(n):
    import numpy as np

    if n==1:
        return np.array([0])
    elif n==2:
        return np.array([0,1])
    else:
        guys1=np.r_[0:n:2]
        guys2=np.r_[1:n:2]
        return np.concatenate([guys1,guys2[recursive_eo(len(guys2))]])

def perm2P(perm):
    import numpy as np

    n=len(perm)
    v=np.zeros((n,n))
    for i in range(n):
        v[i,perm[i]]=1
    return v


def test():
    import numpy as np
    import numpy.random as npr

    for nblock in [1,3]:
        for nchain in [2,6,30,31,32,33]:
            sh1=(nchain,nblock,nchain,nblock)
            sh2=(nchain*nblock,nchain*nblock)
            Ld=[npr.randn(nblock,nblock) for i in range(nchain)]
            Lo=[npr.randn(nblock,nblock) for i in range(nchain-1)]
            L=np.zeros(sh1)
            for i in range(nchain):
                L[i,:,i]=Ld[i]+np.eye(nblock)*3
            for i in range(1,nchain):
                L[i,:,i-1]=Lo[i-1]
            L=L.reshape(sh2); J=L@L.T; J=J.reshape(sh1)

            # the slow analysis
            Tm=np.kron(perm2P(recursive_eo(nchain)),np.eye(nblock))
            L=np.linalg.cholesky(Tm@J.reshape(sh2)@Tm.T)


            # the fast analysis
            Rs=np.array([J[i,:,i] for i in range(nchain)])
            Os=np.array([J[i,:,i-1] for i in range(1,nchain)])
            decomp=decompose(torch.from_numpy(Rs),torch.from_numpy(Os))
            ms,Ds,Fs,Gs=decomp

            # check mahalanobis and halfsolve
            v=npr.randn(nchain,nblock)
            mahal=np.mean(v.ravel()*np.linalg.solve(J.reshape(sh2),v.ravel()))
            mahal2=np.mean(np.concatenate(halfsolve(decomp,torch.from_numpy(v)))**2)
            assert np.allclose(mahal,mahal2)
            assert np.allclose(np.linalg.solve(L,Tm@v.ravel()),np.concatenate(halfsolve(decomp,torch.from_numpy(v))).ravel())

            # check backhalfsolve
            vrep=[npr.randn(x,nblock) for x in (np.array(ms)+1)//2]
            v=np.concatenate(vrep)
            rez=np.linalg.solve(L.T@Tm,v.ravel())
            assert np.allclose(backhalfsolve(decomp,torch.from_numpy(vrep)).numpy().ravel(),rez)

if __name__ == '__main__':
    test()










    
