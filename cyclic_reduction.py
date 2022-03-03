import torch
from torchtyping import TensorType, patch_typeguard
import inspect
import numpy as np


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
        temp = leaf1[:-1]+leaf2
        tq = torch.cat([leaf1[:-1]+leaf2,leaf1[-1].unsqueeze(0)],dim=0) #double check this unsqueeze
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
        return torch.einsum('ijk,ik...->ij',diags,x[:-1]) + torch.einsum('ijk,ik...->ij',offdiags,x[1:])
    else:
        leaf1 = torch.einsum('ijk,ik...->ij',diags,x)
        leaf2 = torch.einsum('ijk,ik...->ij',offdiags,x[1:])
        leaf_sum = leaf1[:-1] + leaf2
        #remembering the final row of the matrix which just has a diagional element and not an upper diagional element
        return torch.cat([leaf_sum, leaf1[-1].unsqueeze(0)], dim=0)

def U_Tx(diags,offdiags,x):
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
        leaf1= torch.einsum('ikj,ik...->ij',diags,x)
        leaf2= torch.einsum('ikj,ik...->ij',offdiags,x)
        return torch.cat([leaf1[0].unsqueeze(0),leaf1[1:]+leaf2[:-1],leaf2[-1].unsqueeze(0)], dim=0)
    else:
        leaf1= torch.einsum('ikj,ik...->ij',diags,x) #double check ellipsis
        leaf2= torch.einsum('ikj,ik...->ij',offdiags,x[:-1])
        return torch.cat([leaf1[0].unsqueeze(0),leaf1[1:]+leaf2], dim=0) #unsqueezing again

def SigU(sig_dblocks,sig_offdblocks,u_dblocks,u_offdblocks):
    '''
    Let Sig be a symmetric block-tridiagonal matrix whose
    - diagonal blocks are sig_dblocks
    - lower off-diagonals are sig_offdblocks
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by u_dblocks
    - upper off-diagonals are given by u_offdblocks
    We would like to compute block-tridiagonal blocks of Sig @ U
    '''
    
    #square matrix
    if u_dblocks.shape[0]==u_offdblocks.shape[0]+1:
        main_diagional = torch.cat([
            (sig_dblocks[0]@u_dblocks[0]).unsqueeze(0), 
            torch.matmul(sig_dblocks[1:],u_dblocks[1:]) + torch.matmul(sig_offdblocks,u_offdblocks)
        ], axis=0)
        
        #a = torch.matmul(sig_dblocks[:-1],sig_offdblocks)
        # print(sig_offdblocks.shape)
        # print(u_dblocks[1:].shape, u_dblocks[1:].T.shape)
        # b = torch.matmul(torch.transpose(sig_offdblocks, 1, 2), u_dblocks[1:])
        # print("offdiags")
        # print(sig_offdblocks)
        upper_diagional = torch.matmul(sig_dblocks[:-1], u_offdblocks) + torch.matmul(torch.transpose(sig_offdblocks, 1, 2), u_dblocks[1:]) #make sure transposing right dimensions
        # print("upline")
        # print(upper_diagional)
    #non-square matrix
    else:
        main_diagional = torch.cat([
            (sig_dblocks[0]@u_dblocks[0]).unsqueeze(0),
            torch.matmul(sig_dblocks[1:],u_dblocks[1:]) + torch.matmul(sig_offdblocks,u_offdblocks[:-1])
        ], axis=0)
        
        upper_diagional = torch.cat([
            torch.matmul(sig_dblocks[:-1],u_offdblocks[:-1]) + torch.matmul(torch.transpose(sig_offdblocks, 1, 2), u_dblocks[1:]),
            (sig_dblocks[-1]@u_offdblocks[-1]).unsqueeze(0)
        ], axis=0)

    return main_diagional, upper_diagional

def UtV_diags(u_dblocks: TensorType["num_blocks", "block_dim", "block_dim"], 
            u_offdblocks: TensorType["num_blocks", "block_dim", "block_dim"] ,
            v_dblocks: TensorType["num_blocks", "block_dim", "block_dim"],
            v_offdblocks: TensorType["num_blocks", "block_dim", "block_dim"]
    ):
    '''
    U is upper didiagonal with
    - diagional blocks are u_dblocks
    - upper off-diagionals are u_offdblocks
    V is upper didiagonal with
    - diagional blocks are v_dblocks
    - upper off-diationals are v_offdblocks
    We want the diagonal blocks of U.T @ V
    '''

    #square matrix
    if u_dblocks.shape[0]==u_offdblocks.shape[0]+1:
        #print(u_dblocks[0].shape)
        return torch.cat([
            (u_dblocks[0].T@v_dblocks[0]).unsqueeze(0),
            torch.matmul(torch.transpose(u_dblocks[1:], 1, 2),v_dblocks[1:]) + torch.matmul(torch.transpose(u_offdblocks, 1, 2),v_offdblocks)
        ],axis=0)

    #non-square matrix
    else:
        return torch.concat([
            (u_dblocks[0].T@v_dblocks[0]).unsqueeze(0),
            torch.matmul(torch.transpose(u_dblocks[1:], 1, 2) ,v_dblocks[1:]) + torch.matmul(torch.transpose(u_offdblocks[:-1], 1, 2),v_offdblocks[:-1]),
            (u_offdblocks[-1].T@v_offdblocks[-1]).unsqueeze(0)
        ],axis=0)

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


def decompose_loop(Rs, Os):
    num_blocks = Rs.shape[0]
    Rs_even = Rs[::2]
    Ks_even = torch.linalg.cholesky(Rs_even)

    Os_even = Os[::2]
    Os_odd = Os[1::2]

    Os_even_T = Os_even.transpose(1, 2)

    #Os could be different block size than Ks
    N2 = Os_even.shape[0]

    F = torch.transpose(torch.triangular_solve(input=Os_even_T, A=Ks_even[:N2], upper=False)[0], 1, 2)
    #Why is is Os_odd not transposed?
    G = torch.transpose(torch.triangular_solve(input=Os_odd, A=Ks_even[1::][:N2], upper=False)[0], 1, 2)

    UU_T_diags, UU_T_offdiags = UU_T(F, G)

    #Constructing J~
    Rs = Rs[1::2] - UU_T_diags
    Os = -UU_T_offdiags
    
    return (num_blocks, Ks_even, F, G), (Rs, Os)

#J is a symmetric block tridiagional matrix with R blocks on the main diagional, and O blocks below it
#diagonal block matrices must be invertible

def decompose(Rs: TensorType["num_blocks", "block_dim", "block_dim"], Os: TensorType["num_blocks", "block_dim", "block_dim"]):
    num_blocks = Rs.shape[0]
    block_dim = Rs.shape[1]
    Ds = []
    Fs = []
    Gs = []
    ms = []
    while len(Rs) > 1:
        (num_blocks, Ks_even, F, G), (Rs, Os) = decompose_loop(Rs, Os)
        ms += [num_blocks]
        Ds += [Ks_even]
        Fs += [F]
        Gs += [G]
        # Rs_even = Rs[::2]
        # Ks_even = torch.linalg.cholesky(Rs_even)

        # Os_even = Os[::2]
        # Os_odd = Os[1::2]

        # Os_even_T = Os_even.transpose(1, 2)

        # #Os could be different block size than Ks
        # N2 = Os_even.shape[0]

        # F = torch.transpose(torch.triangular_solve(input=Os_even_T, A=Ks_even[:N2], upper=False)[0], 1, 2)
        # #Why is is Os_odd not transposed?
        # G = torch.transpose(torch.triangular_solve(input=Os_odd, A=Ks_even[1::][:N2], upper=False)[0], 1, 2)

        # UU_T_diags, UU_T_offdiags = UU_T(F, G)

        # #Constructing J~
        # Rs = Rs[1::2] - UU_T_diags
        # Os = -UU_T_offdiags
        
        # num_blocks = Rs.shape[0]

        

    Ds += [torch.linalg.cholesky(Rs)]
    ms += [1]
    
    return torch.tensor(ms), Ds, Fs, Gs

def halfsolve(decomp, y: TensorType["num_blocks", "block_dim"]):
    ms, Ds, Fs, Gs = decomp

    #ytilde is our effective y for each L~ as we recurr
    ytilde=y
    xs=[]
    for i in range(ms.shape[0]):
        #effect of permutation matrix P_m (takes even b)
        y = ytilde[::2]
        #taking the ith matrix of Ds from cyclic reduction
        xs.append(torch.triangular_solve(input = y.unsqueeze(-1), A = Ds[i], upper=False)[0][...,0])

        if ytilde.shape[0] > 1:
            #From the equation (L~)x_2 = (Q_m)b - U(x_1)
            ytilde = ytilde[1::2] - Ux(Fs[i], Gs[i], xs[-1])
        else:
            break

    return xs

def backhalfsolve(decomp, ycrr):
    '''
    Input:
    - decomp, the cyclic reduction representation of a block tridiagonal matrix
      with nchain diagonal blocks, each of size nblock
    - ycrr, the cyclic reduction representation of a vector y:
    returns the value of T_n^T L^-T y //My Question: What does T_n mean here?
    '''
    ms, Ds, Fs, Gs = decomp

    #bottom up recursion: start with the base case, corresponds to the final elements of D and b
    ytilde = ycrr[-1]
    #solving L~^T(Q_m)x = b_2
    #base case is L~ = D[-1], and this corresponds to the final D block, and it's lower triangular b/c its just a cholesky decomp
    xs=torch.triangular_solve(input = ytilde.unsqueeze(-1), A = Ds[-1], transpose=True, upper=False)[0][...,0]#double check tf.linalg.triangular_solve documentation for adjoint

    #up the tower of blocks
    for i in range(1, ms.shape[0] + 1):
        #makes sure we haven't reached the top of the D-blocks
        if len(ycrr)-i-1>=0:
            #next two lines correspond to equation D^T(P_m)x = b_1 - U^T(Q_m)x 
            ytilde = ycrr[-i-1] - U_Tx(Fs[-i], Gs[-i], xs)
            x_even = torch.triangular_solve(input = ytilde.unsqueeze(-1), A = Ds[-i-1], transpose=True, upper=False)[0][...,0]

            #Note that xs corresponds to (Q_m)x, which is x_odd for all solutions of x below this point in the recurrsion, 
            #so in order to get the right x_vector for this point in the recurrsion we need to interleave the even and odd solutions
            xs = interleave(x_even,xs) #unsqueeze(0)
        else:
            break

    return xs

def mahal_and_det(Rs,Os,x):
    '''
    Let J denote a symmetric positive-definite block-tridiagonal matrix whose
    - diagonal blocks are given by Rs
    - lower off-diagonal blocks are given by Os
    returns
    - mahal: x J^-1 x
    - det: |J|
    We here obtain a cyclic reduction representation of J which can be used
    for further analysis.
    '''
    ms=[]
    Ds=[]
    Fs=[]
    Gs=[]

    det=0
    mahal=0

    ytilde=x

    while Rs.shape[0]>1:
        # do the work
        (numblocks,Ks_even,F,G),(Rs,Os) = decompose_loop(Rs,Os)

        # det
        #print(Ks_even.shape)
        #t = torch.diag(Ks_even)
        det += torch.sum(torch.log(torch.diagonal(Ks_even, dim1=1, dim2=2)))

        # process the even entries
        y=ytilde[::2]
        newx = torch.triangular_solve(input=Ks_even, A=y.unsqueeze(-1), upper=False)[0][...,0]
        mahal+= torch.sum(newx**2)

        # recurse on the odd entries
        ytilde = ytilde[1::2] - Ux(F,G,newx)

    D1=torch.linalg.cholesky(Rs)
    det += torch.sum(torch.log(torch.diagonal(Ks_even, dim1=1, dim2=2)))
    y=ytilde[::2]
    newx = torch.triangular_solve(input=Ks_even, A=y.unsqueeze(-1), upper=False)[0][...,0]
    mahal+= torch.sum(newx**2)

    return mahal, 2*det



def solve(decomp, y: TensorType["num_blocks", "block_dim"]):
    v = halfsolve(decomp,y)
    w = backhalfsolve(decomp,v)
    return w

def det(decomp):
    #computes log |J| given L which is a cyclic reduction decomposition of J
    ms,Ds,Fs,Gs = decomp
    diags = torch.cat([torch.cat([torch.diag(x) for x in y]) for y in Ds])
    return 2 * torch.sum(torch.log(diags))

def mahal(decomp,y):
    '''
    get y^T J^-1 y
    '''

    v=halfsolve(decomp,y)
    return torch.sum(torch.cat(v,dim=0)**2)
    
def inverse_blocks(decomp):
    '''
    returns diagonal and lower off-diagonal blocks of J^-1
    '''
    ms,Ds,Fs,Gs = decomp
    # print("First D:")
    # print(Ds[-1])
    Sig_diag=torch.linalg.inv(Ds[-1])
    # print(np.linalg.inv(Ds[-1].numpy()))
    Sig_diag=torch.matmul(torch.transpose(Sig_diag, 1, 2), Sig_diag)
    #print(Sig_diag)
    Sig_off = torch.zeros((0,)+Sig_diag.shape[1:],dtype=Sig_diag.dtype)
    for i in range(1,len(Ds)):
        #print(i)
        D=Ds[-i-1]; F=Fs[-i]; G=Gs[-i]
        
        # invert D
        Di=torch.linalg.inv(D)
        DtiDi=torch.matmul(torch.transpose(Di, 1, 2), Di)

        # compute U D^-1
        #FDi and GDi are the diagional and lower diational blocks of U D^-1 respectively
        FDi = torch.matmul(F,Di[:len(F)]) #batch matrix multiplication
        GDi = torch.matmul(G,Di[1:])

        # compute the diagonal and upper-diagonal parts of Sig UD^-1
        SUDi_diag,SUDi_off = SigU(-Sig_diag,-Sig_off,FDi,GDi)
        # print("SUDi_diag:")
        # print(SUDi_diag)
        # print("SUDi_off")
        # print(SUDi_off)

        # compute the diagonal parts of D^-T U^T Sig UD^-1
        UtSUDi_diag = -UtV_diags(FDi,GDi,SUDi_diag,SUDi_off) + DtiDi
        # print("UtSUDi_diag:")
        # print(UtSUDi_diag)

        Sig_diag, Sig_off = interleave(UtSUDi_diag,Sig_diag), interleave(SUDi_diag,torch.transpose(SUDi_off, 1, 2))
        # print(Sig_diag)
    return Sig_diag, Sig_off


###############################################################
############################Tests##############################
###############################################################

def eo(n):
    import numpy as np
    guys1=np.r_[0:n:2]
    guys2=np.r_[1:n:2]
    return np.concatenate([guys1,guys2])

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

def make_U(diags,offdiags):
    '''
    Let U be an upper bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags
    We return U
    '''
    import numpy as np


    n=diags.shape[0]
    m=offdiags.shape[0]
    k=diags.shape[1]

    if n==m:
        V=np.zeros((n,k,n+1,k))
        for i in range(n):
            V[i,:,i]=diags[i]
        for i in range(n):
            V[i,:,i+1]=offdiags[i]
        return V.reshape((n*k,(n+1)*k))
    else:
        V=np.zeros((n,k,n,k))
        for i in range(n):
            V[i,:,i]=diags[i]
        for i in range(n-1):
            V[i,:,i+1]=offdiags[i]
        return V.reshape((n*k,n*k))

def _test_U_stuff(nblock,n,even):
    import numpy as np
    import numpy.random as npr

    def fl(A):
        return A.reshape((A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]))
    def unfl(A):
        return A.reshape((A.shape[0]//nblock,nblock,A.shape[1]//nblock,nblock))

    A=npr.randn(n,nblock,nblock)
    if even:
        B=npr.randn(n,nblock,nblock)
        x=npr.randn(n+1,nblock)
    else:
        B=npr.randn(n-1,nblock,nblock)
        x=npr.randn(n,nblock)
    y=npr.randn(n,nblock)
    U=make_U(A,B)

    Sig=npr.randn(n*nblock,n*nblock)
    Sig=unfl(Sig@Sig.T)
    dblocks=np.array([Sig[i,:,i] for i in range(len(Sig))])
    offblocks=np.array([Sig[i+1,:,i] for i in range(len(Sig)-1)])

    # is UUt is what it says it is?
    UUt_d,UUt_o=UU_T(torch.from_numpy(A),torch.from_numpy(B))
    UUt_full=unfl(U@U.T)
    for i in range(n):
        assert np.allclose(UUt_d[i],UUt_full[i,:,i])
    for i in range(n-1):
        assert np.allclose(UUt_o[i],UUt_full[i+1,:,i])

    # is Ux right?
    assert np.allclose(U@x.ravel(),Ux(torch.from_numpy(A),torch.from_numpy(B),torch.from_numpy(x)).numpy().ravel())

    # is UTx right?
    assert np.allclose(U.T@y.ravel(),U_Tx(torch.from_numpy(A),torch.from_numpy(B),torch.from_numpy(y)).numpy().ravel())

    # is tridi right?
    SigU_full = unfl(fl(Sig)@U)
    SigU_mid,SigU_hi = SigU(torch.from_numpy(dblocks),torch.from_numpy(offblocks),torch.from_numpy(A),torch.from_numpy(B))
    assert np.allclose(
        SigU_mid,
        np.array([SigU_full[i,:,i] for i in range(SigU_full.shape[0])])
    )
    assert np.allclose(
        SigU_hi,
        np.array([SigU_full[i,:,i+1] for i in range(SigU_hi.shape[0])])
    )

    # finally, we need to look at U^T SigU
    UTSigU = unfl(U.T @ fl(SigU_full))
    centrals=np.array([UTSigU[i,:,i] for i in range(UTSigU.shape[0])])
    centrals_guess = UtV_diags(torch.from_numpy(A),torch.from_numpy(B),SigU_mid,SigU_hi)
    assert np.allclose(centrals.ravel(),centrals_guess.numpy().ravel())


def test_U_stuff():
    _test_U_stuff(1,4,True)
    _test_U_stuff(1,4,False)
    _test_U_stuff(2,3,True)
    _test_U_stuff(2,3,False)


def test():
    import numpy as np
    import numpy.random as npr
    npr.seed(10)

    for nblock in [1,3]:
        print(nblock)
        for nchain in [2,6,30,31,32,33]: #2
            print(nchain)
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

            #check mahalanobis and halfsolve
            v=npr.randn(nchain,nblock)

            mahal1=np.sum(v.ravel()*np.linalg.solve(J.reshape(sh2),v.ravel()))
            mahal2 = mahal(decomp, torch.from_numpy(v))
            #mahal2=np.mean(np.concatenate(halfsolve(decomp,torch.from_numpy(v)))**2)
            assert np.allclose(mahal1,mahal2)
            
            assert np.allclose(np.linalg.solve(L,Tm@v.ravel()),np.concatenate(halfsolve(decomp,torch.from_numpy(v))).ravel())

            # check determinant
            #diags= np.concatenate([[np.diag(x) for x in y] for y in Ds])
            #det1=2*np.sum(np.log(diags))
            det1 = det(decomp)
            det2 = np.linalg.slogdet(J.reshape(sh2))[1]
            assert np.allclose(det1,det2)

            #check mahal and det
            mahal3, det3 = mahal_and_det(torch.from_numpy(Rs),torch.from_numpy(Os),torch.from_numpy(v))
            print(mahal1, mahal2, mahal3)
            #assert np.allclose(mahal1,mahal3)
            print(det1, det2, det3)
            assert np.allclose(det1,det3)


            # check backhalfsolve
            vrep=[torch.tensor(npr.randn(x,nblock)) for x in (np.array(ms)+1)//2]
            v=np.concatenate(vrep)
            rez=np.linalg.solve(L.T@Tm,v.ravel())
            assert np.allclose(backhalfsolve(decomp,vrep).numpy().ravel(),rez)

            # check Sig
            Sig=np.linalg.inv(J.reshape(sh2)).reshape(sh1)
            Sig_diag=np.array([Sig[i,:,i] for i in range(J.shape[0])])
            Sig_off=np.array([Sig[i+1,:,i] for i in range(J.shape[0]-1)])
            guess_diag,guess_off = inverse_blocks(decomp)
            a = Sig_diag.ravel()
            b = guess_diag.numpy().ravel()
            assert np.allclose(guess_diag.numpy().ravel(),Sig_diag.ravel())
            assert np.allclose(guess_off.numpy().ravel(),Sig_off.ravel())

            


def more_tests():
    print("---------------")
    Rs = torch.tensor([2.0, 2.0, 2.0]).unsqueeze(-1).unsqueeze(-1)
    Os = torch.tensor([-1.0, -1.0]).unsqueeze(-1).unsqueeze(-1)
    b = torch.tensor([1.0, 1.0, 1.0]).unsqueeze(-1)
    decomp = decompose(Rs, Os)
    print(solve(decomp, b))
    #print()


if __name__ == '__main__':
    test()
    more_tests()
    test_U_stuff()










    
