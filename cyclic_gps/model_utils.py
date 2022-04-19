import torch
import math

#utility function external to the LEG model class
def compute_G(N, R):
  """describe G here. it's an important quantity of LEG, needed for cov/precision"""
  G = N @ N.T + R - R.T
  # add small diag noise
  G += torch.eye(N.shape[0], dtype=N.dtype) * (1e-5)
  return G

def compute_eG(G_val, G_vec, G_vec_inv, diffs):
        '''
        Input:
        - G_val:  n
        - G_vec:  n x n
        - G_vec_inv: n x n  (=Gvec^-1)
        - diffs:     m
        Output:
        - exp(-.5*d*Gvec diag(Gval) Gveci),  m x n x n
        Note that this function doesn't take care of transposing if diff is positive
        '''
        diffs = diffs.unsqueeze(-1).unsqueeze(-1)
        G_val = G_val.unsqueeze(0).unsqueeze(0)
        G_vec = G_vec.unsqueeze(0)
        gd = G_vec * torch.exp(-.5 * diffs * G_val) #m x n x n
        gdg_inv = gd @ G_vec_inv.unsqueeze(0)

        return torch.real(gdg_inv)

def build_2x2_block(a, b, c, d):
    '''
    Input:
    a: ... x n1 x n2
    b: ... x n1 x n3
    c: ... x n4 x n2
    d: ... x n4 x n3
    Output:
    [
        [a b],
        [c,d]
    ]
    suitably batched
    '''

    ab = torch.cat([a,b],axis=-1)
    cd = torch.cat([c,d],axis=-1)

    abcd = torch.cat([ab,cd],axis=-2)

    return abcd

def build_3x3_block(a,b,c,d,e,f,g,h,i):
    abc = torch.cat([a,b,c],axis=-1)
    cde = torch.cat([d,e,f],axis=-1)
    fgh = torch.cat([g,h,i],axis=-1)

    rez = torch.cat([abc,cde,fgh],axis=-2)

    return rez
    


def gaussian_stitch(joint_mean, joint_cov, marginal_mean, marginal_cov):
    '''
    Input:
    - joint_mean: ... x n
    - joint_cov:  ... x n x n
    - marginal_mean (insample posterior mean): ... x m
    - marginal_cov (insample posterior cov):  ... x m x m
    with n > m
    Output:
      E[q(y)], cov[q(y)]
    where
      q(x,y) = p(x)p(y|x)
      p(x,y)= N(mean1,cov1)
      p(x)= N(mean2,cov2)
      x in R^m
      y in R^(m-n)
    '''
    # print("joint_cov:")
    # print(joint_cov.shape)
    n = joint_cov.shape[-1]
    m = marginal_cov.shape[-1]
    slx=slice(0,m)
    sly=slice(m,None)
    
    #mean_transformer = Cov[p(x,y)] @ (Cov[p(x)])^-1 aka  + C^TA^{-1}
    
    # print("joint_cov_y_x: {}".format(joint_cov[...,sly,slx]))
    # print("joint_cov_x_x: {}".format(joint_cov[...,slx,slx]))
    mean_transformer = joint_cov[...,sly,slx] @  torch.linalg.inv(joint_cov[...,slx,slx])
    #print("mean_transformer: {}".format(mean_transformer))

    #E[q(y)] = mean_y + mean_transformer @ mean_x note it is very close to E[p(y|x)] = b + C^TA^{-1}(x - a)
    mean = joint_mean[...,sly] + (mean_transformer @ marginal_mean[...,None])
    #print(mean.shape)
    mean = mean[...,0]

    #Cov[p(y|x)]
    conditional_cov = joint_cov[...,sly,sly] - mean_transformer @ joint_cov[...,slx,sly]
    
    #Cov[q[y]]
    cov = conditional_cov + (mean_transformer @ marginal_cov @ mean_transformer.T)
    # print("covariance shape:")
    # print(cov.shape)
    return mean, cov


def compute_prior_covariance(ts, G):
    n = len(ts)
    l = G.shape[0]
    G_t = G.T
    G = G.unsqueeze(0)
    G_t = G_t.unsqueeze(0)
    C = torch.empty(size=(n * l, n * l), dtype=G.dtype)
    for i in range(n):
        for j in range(n):
            if i > j:
                diff = ts[i] - ts[j]
                expd = torch.matrix_exp(-0.5 * G * diff)
            elif j > i:
                diff = ts[j] - ts[i]
                expd = torch.matrix_exp(-0.5 * G_t * diff)
            else:
                expd = torch.eye(n=l)
            C[i*l: i*l+l, j*l: j*l+l] = expd
    return C

    
def compute_log_marginal_likelihood(N, R, B, Lambda, ts, xs):
    n = len(xs)
    G = compute_G(N, R)
    Bs = [B for _ in range(n)]
    B_tilde = torch.block_diag(*Bs) #could maybe use expand
    Lambdas = [Lambda for _ in range(n)]
    Lambda_tilde = torch.block_diag(*Lambdas)
    Sigma = compute_prior_covariance(ts=ts, G=G)
    post_cov = B_tilde @ Sigma @ B_tilde.T + Lambda_tilde
    mahal = xs.reshape(1, -1) @ torch.linalg.inv(post_cov) @ xs.reshape(-1, 1)
    det = torch.log(torch.det(2*math.pi*post_cov))
    return (-.5 * mahal) + (-.5 * det)
    


