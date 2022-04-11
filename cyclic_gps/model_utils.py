import torch

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
    n = joint_mean.shape[-1]
    m = marginal_mean.shape[-1]
    slx=slice(0,m)
    sly=slice(m,None)
    
    #mean_transformer = Cov[p(x,y)] @ (Cov[p(x)])^-1 aka  + C^TA^{-1}
    
    mean_transformer = joint_cov[...,sly,slx] @  torch.linalg.inv(joint_cov[...,slx,slx])

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
    


