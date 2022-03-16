from data_utils import threshold_timesteps



def fit_model_family(model, ts, xs):
	'''
    Fits a custom LEG model
    Input:
    - ts: list of timestamp-vectors: nsamp x [VARIABLE]
    - xs: list of observations:      nsamp x [VARIABLE] x n 
    '''

   	time_info = [threshold_timesteps(torch.tensor(ts_vec)) for ts_vec in ts]
   	xs = [torch.tensor(x) for x in xs]
    n=xs[0].shape[1]

    #total number of observations
    nobs = torch.sum(torch.tensor([torch.prod(x.shape) for x in xs]))
