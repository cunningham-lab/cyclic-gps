from data_utils import threshold_timesteps



#takes into account multiple observations per time point (Note that this is different than having multidimensional observations)
# def compute_loss(model, time_info, xs, nats, nobs):
# 	loss = 0
# 	#for each sample, take the tensor of observations, the thresholded time stamps of those observations, and the indexes of the time stamps
# 	for x, (sub_ts, sub_idxs) in zip(xs, time_info):
# 		loss += model.log_likelihood(sub_ts, x, sub_idxs)
# 	loss = -loss/nobs
# 	nats.append(loss)

# 	return loss





#NOT USED NOW
def fit_model_family(model, ts, xs, num_epochs):
	'''
    Fits a custom LEG model
    Input:
    - ts: list of timestamp-vectors: nsamp x [VARIABLE]
    - xs: list of observations:      nsamp x [VARIABLE] x n 
    '''

   	#time_info = [threshold_timesteps(torch.tensor(ts_vec)) for ts_vec in ts] #list of tuples
   	ts = [torch.tensor(t) for t in ts]
   	xs = [torch.tensor(x) for x in xs]
    n=xs[0].shape[1]

    #total number of observations
    nobs = torch.sum(torch.tensor([torch.prod(x.shape) for x in xs]))

    nats = []

    

    for epoch in num_epochs:

    	loss = compute_loss(model, time_info, xs, nats, nobs)
    	loss.backwards()



