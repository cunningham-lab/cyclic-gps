import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_predictions(
    observation_ts: np.ndarray, 
    observation_xs: np.ndarray, 
    test_ts: List[np.ndarray], 
    pred_means: List[np.ndarray], 
    pred_variances: List[np.ndarray] = None, 
    lower_confidence: List[np.ndarray] = None, 
    upper_confidence: List[np.ndarray] = None,
    labels: str = None
):
    assert not(pred_variances is None and (lower_confidence is None or upper_confidence is None)) 
    print("observations ts and xs shape: {}, {}".format(observation_ts.shape, observation_xs.shape))
    plt.plot(observation_ts, observation_xs[:, 0], label='Observations')

    #TODO: add if statement to below assert to account for pred_variances = none 
    #assert(len(test_ts) == len(pred_means) and len(pred_means) == len(pred_variances))
   
    for i in range(len(test_ts)):
        if labels:
            plt.plot(test_ts[i], pred_means[i][:,0], "C{}".format(i + 1), label=labels[i])
        else:
            plt.plot(test_ts[i], pred_means[i][:,0], 'C1', label='Predictions')
        if not(pred_variances is None):
            plt.fill_between(test_ts[i],
                pred_means[i][:,0]+2*np.sqrt(pred_variances[i][:,0,0]),
                pred_means[i][:,0]-2*np.sqrt(pred_variances[i][:,0,0]),
                color='black',alpha=.5,label='Uncertainty')
        else:
            plt.fill_between(test_ts[i],
                upper_confidence[i],
                lower_confidence[i],
                color='black',alpha=.5,label='Uncertainty')

