import numpy as np
import scipy.stats as ss

from mlfinlab.codependence.information import get_optimal_number_of_bins
from sklearn.metrics import mutual_info_score

# https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/codependence/information.py
def get_mutual_info(x, y):
  corr_coef = np.corrcoef(x, y)[0][1]
  n_bins = get_optimal_number_of_bins(x.shape[0], corr_coef=corr_coef)
  contingency = np.histogram2d(x, y, n_bins)[0]
  mutual_info = mutual_info_score(None, None, contingency=contingency)
  marginal_x = ss.entropy(np.histogram(x, n_bins)[0])
  marginal_y = ss.entropy(np.histogram(y, n_bins)[0])
  mutual_info /= min(marginal_x, marginal_y)
  return mutual_info

# https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/codependence/gnpr_distance.py
def gnpr_distance(x, y, theta=.5, bandwidth=.1):
  num_obs = x.shape[0]
  dist_1 = 3 / (num_obs * (num_obs**2 - 1)) * (np.power(x - y, 2).sum())
  min_val = min(x.min(), y.min())
  max_val = max(x.max(), y.max())
  bins = np.arange(min_val, max_val + bandwidth, bandwidth)
  hist_x = np.histogram(x, bins)[0] / num_obs
  hist_y = np.histogram(y, bins)[0] / num_obs
  dist_0 = np.power(hist_x**(1/2) - hist_y**(1/2), 2).sum() / 2
  distance = theta * dist_1 + (1 - theta) * dist_0
  return distance**(1/2)
