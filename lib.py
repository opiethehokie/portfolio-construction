import math
import os
import sys
import warnings

from datetime import timedelta

import fastcluster
import networkx as nx
import numpy as np
import pandas as pd
import requests_cache
import scipy.stats as ss
import yfinance as yf

sys.stdout = open(os.devnull, "w")
from mlfinlab.data_structures import standard_data_structures
sys.stdout = sys.__stdout__

from mlfinlab_mods import get_mutual_info, gnpr_distance

from mlfinlab.features.fracdiff import frac_diff
from mlfinlab.codependence.correlation import distance_correlation, angular_distance
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas_datareader import data as pdr
from scipy.cluster.hierarchy import cophenet
from sklearn.utils import resample
from statsmodels.tsa.stattools import adfuller

yf.pdr_override()

def get_time_interval_returns(tickers, start, end, return_type='percent', interval='1d'):
  session = requests_cache.CachedSession(backend='sqlite', expire_after=timedelta(days=1))
  data = pdr.get_data_yahoo(tickers, start=start, end=end, session=session, auto_adjust=True, interval=interval)
  close = data['Close'].dropna()
  close.index = pd.to_datetime(close.index)
  if return_type == 'fractional':
    close = pd.DataFrame(np.log(close)).diff().dropna()
    returns = frac_diff(close, .1) # https://mlfinlab.readthedocs.io/en/latest/implementations/frac_diff.html
    #print(returns.dropna().apply(adfuller, maxlag=1, regression='c', autolag=None)) # want diff where ADF critical vals < -2.83 for 95% confidence
  elif return_type == 'log':
    returns = pd.DataFrame(np.log(close)).diff() # https://mathbabe.org/2011/08/30/why-log-returns/
  else:
    returns = close.pct_change(1)
  return returns.dropna()

# https://mlfinlab.readthedocs.io/en/latest/implementations/data_structures.html#volume-bars
def get_volume_bar_returns(tickers, start, end, log=True):
  session = requests_cache.CachedSession(backend='sqlite', expire_after=timedelta(days=1))
  # use S&P 500 as a volume proxy instead of tring to combine basket of volume bars
  market_proxy_data = pdr.get_data_yahoo('SPY', start=start, end=end, session=session, interval='5m', auto_adjust=True)
  market_proxy_data.reset_index(level=0, inplace=True)
  market_proxy_data = pd.concat([market_proxy_data['Datetime'], market_proxy_data['Close'], market_proxy_data['Volume']], axis=1)
  market_proxy_data.columns = ['date', 'price', 'volume']
  avg_market_proxy_daily_volume = market_proxy_data['volume'].sum() / 59
  market_proxy_bars = standard_data_structures.get_volume_bars(market_proxy_data, threshold=avg_market_proxy_daily_volume, verbose=False)
  data = pdr.get_data_yahoo(tickers, start=start, end=end, session=session, interval='5m', auto_adjust=True).fillna(method='ffill').fillna(method='bfill')
  # sample tickers using SPY's volume bar date times
  bars = data[data.index.isin(market_proxy_bars['date_time'])]
  returns = pd.DataFrame(np.log(bars['Close'])).diff() if log else bars['Close'].pct_change(1)
  return returns.dropna()

# https://blog.thinknewfound.com/2016/10/shock-covariance-system/
# https://sixfigureinvesting.com/2016/03/modeling-stock-market-returns-with-laplace-distribution-instead-of-normal/
def shock_cov_matrix(returns, n=1):
  cov = returns.cov()
  perturbed_covs = []
  rng = np.random.default_rng()
  for _ in range(n):
    eig_vals, eig_vecs = np.linalg.eig(cov)
    kth_eig_val = rng.choice(eig_vals, p=[v / eig_vals.sum() for v in eig_vals])
    k = np.nonzero(eig_vals == kth_eig_val)
    perturbed_kth_eig_val = kth_eig_val * math.exp(rng.laplace(0, 1.2))
    eig_vals[k] = perturbed_kth_eig_val
    perturbed_covs.append(np.linalg.multi_dot([eig_vecs, np.diag(eig_vals), eig_vecs.T]))
  perturbed_cov = np.mean(np.array(perturbed_covs), axis=0)
  return pd.DataFrame(perturbed_cov, columns=returns.columns, index=returns.columns)

# https://hudsonthames.org/portfolio-optimisation-with-mlfinlab-estimation-of-risk/
def robust_covariances(returns):
  return [
    RiskEstimators().minimum_covariance_determinant(returns, price_data=False),
    RiskEstimators().empirical_covariance(returns, price_data=False),
    RiskEstimators().shrinked_covariance(returns, price_data=False, shrinkage_type='lw'),
    RiskEstimators().exponential_covariance(returns, price_data=False),
    RiskEstimators.corr_to_cov(returns.corr(method=distance_correlation), returns.std()),
    RiskEstimators.corr_to_cov(returns.corr(method=angular_distance), returns.std()),
    RiskEstimators.corr_to_cov(returns.corr(method=get_mutual_info), returns.std()),
    RiskEstimators.corr_to_cov(returns.corr(method=gnpr_distance), returns.std()),
    RiskEstimators().denoise_covariance(returns.cov(), returns.shape[0] / returns.shape[1], denoise_method='target_shrink'),
    RiskEstimators().denoise_covariance(returns.cov(), returns.shape[0] / returns.shape[1], denoise_method='const_resid_eigen', detone=True),
    shock_cov_matrix(returns)
  ]

# https://mlfinlab.readthedocs.io/en/latest/data_generation/bootstrap.html
def bootstrap_returns(returns, method='row'):
  if method == 'row':
    return resample(returns)
  elif method == 'block':
    return resample(returns, stratify=returns)
  else:
    return returns

# https://investresolve.com/blog/tag/independent-bets/
# https://thequantmba.wordpress.com/2017/06/06/max-diversification-in-python/
def print_stats(returns, weights, trading_periods):
  weighted_returns = returns * weights
  weighted_portfolio_returns = np.sum(weighted_returns, axis=1)
  weighted_var = weights.T @ returns.cov() @ weights
  portfolio_vol = np.sqrt(trading_periods * weighted_var) * 100
  independent_bets = np.divide(weights.T @ returns.std(), np.sqrt(weighted_var))**2
  portfolio_skew = ss.skew(weighted_portfolio_returns) / np.sqrt(12)
  portfolio_kurtosis = ss.kurtosis(weighted_portfolio_returns) / 12
  #print(returns.corr())
  print('total annual vol: %.2f%%' % portfolio_vol)
  print('independent bets: %.2f' % independent_bets)
  print('monthly skew: %.2f (positive good)' % portfolio_skew)
  print('monthly kurtosis: %.2f (fat tails above 0)' % portfolio_kurtosis)

# https://marti.ai/qfin/2020/08/14/correlation-matrix-features.html
# https://gmarti.gitlab.io/qfin/2020/09/04/correlation-matrix-features-market-regimes.html
def extract_features(corr):
  n = corr.shape[0]
  a, b = np.triu_indices(n, k=1)
  features = pd.Series()
  # coefficients
  coeffs = pd.Series(corr[a, b].flatten())
  coeffs_stats = coeffs.describe()
  for stat in coeffs_stats.index[1:]:
    features[f'coeffs_{stat}'] = coeffs_stats[stat]
  features['coeffs_1%'] = coeffs.quantile(q=0.01)
  features['coeffs_99%'] = coeffs.quantile(q=0.99)
  features['coeffs_10%'] = coeffs.quantile(q=0.1)
  features['coeffs_90%'] = coeffs.quantile(q=0.9)
  # eigenvals
  eigenvals, eigenvecs = np.linalg.eig(corr)
  permutation = np.argsort(eigenvals)[::-1]
  eigenvals = eigenvals[permutation]
  eigenvecs = eigenvecs[:, permutation]
  pf_vector = eigenvecs[:, np.argmax(eigenvals)]
  if len(pf_vector[pf_vector < 0]) > len(pf_vector[pf_vector > 0]):
      pf_vector = -pf_vector
  features['varex_eig1'] = float(eigenvals[0] / sum(eigenvals))
  features['varex_eig_top5'] = (float(sum(eigenvals[:5])) / float(sum(eigenvals)))
  features['varex_eig_top30'] = (float(sum(eigenvals[:30])) / float(sum(eigenvals)))
  # variance explained by eigenvals outside of the Marcenko-Pastur (MP) distribution
  T, N = 252, n
  MP_cutoff = (1 + np.sqrt(N / T))**2
  features['varex_eig_MP'] = (float(sum([e for e in eigenvals if e > MP_cutoff])) / float(sum(eigenvals)))
  # determinant
  features['determinant'] = np.prod(eigenvals)
  # condition number
  features['condition_number'] = abs(eigenvals[0]) / abs(eigenvals[-1])
  # stats of the first eigenvector entries
  pf_stats = pd.Series(pf_vector).describe()
  for stat in pf_stats.index[1:]:
    features[f'pf_{stat}'] = float(pf_stats[stat])
  # stats on the MST
  dist = (1 - corr) / 2
  G = nx.from_numpy_matrix(dist) 
  mst = nx.minimum_spanning_tree(G)
  features['mst_avg_shortest'] = nx.average_shortest_path_length(mst)
  closeness_centrality = (pd.Series(list(nx.closeness_centrality(mst).values())).describe())
  for stat in closeness_centrality.index[1:]:
    features[f'mst_centrality_{stat}'] = closeness_centrality[stat]
  # stats on the linkage
  dist = np.sqrt(2 * (1 - corr))
  for algo in ['ward', 'single', 'complete', 'average']:
      Z = fastcluster.linkage(dist[a, b], method=algo)
      features[f'coph_corr_{algo}'] = cophenet(Z, dist[a, b])[0]
  return features.sort_index()
