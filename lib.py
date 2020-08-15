import math
import os
import sys
import warnings

from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests_cache
import scipy.stats as ss
import yfinance as yf

sys.stdout = open(os.devnull, "w")
from mlfinlab.data_structures import standard_data_structures
sys.stdout = sys.__stdout__

from mlfinlab.features.fracdiff import frac_diff
from mlfinlab.codependence.information import get_optimal_number_of_bins
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas_datareader import data as pdr
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.stattools import adfuller

yf.pdr_override()

def get_daily_prices(tickers, start, end):
  session = requests_cache.CachedSession(backend='sqlite', expire_after=timedelta(days=1))
  data = pdr.get_data_yahoo(tickers, start=start, end=end, session=session, auto_adjust=True)
  close = data['Close']
  close.index = pd.to_datetime(close.index)
  return close.fillna(method='ffill')

def get_daily_returns(tickers, start, end, return_type='percent'):
  close = get_daily_prices(tickers, start, end)
  if return_type == 'fractional':
    close = pd.DataFrame(np.log(close)).diff().dropna()
    returns = frac_diff(close, .1) # https://mlfinlab.readthedocs.io/en/latest/implementations/frac_diff.html
    # want diff where ADF critical vals < -2.83 for 95% confidence
    #print(returns.dropna().apply(adfuller, maxlag=1, regression='c', autolag=None))
  elif return_type == 'log':
    returns = pd.DataFrame(np.log(close)).diff() # https://mathbabe.org/2011/08/30/why-log-returns/
  else:
    returns = close.pct_change(1)
  return returns.dropna()

# https://mlfinlab.readthedocs.io/en/latest/implementations/data_structures.html#volume-bars
def get_volume_bar_returns(tickers, start, end, log=False):
  session = requests_cache.CachedSession(backend='sqlite', expire_after=timedelta(days=1))
  # use S&P 500 as a volume proxy instead of tring to combine basket of volume bars
  market_proxy_data = pdr.get_data_yahoo('SPY', start=start, end=end, session=session, interval='5m', auto_adjust=True)
  market_proxy_data.reset_index(level=0, inplace=True)
  market_proxy_data = pd.concat([market_proxy_data['Datetime'], market_proxy_data['Close'], market_proxy_data['Volume']], axis=1)
  market_proxy_data.columns = ['date', 'price', 'volume']
  avg_market_proxy_daily_volume = market_proxy_data['volume'].sum() / 59
  market_proxy_bars = standard_data_structures.get_volume_bars(market_proxy_data, threshold=avg_market_proxy_daily_volume, verbose=False)
  data = pdr.get_data_yahoo(tickers, start=start, end=end, session=session, interval='5m', auto_adjust=True).fillna(method='ffill')
  # sample tickers using SPY's volume bar date times
  bars = data[data.index.isin(market_proxy_bars['date_time'])]
  returns = pd.DataFrame(np.log(bars['Close'])).diff() if log else bars['Close'].pct_change(1)
  return returns.dropna()

# https://blog.thinknewfound.com/2016/10/shock-covariance-system/
def shock_cov_matrix(returns, n=1):
  cov = returns.cov()
  perturbed_covs = []
  for _ in range(n):
    eig_vals, eig_vecs = np.linalg.eig(cov)
    kth_eig_val = np.random.choice(eig_vals, p=[v / eig_vals.sum() for v in eig_vals])
    k = np.nonzero(eig_vals == kth_eig_val)
    perturbed_kth_eig_val = kth_eig_val * math.exp(np.random.normal(0, 1)) # exponential scaling
    eig_vals[k] = perturbed_kth_eig_val
    perturbed_covs.append(np.linalg.multi_dot([eig_vecs, np.diag(eig_vals), eig_vecs.T]))
  perturbed_cov = np.mean(np.array(perturbed_covs), axis=0)
  return pd.DataFrame(perturbed_cov, columns=returns.columns, index=returns.columns)

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
