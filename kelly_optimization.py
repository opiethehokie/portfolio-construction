import numpy as np
import pandas as pd

from datetime import date
from dateutil.relativedelta import relativedelta

from cvxopt import matrix, solvers

from lib import get_time_interval_returns, robust_covariances, print_stats

# https://breakingthemarket.com/convergence-time/
def get_returns(end_date=date.today()):
  return [
    get_time_interval_returns(tickers, end_date + relativedelta(months=-5), end_date, interval='1wk'),
    get_time_interval_returns(tickers, end_date + relativedelta(months=-12), end_date, interval='1wk')
  ]

# https://epchan.blogspot.com/2014/08/kelly-vs-markowitz-portfolio.html?m=1
# https://github.com/jeromeku/Python-Financial-Tools/blob/master/portfolio.py
# https://github.com/thk3421-models/KellyPortfolio/blob/main/kelly.py
def kelly_weight_optimization(returns, cov, trading_periods=52, opt_leverage=1000.0, actual_leverage=2.0):
  C = cov * trading_periods
  M = returns.mean() * trading_periods
  #F = np.linalg.inv(C) @ M
  n = returns.shape[1]
  S = matrix(C.to_numpy())
  q = matrix(M.to_numpy())
  G = matrix(np.vstack((-matrix(np.eye(n)), matrix(np.eye(n)))))
  h = matrix(np.vstack((matrix(0.0, (n, 1)), matrix(opt_leverage / actual_leverage, (n, 1))))) # min/max weights
  #G = -matrix(np.eye(n))
  #h = matrix(0.0, (n ,1)) # long-only weights
  A = matrix(1.0, (1, n))
  b = matrix(opt_leverage) # Ax = b
  solvers.options['show_progress'] = False
  F = np.array(solvers.qp(S, -q, G, h, A, b)['x']).flatten()
  F *= (actual_leverage / opt_leverage)
  return pd.DataFrame.from_dict({ ticker : np.round(weight * 100, 1) for ticker, weight in zip(returns.columns.values.tolist(), F)}, orient='index')

tickers = ['VOO', 'TLT', 'SGOL', 'MINT']

multi_returns = get_returns()
allocations = []

for returns in multi_returns:
  covs = robust_covariances(returns)
  for cov in covs:
    cov = pd.DataFrame(cov, index=tickers, columns=tickers)
    allocations.append(kelly_weight_optimization(returns, cov))

soft_majority_vote = pd.concat(allocations).groupby(level=0).mean().round(1) # simple bagging ensemble
print(soft_majority_vote)

returns = get_time_interval_returns(tickers, date.today() + relativedelta(days=-30), date.today(), return_type='log')
weights = soft_majority_vote[0].values / 100
print_stats(returns, weights, 252)
