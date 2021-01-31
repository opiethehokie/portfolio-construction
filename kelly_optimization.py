import pprint

import numpy as np
import pandas as pd

from datetime import date
from dateutil.relativedelta import relativedelta

from cvxopt import matrix, solvers

from lib import get_time_interval_returns, get_volume_bar_returns, bootstrap_returns, robust_covariances

def get_returns(end_date=date.today()):
  short_term_log = get_volume_bar_returns(tickers, date.today() + relativedelta(days=-59), date.today())
  med_term_frac = get_time_interval_returns(tickers, date.today() + relativedelta(months=-30), end_date, return_type='fractional', interval='1d')
  long_term_monthly = get_time_interval_returns(tickers, end_date + relativedelta(months=-62), end_date, interval='1mo')
  return [short_term_log, med_term_frac, long_term_monthly]

# https://epchan.blogspot.com/2014/08/kelly-vs-markowitz-portfolio.html?m=1
# https://github.com/jeromeku/Python-Financial-Tools/blob/master/portfolio.py
def kelly_weight_optimization(returns, cov, fraction=None, target_leverage=None, long_only=False, trading_days=252, three_month_treasury_rate=.0009):
    if fraction or (target_leverage and target_leverage > 1):
        risk_free_rate = (three_month_treasury_rate * 4) / trading_days
        returns -= risk_free_rate
    C = cov * trading_days
    M = returns.mean() * trading_days
    if long_only:
        n = returns.shape[1]
        S = matrix(C.to_numpy())
        q = matrix(M.to_numpy())
        G = matrix(np.eye(n) * -1)
        h = matrix(np.zeros(n)) # long-only constraint (forces 0 weights in a lot of cases)
        A = None
        b = None
        solvers.options['show_progress'] = False
        F = np.array(solvers.qp(S, -q, G, h, A, b)["x"]).flatten()
    else:
        F = np.linalg.inv(C).dot(M) # same as solvers.qp(S, -q) i.e. no constraints
    if fraction:
        F *= fraction
    elif target_leverage:
        ones = np.ones(returns.shape[1]) / target_leverage
        F = np.divide(F, ones.T.dot(F))
    return pd.DataFrame.from_dict({ ticker : np.round(leverage * 100, 1) for ticker, leverage in zip(returns.columns.values.tolist(), F)}, orient='index')

tickers = ['VTI','VXUS','BND','BNDX']

econ_tree = pd.DataFrame(np.array([
['VTI', 101010, 1010, 10],
['VXUS', 102010, 1020, 10],
['BND', 201010, 2010, 20],
['BNDX', 202010, 2020, 20]
]), columns=['TICKER', 'SECTOR', 'REGION', 'TYPE'])

multi_returns = get_returns()
allocations = []

for returns in multi_returns:
    returns = bootstrap_returns(returns, method='block')
    covs = robust_covariances(returns, econ_tree)
    for cov in covs:
        cov = pd.DataFrame(cov, index=tickers, columns=tickers)
        allocations.append(kelly_weight_optimization(returns, cov, target_leverage=1, long_only=True))

soft_majority_vote = pd.concat(allocations).groupby(level=0).mean().round(3) # simple bagging ensemble
print(soft_majority_vote)
