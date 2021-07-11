import numpy as np
import pandas as pd

from datetime import date
from dateutil.relativedelta import relativedelta

from cvxopt import matrix, solvers

from lib import get_time_interval_returns, get_volume_bar_returns, bootstrap_returns, robust_covariances, print_stats

def get_returns(end_date=date.today()):
    return [
        get_volume_bar_returns(tickers, end_date + relativedelta(days=-59), end_date),
        get_time_interval_returns(tickers, end_date + relativedelta(months=-24), end_date, return_type='fractional', interval='1d'),
        get_time_interval_returns(tickers, end_date + relativedelta(months=-36), end_date, interval='1wk'),
        get_time_interval_returns(tickers, end_date + relativedelta(months=-60), end_date, interval='1mo')
    ]

# https://epchan.blogspot.com/2014/08/kelly-vs-markowitz-portfolio.html?m=1
# https://github.com/jeromeku/Python-Financial-Tools/blob/master/portfolio.py
def kelly_weight_optimization(returns, cov, trading_days=252, min_size=.03, max_size=.25):
    C = cov * trading_days
    M = returns.mean() * trading_days
    if min_size and max_size:
        n = returns.shape[1]
        S = matrix(C.to_numpy())
        q = matrix(M.to_numpy())
        G = matrix(np.vstack((matrix(np.eye(n) * -1), matrix(np.eye(n) * 1))))
        h = matrix(np.vstack((matrix(-min_size, (n, 1)), matrix(max_size, (n,1)))))
        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        solvers.options['show_progress'] = False
        F = np.array(solvers.qp(S, -q, G, h, A, b)['x']).flatten()
    else:
        F = np.linalg.inv(C).dot(M) # same as solvers.qp(S, -q) i.e. no constraints
        # fractional kelly
        #F *= fraction
        # target leverage
        #ones = np.ones(returns.shape[1]) / leverage
        #F = np.divide(F, ones.T.dot(F))
    return pd.DataFrame.from_dict({ ticker : np.round(leverage * 100, 1) for ticker, leverage in zip(returns.columns.values.tolist(), F)}, orient='index')

tickers = ['VTI','VEA','VWO','VNQ','VNQI','SGOL','PDBC','SCHP','TYD','LEMB','FMF','GBTC']

multi_returns = get_returns()
allocations = []

for returns in multi_returns:
    returns = bootstrap_returns(returns, method='block')
    covs = robust_covariances(returns)
    for cov in covs:
        cov = pd.DataFrame(cov, index=tickers, columns=tickers)
        allocations.append(kelly_weight_optimization(returns, cov, min_size=.03, max_size=.25))

soft_majority_vote = pd.concat(allocations).groupby(level=0).mean().round(3) # simple bagging ensemble
print(soft_majority_vote)

returns = get_time_interval_returns(tickers, date.today() + relativedelta(days=-60), date.today(), return_type='log')
weights = soft_majority_vote[0].values / 100
print_stats(returns, weights, 252)
