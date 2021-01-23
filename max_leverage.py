import pprint

import numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

from cvxopt import matrix, solvers

from lib import get_time_interval_returns

trading_days_per_year = 252
daily_risk_free_rate = (.0009 * 4) / trading_days_per_year # daily 3-month treasury rate

# http://ddnum.com/articles/leveragedETFs.php
# https://rhsfinancial.com/2017/06/20/line-aggressive-crazy-leverage/
# https://blog.thinknewfound.com/2018/01/levered-etfs-long-run/
def kelly_max_leverage(returns):
    returns -= daily_risk_free_rate
    mean_daily_returns = returns.mean() * trading_days_per_year
    var = returns.var() * trading_days_per_year
    return np.round(mean_daily_returns / var, 1)

# https://epchan.blogspot.com/2014/08/kelly-vs-markowitz-portfolio.html?m=1
# https://github.com/jeromeku/Python-Financial-Tools/blob/master/portfolio.py
def kelly_weight_optimization(returns, fraction=None, target_leverage=None, long_only=False):
    if fraction or (target_leverage and target_leverage > 1):
        returns -= daily_risk_free_rate
    C = returns.cov() * trading_days_per_year
    M = returns.mean() * trading_days_per_year
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
    
    return { ticker : np.round(leverage * 100, 1) for ticker, leverage in zip(returns.columns.values.tolist(), F)}

tickers = ['VTI','VXUS','BND','BNDX']

returns = get_time_interval_returns(tickers, date.today() + relativedelta(months=-60), date.today(), return_type='log')

pprint.pprint(kelly_weight_optimization(returns, target_leverage=1, long_only=True))
