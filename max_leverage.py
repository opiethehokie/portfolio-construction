import pprint

import numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

from lib import get_daily_returns

trading_days_per_year = 252
daily_risk_free_rate = .0011 / trading_days_per_year # daily 3-month treasury rate

# http://ddnum.com/articles/leveragedETFs.php
# https://rhsfinancial.com/2017/06/20/line-aggressive-crazy-leverage/
# https://blog.thinknewfound.com/2018/01/levered-etfs-long-run/
def kelly_max_leverage(returns):
    returns -= daily_risk_free_rate
    mean_daily_returns = returns.mean() * trading_days_per_year
    var = returns.var() * trading_days_per_year
    return np.round(mean_daily_returns / var, 1) # ignores skewness and kurtosis

# https://epchan.blogspot.com/2014/08/kelly-vs-markowitz-portfolio.html?m=1
def kelly_weight_optimization(returns, fraction=None, target_leverage=None):
    returns -= daily_risk_free_rate
    C = returns.cov() * trading_days_per_year
    M = returns.mean() * trading_days_per_year
    F = np.linalg.inv(C).dot(M)
    if fraction:
        F *= fraction
    if target_leverage:
        ones = np.ones(returns.shape[1]) / target_leverage
        F = np.divide(F, ones.T.dot(F))
    return { ticker : round(leverage * 100, 1) for ticker, leverage in zip(returns.columns.values.tolist(), F)}

tickers = ['VTI','VXUS','BND','BNDX']

# constant - should still rebalance at least quarterly -> monthly
returns = get_daily_returns(tickers, date.today() + relativedelta(months=-60), date.today())
print(kelly_max_leverage(returns))

weights = kelly_weight_optimization(returns, fraction=.5)
pprint.pprint(weights)

# dynamic based on volatility
returns = get_daily_returns(tickers, date.today() + relativedelta(days=-60), date.today())
print(kelly_max_leverage(returns))

weights = kelly_weight_optimization(returns, target_leverage=1)
pprint.pprint(weights)
