# http://ddnum.com/articles/leveragedETFs.php
# https://rhsfinancial.com/2017/06/20/line-aggressive-crazy-leverage/
# https://blog.thinknewfound.com/2018/01/levered-etfs-long-run/

import numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

from lib import get_daily_returns

trading_days_per_year = 251
daily_risk_free_rate = .0011 / trading_days_per_year # daily 3-month treasury rate

def kelly_max_leverage(returns):
    returns -= daily_risk_free_rate
    mean_daily_returns = returns.mean(axis=0) * trading_days_per_year
    var = returns.var() * trading_days_per_year
    return mean_daily_returns / var # ignores skewness and kurtosis

def kelly_weight_optimization(returns):
    returns -= daily_risk_free_rate
    C = returns.cov() * trading_days_per_year
    M = returns.mean() * trading_days_per_year
    F = np.linalg.inv(C).dot(M)
    return { ticker : leverage for ticker, leverage in zip(returns.columns.values.tolist(), F)}

tickers = ['VOO', 'BND']

# constant - should still rebalance at least quarterly -> monthly
returns = get_daily_returns(tickers, date.today() + relativedelta(months=-60), date.today())
print(kelly_max_leverage(returns))

weights = kelly_weight_optimization(returns)
print(weights)
print(sum(weights.values()))

# dynamic based on volatility
returns = get_daily_returns(tickers, date.today() + relativedelta(days=-60), date.today())
print(kelly_max_leverage(returns))

weights = kelly_weight_optimization(returns)
print(weights)
print(sum(weights.values()))
