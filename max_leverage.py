# http://ddnum.com/articles/leveragedETFs.php
# https://rhsfinancial.com/2017/06/20/line-aggressive-crazy-leverage/
# https://blog.thinknewfound.com/2018/01/levered-etfs-long-run/

from datetime import date
from dateutil.relativedelta import relativedelta

from lib import get_daily_returns

tickers = ['VOO']
daily_risk_free_rate = (.0092 - .0003) / 250 # ETF ER for leverage instead of 3-month treasury rate

def max_leverage(returns):
    mean_daily_returns = returns.mean(axis=0)
    var = returns.std() * returns.std()
    return (mean_daily_returns - daily_risk_free_rate) / var # ignores skewness and kurtosis

# constant - should still rebalance at least quarterly -> monthly
returns = get_daily_returns(tickers, date.today() + relativedelta(months=-60), date.today())
print(max_leverage(returns))

# dynamic based on volatility
returns = get_daily_returns(tickers, date.today() + relativedelta(days=-60), date.today())
print(max_leverage(returns))
