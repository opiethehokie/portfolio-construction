# http://ddnum.com/articles/leveragedETFs.php

from datetime import date
from dateutil.relativedelta import relativedelta

from lib import get_returns

tickers = ['SPY','VTI','EFA','IEF','TLT','VWO']
returns = get_returns(tickers, date.today() + relativedelta(months=-180), date.today()) # 15 years

mean_daily_returns = returns.mean(axis=0)
var = returns.std() * returns.std()

max_leverage = mean_daily_returns / var # ignores skewness and kurtosis

print(max_leverage)
