import numpy as np
import scipy.stats as stats

from datetime import date
from dateutil.relativedelta import relativedelta

from lib import get_time_interval_returns


trading_days_per_year = 251
tickers = ['BKLN','FMF','LEMB','PDBC','SCHP','SGOL','TYD','VEA','VNQ','VNQI','VTI','VWO']
weights = np.array([.119,.224,.067,.055,.217,.087,.072,.034,.031,.03,.034,.03])

assert len(tickers) == len(weights)
assert sum(weights) == 1

daily_returns = get_time_interval_returns(tickers, date.today() + relativedelta(months=-24), date.today())

# https://investresolve.com/blog/tag/independent-bets/
# https://thequantmba.wordpress.com/2017/06/06/max-diversification-in-python/

weighted_daily_returns = daily_returns * weights
weighted_daily_portfolio_returns = np.sum(weighted_daily_returns, axis=1)
weighted_daily_var = weights.T @ daily_returns.cov() @ weights

individual_vol = np.sqrt(trading_days_per_year * daily_returns.var()) * 100
portfolio_vol = np.sqrt(trading_days_per_year * weighted_daily_var) * 100
independent_bets = np.divide(weights.T @ daily_returns.std(), np.sqrt(weighted_daily_var))**2
portfolio_skew = stats.skew(weighted_daily_portfolio_returns) / np.sqrt(12)
portfolio_kurtosis = stats.kurtosis(weighted_daily_portfolio_returns) / 12

print()
print(daily_returns.corr())
print()
print(individual_vol.to_string())
print()
print('total annual vol: %.2f%%' % portfolio_vol)
print('monthly skew: %.2f (positive good)' % portfolio_skew)
print('monthly kurtosis: %.2f (fat tails above 0)' % portfolio_kurtosis)
print('independent bets: %.2f' % independent_bets)
