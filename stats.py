#  the number of independent risk factors expressed in a portfolio is equal to the square of the Diversification Ratio of the portfolio

# https://investresolve.com/blog/tag/independent-bets/
# https://thequantmba.wordpress.com/2017/06/06/max-diversification-in-python/

import numpy as np
import scipy.stats as stats

from datetime import date
from dateutil.relativedelta import relativedelta

from lib import get_returns


def independent_bets(returns, weights):
    covariances = returns.cov()
    std = np.sqrt(np.diag(covariances))
    diversification_ratio = np.divide(weights.T @ std, np.sqrt(weights.T @ covariances @ weights))
    return np.square(diversification_ratio)


#tickers = ['PPLC','PPDM','PPEM','VNQ','VNQI','SGOL','PDBC','BKLN','VTIP','TYD','EDV','VWOB','BWX']
#weights = np.array([.05, .09, .07, .04, .05, .09, .05, .09, .22, .08, .04, .04, .09])
#tickers = ['NTSX','VIOO','VFMF','VFLQ','VEA','INTF','VSS','ISCF','VWO','EMGF','EWX','FM','VNQ','VNQI','MNA','JPHY','DIVY']
#weights = np.array([.10, .08, .18, .03, .09, .09, .09, .09, .03, .03, .03, .01, .04, .05, .02, .02, .02])
tickers = ['GMOM','WTMF','JPMF','VAMO','VMOT','ROMO','VTI','VXUS','DZK','UPRO','JKL']
weights = np.array([.20, .05, .05, .05, .05, .10, .18, .22, .015, .035, .05])
#tickers = ['VT','VTEB']
#weights = np.array([.80, .20])
#tickers = ['VOO','VGIT']
#weights = np.array([.60, .40])

assert len(tickers) == len(weights)
assert sum(weights) == 1

returns = get_returns(tickers, date.today() + relativedelta(months=-24), date.today())
weighted_returns = returns * weights

print()
print(independent_bets(returns, weights))
print()
print(returns.corr())
print()
individual_vol = np.sqrt(252 * returns.var()) * 100
print(individual_vol.to_string())
print()
portfolio_vol = np.sum(np.sqrt(252 * weighted_returns.var())) * 100
print(portfolio_vol)
print()
portfolio_skew = stats.skew(np.sum(weighted_returns, axis=1)) # greater than 0 means more right tail than left (good)
print(portfolio_skew)
portfolio_kurtosis = stats.kurtosis(np.sum(weighted_returns, axis=1)) # greater than 0 means fatter tails than normal distribution (bad)
print(portfolio_kurtosis)
