
# https://qoppac.blogspot.com/2018/12/portfolio-construction-through.html
# https://qoppac.blogspot.com/2018/12/portfolio-construction-through_7.html
# https://qoppac.blogspot.com/2018/12/portfolio-construction-through_14.html

import numpy as np
import pandas as pd
import requests_cache
import datetime
import sys
import pprint
import yfinance as yf

from pandas_datareader import data as pdr
from handcrafting import Portfolio, NO_RISK_TARGET, NO_TOP_LEVEL_WEIGHTS # from https://gist.github.com/robcarver17/c2e9c594af31894b9942396f719da449

yf.pdr_override()

tickers = ['LEMB','SGOL','PPEM','PDBC','PPLC','PPSC','LTPZ','VNQ','VNQI','JPHY','PPDM','TMF','TYD'] #,'DIVY','BTAL']

# https://blog.thinknewfound.com/2017/11/risk-parity-much-data-use-estimating-volatilities-correlations/
expire_after = datetime.timedelta(days=1)
start = datetime.datetime(2018, 9, 17)
end = datetime.datetime(2019, 9, 13)

allow_leverage = False
risk_target = NO_RISK_TARGET
use_SR_estimates = False
top_level_weights = NO_TOP_LEVEL_WEIGHTS

def calc_weekly_return_percents(ticker):
    session = requests_cache.CachedSession(cache_name=ticker, backend='sqlite', expire_after=expire_after)
    data = pdr.get_data_yahoo(ticker, start=start, end=end, session=session)
    close = data['Adj Close']
    close.index = pd.to_datetime(close.index)
    weekly_close = close.resample('W-FRI').ffill()
    new_values = weekly_close[1:].values
    old_values = weekly_close[:-1].values
    return (new_values - old_values) / old_values
    # https://mathbabe.org/2011/08/30/why-log-returns/
    #return np.log(np.divide(new_values, old_values))

returns = pd.DataFrame(dict([(ticker, calc_weekly_return_percents(ticker)) for ticker in tickers]))

p=Portfolio(returns, risk_target=risk_target, top_level_weights=top_level_weights, allow_leverage=allow_leverage, use_SR_estimates=use_SR_estimates)
#print(p.corr_matrix)
#print('sharpe ratios: ', p.sharpe_ratio)
##print(p.diags)
pprint.pprint(p.show_subportfolio_tree())
pprint.pprint(dict([(instr, round(wt * 100, 1)) for instr, wt in zip(p.instruments, p.cash_weights)]))

