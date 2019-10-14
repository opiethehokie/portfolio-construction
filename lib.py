from datetime import date, timedelta

import pandas as pd
import requests_cache
import yfinance as yf

from pandas_datareader import data as pdr

yf.pdr_override()

def get_returns(tickers, start, end):
  session = requests_cache.CachedSession(backend='sqlite', expire_after=timedelta(days=1))
  data = pdr.get_data_yahoo(tickers, start=start, end=end, session=session)
  close = data['Adj Close']
  close.index = pd.to_datetime(close.index)
  #new_values = close[1:].values
  #old_values = close[:-1].values
  #return np.log(np.divide(new_values, old_values))
  #return (new_values - old_values) / old_values
  returns = close.pct_change(1).dropna()
  return returns
