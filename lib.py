import warnings

from datetime import date, timedelta
from mlfinlab.data_structures import standard_data_structures

import pandas as pd
import requests_cache
import yfinance as yf

warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas_datareader import data as pdr

yf.pdr_override()

def get_daily_returns(tickers, start, end):
  session = requests_cache.CachedSession(backend='sqlite', expire_after=timedelta(days=1))
  data = pdr.get_data_yahoo(tickers, start=start, end=end, session=session, auto_adjust=True)
  close = data['Close']
  close.index = pd.to_datetime(close.index)
  returns = close.pct_change(1).dropna() # https://mathbabe.org/2011/08/30/why-log-returns/
  return returns

def get_volume_bars(tickers, start, end):
  session = requests_cache.CachedSession(backend='sqlite', expire_after=timedelta(days=1))
  data = pdr.get_data_yahoo(tickers, start=start, end=end, session=session, interval='15m', auto_adjust=True)
  data.reset_index(level=0, inplace=True)
  data = pd.concat([data['Datetime'], data['Close'], data['Volume']], axis=1)
  data.columns = ['date', 'price', 'volume']
  bars = standard_data_structures.get_volume_bars(data, threshold=2000000, verbose=True)
  print(bars)
  

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
get_volume_bars(['VTI'], date.today() + relativedelta(days=-59), date.today())