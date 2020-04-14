import os
import sys
import warnings

from datetime import date, timedelta

sys.stdout = open(os.devnull, "w")
from mlfinlab.data_structures import standard_data_structures
sys.stdout = sys.__stdout__

import pandas as pd
import requests_cache
import yfinance as yf

warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas_datareader import data as pdr

yf.pdr_override()

def get_daily_returns(tickers, start, end, log=False):
  session = requests_cache.CachedSession(backend='sqlite', expire_after=timedelta(days=1))
  data = pdr.get_data_yahoo(tickers, start=start, end=end, session=session, auto_adjust=True)
  close = data['Close']
  close.index = pd.to_datetime(close.index)
  returns = pd.log(close).diff() if log else close.pct_change(1) # https://mathbabe.org/2011/08/30/why-log-returns/
  return returns.dropna()

def get_volume_bars(tickers, start, end, log=False):
  session = requests_cache.CachedSession(backend='sqlite', expire_after=timedelta(days=1))
  # use S&P 500 as a volume proxy instead of tring to combine basket of volume bars
  market_proxy_data = pdr.get_data_yahoo('SPY', start=start, end=end, session=session, interval='5m', auto_adjust=True)
  market_proxy_data.reset_index(level=0, inplace=True)
  market_proxy_data = pd.concat([market_proxy_data['Datetime'], market_proxy_data['Close'], market_proxy_data['Volume']], axis=1)
  market_proxy_data.columns = ['date', 'price', 'volume']
  avg_market_proxy_daily_volume = market_proxy_data['volume'].sum() / 59
  market_proxy_bars = standard_data_structures.get_volume_bars(market_proxy_data, threshold=avg_market_proxy_daily_volume, verbose=False)
  data = pdr.get_data_yahoo(tickers, start=start, end=end, session=session, interval='5m', auto_adjust=True)
  # sample tickers using SPY's volume bar date times
  bars = data[data.index.isin(market_proxy_bars['date_time'])]
  returns = pd.log(bars['Close']).diff() if log else bars['Close'].pct_change(1)
  return returns.dropna()
  