import math
import multiprocessing as mp
import random
import warnings

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from lib import get_time_interval_returns, get_volume_bar_returns, bootstrap_returns, robust_covariances, print_stats
from lib import get_mutual_info # to remove extra parameters
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity 
from mlfinlab.portfolio_optimization.herc import HierarchicalEqualRiskContribution

# graph theory (tree instead of complete correlation graph) and machine learning (clustering) based optimization
# instability, concentration, and underperformance improvement on convex optimization (quadratic programming)

def get_returns(end_date=date.today()):
  short_term_log = get_volume_bar_returns(tickers, date.today() + relativedelta(days=-59), date.today())
  med_term_frac = get_time_interval_returns(tickers, date.today() + relativedelta(months=-30), end_date, return_type='fractional', interval='1d')
  long_term_monthly = get_time_interval_returns(tickers, end_date + relativedelta(months=-62), end_date, interval='1mo')
  return [short_term_log, med_term_frac, long_term_monthly]


# https://mlfinlab.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_equal_risk_contribution.html
def herc_model(returns, cov, linkage, metric):
  herc = HierarchicalEqualRiskContribution()
  herc.allocate(asset_names=returns.columns, asset_returns=returns, covariance_matrix=cov, linkage=linkage, risk_measure=metric)
  return herc.weights.transpose().sort_index()

if __name__ == '__main__':

  tickers = ['VTI','VEA','VWO','VNQ','VNQI','SGOL','PDBC','SCHP','TYD','LEMB','FMF','GBTC']

  multi_returns = get_returns()
  linkages = ['single', 'complete', 'ward']
  metrics = ['variance', 'standard_deviation', 'equal_weighting', 'expected_shortfall', 'conditional_drawdown_risk'] # variance is original HRP
  pool = mp.Pool(4)
  allocations = []

  for returns in multi_returns:
    returns = bootstrap_returns(returns, method='block')
    covs = robust_covariances(returns)
    for cov in covs:
      for linkage in linkages:
        for metric in metrics:
          pool.apply_async(herc_model, args=(returns, cov, linkage, metric), callback=lambda result: allocations.append(result))

  pool.close()
  pool.join()

  soft_majority_vote = pd.concat(allocations).groupby(level=0).mean().round(3) * 100 # simple bagging ensemble
  print(soft_majority_vote)

  returns = get_time_interval_returns(tickers, date.today() + relativedelta(years=-5), date.today(), return_type='log')
  #print(returns.corr())
  weights = soft_majority_vote[0].values / 100
  print_stats(returns, weights, 250)

# backtest at https://www.portfoliovisualizer.com/backtest-portfolio?s=y&timePeriod=2&startYear=1985&firstMonth=1&endYear=2020&lastMonth=12&calendarAligned=true&includeYTD=false&initialAmount=10000&annualOperation=0&annualAdjustment=0&inflationAdjusted=true&annualPercentage=0.0&frequency=4&rebalanceType=4&absoluteDeviation=5.0&relativeDeviation=25.0&showYield=false&reinvestDividends=true&portfolioNames=false&portfolioName1=Portfolio+1&portfolioName2=Portfolio+2&portfolioName3=Portfolio+3&symbol1=BKLN&allocation1_1=7.5&symbol2=FMF&allocation2_1=15.9&symbol3=LEMB&allocation3_1=5.5&symbol4=PDBC&allocation4_1=6.1&symbol5=PPDM&allocation5_1=5.5&symbol6=PPEM&allocation6_1=3.5&symbol7=PPLC&allocation7_1=3.0&symbol8=SGOL&allocation8_1=11.7&symbol9=VTIP&allocation9_1=24.7&symbol10=TYD&allocation10_1=7.6&symbol11=VNQ&allocation11_1=3.0&symbol12=VNQI&allocation12_1=3.0&symbol13=VWO&allocation13_1=3.0
