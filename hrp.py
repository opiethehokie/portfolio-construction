import math
import multiprocessing as mp
import warnings

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from lib import get_daily_returns, get_volume_bar_returns, shock_cov_matrix, get_daily_prices
from lib import get_mutual_info # to remove extra parameters
from hcaa import HierarchicalClusteringAssetAllocation # for custom weight constraints
from mlfinlab.codependence.correlation import distance_correlation, angular_distance, absolute_angular_distance, squared_angular_distance
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators

# graph theory (tree instead of complete correlation graph) and machine learning (clustering) based optimization
# instability, concentration, and underperformance improvement on convex optimization (quadratic programming)

# https://blog.thinknewfound.com/2017/11/risk-parity-much-data-use-estimating-volatilities-correlations/
def get_returns():
  short_term_log = get_volume_bar_returns(tickers, date.today() + relativedelta(days=-59), date.today(), log=True)
  short_term_percent = get_volume_bar_returns(tickers, date.today() + relativedelta(days=-59), date.today(), log=False)
  med_term_log = get_daily_returns(tickers, date.today() + relativedelta(months=-6), end_date, return_type='log')
  med_term_percent = get_daily_returns(tickers, end_date + relativedelta(months=-6), end_date, return_type='percent')
  long_term_frac = get_daily_returns(tickers, end_date + relativedelta(months=-24), end_date, return_type='fractional')
  long_term_percent = get_daily_returns(tickers, end_date + relativedelta(months=-24), end_date, return_type='percent')
  return [short_term_log, short_term_percent,
          med_term_log, med_term_percent,
          long_term_frac, long_term_percent
        ]

def get_prices():
  short_term = get_daily_prices(tickers, end_date + relativedelta(days=-59), end_date)
  med_term = get_daily_prices(tickers, end_date + relativedelta(months=-6), end_date)
  long_term = get_daily_prices(tickers, end_date + relativedelta(months=-24), end_date)
  return [short_term, med_term, long_term]

# https://mlfinlab.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_risk_parity.html
def hrp_model(returns, covariance, shocked, shrinkage):
  cov = shock_cov_matrix(returns) if shocked else RiskEstimators.corr_to_cov(returns.corr(method=covariance), returns.std())
  hrp = HierarchicalRiskParity()
  hrp.allocate(asset_names=returns.columns, covariance_matrix=cov, use_shrinkage=shrinkage)
  return hrp.weights.transpose().sort_index()

# https://mlfinlab.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_clustering_asset_allocation.html
def hcaa_model(returns, expected_return_type, linkage, metric, min_weight=0, max_weight=1):
  hcaa = HierarchicalClusteringAssetAllocation(calculate_expected_returns=expected_return_type)
  hcaa.allocate(asset_returns=returns, linkage=linkage, allocation_metric=metric, min_weight=.05, max_weight=.2) # can use constraints to target vol
  return hcaa.weights.transpose().sort_index()

if __name__ == '__main__':

  tickers = ['PPLC','PPDM','PPEM','VNQ','VNQI','SGOL','PDBC','BKLN','VTIP','TYD','EDV','BWX','VWOB']
  #tickers = ['VOO','VEA','VWO','VNQ','VNQI','SGOL','PDBC','BKLN','VTIP','IEF','EDV','BWX','VWOB']
  end_date = date.today()

  multi_returns = get_returns()
  multi_prices = get_prices()
  covariances = ['pearson', distance_correlation, angular_distance, get_mutual_info]
  linkages = ['single', 'complete']
  metrics = ['minimum_variance', 'minimum_standard_deviation', 'equal_weighting', 'expected_shortfall']
  expected_return_types = ['mean', 'exponential']

  pool = mp.Pool(4)
  hrps = []

  for returns in multi_returns:
    for covariance in covariances:
      pool.apply_async(hrp_model, args=(returns, covariance, False, True), callback=lambda result: hrps.append(result))
      pool.apply_async(hrp_model, args=(returns, covariance, True, False), callback=lambda result: hrps.append(result))
    for linkage in linkages:
      for metric in metrics:
        for expected_return_type in expected_return_types:
          pool.apply_async(hcaa_model, args=(returns, expected_return_type, linkage, metric, .05, .2), callback=lambda result: hrps.append(result))

  for prices in multi_prices:
    for linkage in linkages:
      for metric in metrics:
        for expected_return_type in expected_return_types:
          pool.apply_async(hcaa_model, args=(returns, expected_return_type, linkage, 'sharpe_ratio'), callback=lambda result: hrps.append(result))

  pool.close()
  pool.join()

  soft_majority_vote_hrp = pd.concat(hrps).groupby(level=0).mean().round(3) * 100 # simple bagging ensemble
  print(soft_majority_vote_hrp)

# backtest at 
# https://www.portfoliovisualizer.com/backtest-portfolio?s=y&timePeriod=2&startYear=1985&firstMonth=1&endYear=2020&lastMonth=12&calendarAligned=true&includeYTD=false&initialAmount=10000&annualOperation=0&annualAdjustment=0&inflationAdjusted=true&annualPercentage=0.0&frequency=4&rebalanceType=4&absoluteDeviation=5.0&relativeDeviation=25.0&showYield=false&reinvestDividends=true&portfolioNames=false&portfolioName1=Portfolio+1&portfolioName2=Portfolio+2&portfolioName3=Portfolio+3&symbol1=BWX&allocation1_1=8&allocation1_2=9&symbol2=VWOB&allocation2_1=6&allocation2_2=5&symbol3=EDV&allocation3_1=5&allocation3_2=4&symbol4=TYD&allocation4_1=8&symbol5=VTIP&allocation5_1=28&allocation5_2=23&symbol6=BKLN&allocation6_1=6&allocation6_2=6&symbol7=PDBC&allocation7_1=6&allocation7_2=5&symbol8=SGOL&allocation8_1=9&allocation8_2=10&symbol9=VNQI&allocation9_1=4&allocation9_2=4&symbol10=VNQ&allocation10_1=6&allocation10_2=4&symbol11=PPEM&allocation11_1=5&symbol12=PPDM&allocation12_1=5&symbol13=PPLC&allocation13_1=4&symbol14=VWO&allocation14_2=4&symbol15=VEA&allocation15_2=4&symbol16=VOO&allocation16_2=4&symbol17=IEF&allocation17_2=18
