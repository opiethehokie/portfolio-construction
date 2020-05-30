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
  #short_term_percent = get_volume_bar_returns(tickers, date.today() + relativedelta(days=-59), date.today(), log=False)
  med_term_log = get_daily_returns(tickers, date.today() + relativedelta(months=-6), end_date, return_type='log')
  #med_term_percent = get_daily_returns(tickers, end_date + relativedelta(months=-6), end_date, return_type='percent')
  long_term_frac = get_daily_returns(tickers, end_date + relativedelta(months=-24), end_date, return_type='fractional')
  #long_term_percent = get_daily_returns(tickers, end_date + relativedelta(months=-24), end_date, return_type='percent')
  return [short_term_log, 
          #short_term_percent,
          med_term_log, 
          #med_term_percent,
          long_term_frac, 
          #long_term_percent
        ]

def get_prices():
  short_term = get_daily_prices(tickers, end_date + relativedelta(days=-59), end_date)
  med_term = get_daily_prices(tickers, end_date + relativedelta(months=-6), end_date)
  long_term = get_daily_prices(tickers, end_date + relativedelta(months=-24), end_date)
  return [short_term, med_term, long_term]

def get_cov(returns, cov_type):
  if cov_type == 'shrinkage':
    cov = RiskEstimators.shrinked_covariance(returns, shrinkage_type='lw')
  elif cov_type == 'exponential':
    cov = RiskEstimators.exponential_covariance(returns)
  else:
    cov = RiskEstimators.corr_to_cov(returns.corr(method=cov_type), returns.std())
  return RiskEstimators().denoise_covariance(cov, returns.shape[0] / returns.shape[1], .01)

# https://mlfinlab.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_risk_parity.html
def hrp_model(returns, cov_type):
  cov = get_cov(returns, cov_type)
  hrp = HierarchicalRiskParity()
  hrp.allocate(asset_names=returns.columns, covariance_matrix=cov)
  return hrp.weights.transpose().sort_index()

# https://mlfinlab.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_clustering_asset_allocation.html
# can use min/max weight constraints to target vol
def hcaa_model(returns, prices, expected_return_type, cov_type, linkage, metric, min_weight=.03, max_weight=.33):
  hcaa = HierarchicalClusteringAssetAllocation(calculate_expected_returns=expected_return_type)
  if prices is not None:
    hcaa.allocate(asset_names=prices.columns, asset_prices=prices, linkage=linkage, allocation_metric=metric, min_weight=min_weight, max_weight=max_weight)
  else:
    cov = get_cov(returns, cov_type)
    hcaa.allocate(asset_names=returns.columns, asset_returns=returns, covariance_matrix=cov, linkage=linkage, allocation_metric=metric, 
                  min_weight=min_weight, max_weight=max_weight)
  return hcaa.weights.transpose().sort_index()

if __name__ == '__main__':

  #tickers = ['PPLC','PPDM','PPEM','VNQ','VNQI','SGOL','PDBC','BKLN','VTIP','TYD','EDV','BWX','VWOB']
  tickers = ['VOO','VEA','VWO','VNQ','VNQI','SGOL','PDBC','BKLN','VTIP','IEF','EDV','BWX','VWOB']
  end_date = date.today()

  multi_returns = get_returns()
  multi_prices = get_prices()
  cov_types = ['shrinkage', 'exponential', distance_correlation, angular_distance, get_mutual_info] # mutual info brings in information theory and entropy
  linkages = ['single', 'complete']
  metrics = ['minimum_variance', 'minimum_standard_deviation', 'equal_weighting', 'expected_shortfall']
  expected_return_types = ['mean', 'exponential']

  pool = mp.Pool(4)
  hrps = []

  for returns in multi_returns:
    for cov_type in cov_types:
      pool.apply_async(hrp_model, args=(returns, cov_type), callback=lambda result: hrps.append(result))
      for linkage in linkages:
        for metric in metrics:
          for expected_return_type in expected_return_types:
            pool.apply_async(hcaa_model, args=(returns, None, expected_return_type, cov_type, linkage, metric), callback=lambda result: hrps.append(result))

  for prices in multi_prices:
    for linkage in linkages:
      for metric in metrics:
        for expected_return_type in expected_return_types:
          pool.apply_async(hcaa_model, args=(None, prices, expected_return_type, None, linkage, 'sharpe_ratio'), callback=lambda result: hrps.append(result))

  pool.close()
  pool.join()

  soft_majority_vote_hrp = pd.concat(hrps).groupby(level=0).mean().round(3) * 100 # simple bagging ensemble
  print(soft_majority_vote_hrp)

# backtest at 
# https://www.portfoliovisualizer.com/backtest-portfolio?s=y&timePeriod=2&startYear=1985&firstMonth=1&endYear=2020&lastMonth=12&calendarAligned=true&includeYTD=false&initialAmount=10000&annualOperation=0&annualAdjustment=0&inflationAdjusted=true&annualPercentage=0.0&frequency=4&rebalanceType=4&absoluteDeviation=5.0&relativeDeviation=25.0&showYield=false&reinvestDividends=true&portfolioNames=false&portfolioName1=Portfolio+1&portfolioName2=Portfolio+2&portfolioName3=Portfolio+3&symbol1=BWX&allocation1_2=9.4&allocation1_3=9.9&symbol2=VWOB&allocation2_2=4.9&allocation2_3=5.9&symbol3=EDV&allocation3_2=6.8&allocation3_3=6.9&symbol4=TYD&allocation4_3=7.6&symbol5=VTIP&allocation5_2=21.8&allocation5_3=24.9&symbol6=BKLN&allocation6_2=6.1&allocation6_3=6.6&symbol7=PDBC&allocation7_2=6.2&allocation7_3=7.7&symbol8=SGOL&allocation8_2=7.8&allocation8_3=9.5&symbol9=VNQI&allocation9_2=3.9&allocation9_3=4.2&symbol10=VNQ&allocation10_2=3.8&allocation10_3=4.2&symbol11=PPEM&allocation11_3=4.2&symbol12=PPDM&allocation12_3=4.6&symbol13=PPLC&allocation13_3=3.8&symbol14=VWO&allocation14_2=4.2&symbol15=VEA&allocation15_2=4&symbol16=VOO&allocation16_2=4.1&symbol17=IEF&allocation17_2=17
