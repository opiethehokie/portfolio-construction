import math
import multiprocessing as mp
import random
import warnings

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from lib import get_daily_returns, get_volume_bar_returns
from lib import get_mutual_info # to remove extra parameters
from mlfinlab.codependence.correlation import distance_correlation, angular_distance
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity 
from mlfinlab.portfolio_optimization.herc import HierarchicalEqualRiskContribution
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators

# graph theory (tree instead of complete correlation graph) and machine learning (clustering) based optimization
# instability, concentration, and underperformance improvement on convex optimization (quadratic programming)

# https://blog.thinknewfound.com/2017/11/risk-parity-much-data-use-estimating-volatilities-correlations/
def get_returns():
  short_term_log = get_volume_bar_returns(tickers, date.today() + relativedelta(days=-59), date.today(), log=True)
  med_term_frac = get_daily_returns(tickers, date.today() + relativedelta(months=-24), end_date, return_type='fractional')
  long_term_frac = get_daily_returns(tickers, end_date + relativedelta(months=-60), end_date, return_type='fractional')
  return [short_term_log, med_term_frac, long_term_frac]

# https://hudsonthames.org/portfolio-optimisation-with-mlfinlab-estimation-of-risk/
def get_covariances(data, price_data=False):
  covs = []
  covs.append(RiskEstimators().minimum_covariance_determinant(data, price_data=price_data))
  covs.append(RiskEstimators().empirical_covariance(data, price_data=price_data))
  covs.append(RiskEstimators().shrinked_covariance(data, price_data=price_data, shrinkage_type='lw'))
  covs.append(RiskEstimators().semi_covariance(data, price_data=price_data))
  covs.append(RiskEstimators().exponential_covariance(data, price_data=price_data))
  covs.append(RiskEstimators.corr_to_cov(data.corr(method=distance_correlation), data.std()))
  covs.append(RiskEstimators.corr_to_cov(data.corr(method=angular_distance), data.std()))
  covs.append(RiskEstimators.corr_to_cov(data.corr(method=get_mutual_info), data.std()))
  covs.append(RiskEstimators().denoise_covariance(data.cov(), data.shape[0] / data.shape[1], denoise_method='target_shrink'))
  covs.append(RiskEstimators().denoise_covariance(data.cov(), data.shape[0] / data.shape[1], denoise_method='const_resid_eigen', detone=True))
  return covs

# https://mlfinlab.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_risk_parity.html
def hrp_model(returns, cov, linkage):
  hrp = HierarchicalRiskParity()
  hrp.allocate(asset_names=returns.columns, covariance_matrix=cov, linkage=linkage)
  return hrp.weights.transpose().sort_index()

# https://mlfinlab.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_equal_risk_contribution.html
def herc_model(returns, cov, linkage, metric):
  herc = HierarchicalEqualRiskContribution()
  herc.allocate(asset_names=returns.columns, asset_returns=returns, covariance_matrix=cov, linkage=linkage, risk_measure=metric)
  return herc.weights.transpose().sort_index()

if __name__ == '__main__':

  tickers = ['VTI','VEA','VWO','VNQ','VNQI','SGOL','PDBC','BKLN','SCHP','TYD','LEMB','FMF']
  end_date = date.today()

  multi_returns = get_returns()
  
  linkages = ['single', 'complete', 'ward']
  metrics = ['equal_weighting', 'expected_shortfall', 'conditional_drawdown_risk']

  pool = mp.Pool(4)
  allocations = []

  for returns in multi_returns:
    covs = get_covariances(returns)
    for cov in covs:
      for linkage in linkages:
        pool.apply_async(hrp_model, args=(returns, cov, linkage), callback=lambda result: allocations.append(result)) # variance-based metric
        for metric in metrics:
          pool.apply_async(herc_model, args=(returns, cov, linkage, metric), callback=lambda result: allocations.append(result))

  pool.close()
  pool.join()

  soft_majority_vote = pd.concat(allocations).groupby(level=0).mean().round(3) * 100 # simple bagging ensemble
  print(soft_majority_vote)

# backtest at https://www.portfoliovisualizer.com/backtest-portfolio?s=y&timePeriod=2&startYear=1985&firstMonth=1&endYear=2020&lastMonth=12&calendarAligned=true&includeYTD=false&initialAmount=10000&annualOperation=0&annualAdjustment=0&inflationAdjusted=true&annualPercentage=0.0&frequency=4&rebalanceType=4&absoluteDeviation=5.0&relativeDeviation=25.0&showYield=false&reinvestDividends=true&portfolioNames=false&portfolioName1=Portfolio+1&portfolioName2=Portfolio+2&portfolioName3=Portfolio+3&symbol1=BKLN&allocation1_1=7.5&symbol2=FMF&allocation2_1=15.9&symbol3=LEMB&allocation3_1=5.5&symbol4=PDBC&allocation4_1=6.1&symbol5=PPDM&allocation5_1=5.5&symbol6=PPEM&allocation6_1=3.5&symbol7=PPLC&allocation7_1=3.0&symbol8=SGOL&allocation8_1=11.7&symbol9=VTIP&allocation9_1=24.7&symbol10=TYD&allocation10_1=7.6&symbol11=VNQ&allocation11_1=3.0&symbol12=VNQI&allocation12_1=3.0&symbol13=VWO&allocation13_1=3.0
