import multiprocessing as mp

from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd

from lib import get_time_interval_returns, get_volume_bar_returns, bootstrap_returns, robust_covariances, print_stats

from mlfinlab.portfolio_optimization.herc import HierarchicalEqualRiskContribution

# graph theory (tree instead of complete correlation graph) and machine learning (clustering) based optimization that's an
# instability, concentration, and underperformance improvement on convex optimization (quadratic programming)

def get_returns(end_date=date.today()):
  return [
    get_volume_bar_returns(tickers, end_date + relativedelta(days=-59), end_date),
    get_time_interval_returns(tickers, end_date + relativedelta(months=-24), end_date, return_type='fractional', interval='1d'),
    get_time_interval_returns(tickers, end_date + relativedelta(months=-36), end_date, interval='1wk'),
    get_time_interval_returns(tickers, end_date + relativedelta(months=-60), end_date, interval='1mo')
  ]

# https://mlfinlab.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_equal_risk_contribution.html
def herc_model(returns, cov, linkage, metric):
  herc = HierarchicalEqualRiskContribution()
  herc.allocate(asset_names=returns.columns, asset_returns=returns, covariance_matrix=cov, linkage=linkage, risk_measure=metric)
  return herc.weights.transpose().sort_index()

if __name__ == '__main__':

  tickers = ['VTI','VEA','VWO','VNQ','VNQI','SGOL','PDBC','SCHP','TYD','LEMB','FMF','GBTC']

  multi_returns = get_returns()
  linkages = ['single', 'complete', 'ward']
  metrics = ['variance', 'standard_deviation', 'equal_weighting', 'expected_shortfall', 'conditional_drawdown_risk'] # variance is like original HRP
  pool = mp.Pool(4)
  allocations = []

  for _ in range(1): # mini monte carlo to help randomness stabalize
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

  returns = get_time_interval_returns(tickers, date.today() + relativedelta(days=-60), date.today(), return_type='log')
  weights = soft_majority_vote[0].values / 100
  print_stats(returns, weights, 252)
