# portfolio-optimizations

ETF portfolio modeling workspace. Inspired by Advances in Financial Machine Learning book.

## TODO

- sequential bagging on corr matrix features to classify regimes (target is unsupervised from (3?) clusters)
- purged K-fold for classification CV
- MDI/MDA feature importance

- backtesting with stats (start with Kelly because it's faster)

- can classified regime feed back into leverage or assets? re-backtest and don't forget 3 rules
- HRP vol targeting (blending RPAR/UPAR) - target can depend on relationship between predicted vol and future returns (read https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3995529), backtest
