import pandas as pd
import seaborn as sns
import warnings
from matplotlib import pyplot as plt
from pypfopt import plotting
from pypfopt import risk_models, EfficientFrontier

warnings.filterwarnings("ignore")
from utils import backtest_portfolio, portfolio_metrics

sns.set(style='whitegrid')
# !pip install pyportfolioopt


# returns = pd.read_pickle('hot/returns.pickle')
prices = pd.read_pickle('hot/prices.pickle')
scope = pd.read_pickle('hot/scope.pickle')

# we will take monthly points starting from 2011-01-31 as that's where the eligible universe rebal dates start
prices = prices['2010-07-01':].dropna(axis=1, how='all').bfill()
ref = prices.div(prices.iloc[0])

# however for the sake of having a proper matrix, we will use Ledoit-Wolf
# and add 6 months of historical data points for lookback purposes
extra_points = prices[(prices.index.year == 2010) & (prices.index.month > 6)]
monthly_indices = extra_points.resample('M').last().index
rebal_dates = scope.index.to_list()
history = rebal_dates + list(monthly_indices)

df = ref[ref.index.isin(history)]
print(df)

S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
plotting.plot_covariance(S, plot_correlation=True)
plt.savefig('plots/q4/ledoit_wolf_covariance_matrix.png')
# plt.show()
print(S)

refresh_weights = False
if refresh_weights:
    results_dict = {}
    for rebal_date in rebal_dates:
        print(rebal_date)
        df_subset = df[df.index <= rebal_date]
        S = risk_models.CovarianceShrinkage(df_subset).ledoit_wolf()
        ef = EfficientFrontier(None, S, weight_bounds=(None, None))
        ef.min_volatility()
        results_dict[rebal_date] = ef.clean_weights()

    results_df = pd.DataFrame.from_dict(results_dict, orient='index').sort_index()
    resampled_results_df = results_df.reindex(ref.index).ffill()
    pd.to_pickle(resampled_results_df, 'weights/opt_weights.pickle')
    print(resampled_results_df)

returns = pd.read_pickle('hot/returns.pickle')
scope = pd.read_pickle('hot/scope.pickle')

equity_curves: dict = dict()
running_sharpe_ratios: dict = dict()

weights: dict = dict(
    equal_weights=pd.read_pickle('weights/equal_weights.pickle'),
    opt_weights=pd.read_pickle('weights/opt_weights.pickle')
)

for approach_name, weights_frame in weights.items():
    start = '2011-01-31'
    backtest = backtest_portfolio(df_returns=returns[start:], df_weights=weights_frame[start:], rebal_cost=0)
    equity_curves[approach_name] = backtest['equity']
    running_sharpe_ratios[approach_name] = portfolio_metrics(backtest['equity'])
    # adding 1 year to collect proper running sharpe ratio metrics
    running_sharpe_ratios[approach_name] = running_sharpe_ratios[approach_name]['2012-01-31':]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
palette = sns.color_palette('muted', n_colors=len(equity_curves))
for idx, (approach_name, equity_curve) in enumerate(equity_curves.items()):
    sns.lineplot(data=equity_curve, label=approach_name, color=palette[idx], ax=ax1)

ax1.set_title('Equity Curves Comparison', fontsize=16)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Equity', fontsize=12)
ax1.legend(title='Approach', fontsize=10)

for idx, (approach_name, sharpe_ratio) in enumerate(running_sharpe_ratios.items()):
    sns.lineplot(data=sharpe_ratio, label=approach_name, color=palette[idx], ax=ax2)

ax2.set_title('Rolling Sharpe Ratio', fontsize=16)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Sharpe Ratio', fontsize=12)
ax2.legend(title='Approach', fontsize=10)

plt.tight_layout()
plt.savefig('plots/q4/equal-weights_vs_optimized_comparison.png')
plt.show()


# conclusion equal weights has a less returns but a higher Sharpe ratio than the naive portfolio,
# which is somewhat expected ;)

# thank you for reading me and thank you for proposing this exercise, I found it interesting
# Jonathan
