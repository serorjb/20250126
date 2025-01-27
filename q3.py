import os.path
import warnings

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import compute_rsi, compute_momentum, backtest_portfolio, portfolio_metrics

sns.set(style='whitegrid')
warnings.filterwarnings("ignore")


# note, given we have a $1 portfolio to start with, I will take the simplifying assumption
# that fractional shares are allowed, which I believe is the spirit of the exercise
# if not, we could use methods like DiscreteAllocation in pypfopt which handle this well


def portfolio_turnover(df_weights: pd.DataFrame, freq: str = 'YE') -> None:
    weight_changes = df_weights.diff().abs()
    yearly_turnover = weight_changes.resample(freq).sum().mean(axis=1)
    avg_yearly_turnover = yearly_turnover.mean()
    print(f"Average Yearly Turnover: {avg_yearly_turnover:.2%}")


def signal2weights(signal_values: pd.DataFrame, scope: pd.DataFrame, long_bias: float = 0.6,
                   equal_weights=False) -> pd.DataFrame:
    """
    NOTE: making the assumption that stocks allocated weight have to be eligible in the universe (csv files provided)
    checking eligibility on every day before quantile computation and weight assignments
    it is slightly inefficient, one could simplify by resampling monthly

    if I had a bit more time I would think about how to refactor this method
    """

    median_signal = signal_values.median(axis=1)
    temp = pd.concat([signal_values, scope], axis=1)
    temp['PXID'] = temp['PXID'].ffill()
    cols = [col for col in temp.columns if col != 'PXID']
    for col in cols:
        mask = temp['PXID'].astype(str).str.contains(str(col))
        temp.loc[~mask, col] = median_signal
    signal = temp.drop(columns=['PXID'])

    if equal_weights:
        cols = [col for col in temp.columns if col != 'PXID']
        masks = {col: temp['PXID'].astype(str).str.contains(str(col)) for col in cols}
        mask_df = pd.DataFrame(masks, index=temp.index)
        n_stocks = mask_df.sum(axis=1)
        equal_weights = mask_df.div(n_stocks, axis=0)
        return equal_weights

    # Calculate quantiles and mask for strong/weak signals
    upper_thresholds = signal.quantile(0.85, axis=1)
    lower_thresholds = signal.quantile(0.15, axis=1)
    strong_signal_mask = signal.ge(upper_thresholds, axis=0)
    weak_signal_mask = signal.le(lower_thresholds, axis=0)

    # Normalize the strong and weak signals to allocate proportionally
    strong_signal_weights = _normalize_signal_weights(signal, strong_signal_mask)
    weak_signal_weights = _normalize_signal_weights(signal, weak_signal_mask)

    # Apply long and short biases
    strong_signal_weights *= long_bias
    weak_signal_weights *= (1 - long_bias)

    return strong_signal_weights.add(weak_signal_weights, fill_value=0)


def _normalize_signal_weights(signal: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    normalized = signal.where(mask)
    normalized = (normalized.sub(normalized.min(axis=1), axis=0)
                  .div(normalized.max(axis=1) - normalized.min(axis=1), axis=0))
    return normalized.div(normalized.sum(axis=1), axis=0).fillna(0)


returns = pd.read_pickle('hot/returns.pickle')
scope = pd.read_pickle('hot/scope.pickle')

refresh = False  # note this takes like 5 minutes to run ,hence we pickle the results

weights: dict = dict()
signal_values = returns.apply(lambda x: -(compute_rsi(x, period=14) - 50) / 100)
# so here we normalise the RSI signal between -1 and 1, and we take the opposite
# i.e. aim for mean reversion, sell when RSI is high (overbought), and buy when RSI is low (oversold)
# we will add a long bias given that the returns for the securities given are skewed positively

# equal weighted
file = 'weights/equal_weights.pickle'
if refresh or not os.path.isfile(file):
    weights['equal_weights'] = signal2weights(signal_values=signal_values, scope=scope, equal_weights=True)
    pd.to_pickle(weights['equal_weights'], file)
else:
    weights['equal_weights'] = pd.read_pickle(file)

# rsi
file = 'weights/rsi_weights.pickle'
if refresh or not os.path.isfile(file):
    weights['rsi_weights'] = signal2weights(signal_values=signal_values, scope=scope)
    pd.to_pickle(weights['rsi_weights'], file)
else:
    weights['rsi_weights'] = pd.read_pickle(file)

# momentum
signal_values = returns.apply(lambda x: -compute_momentum(x, period=21))
file = 'weights/momentum_weights.pickle'
if refresh or not os.path.isfile(file):
    weights['momentum_weights'] = signal2weights(signal_values=signal_values, scope=scope)
    pd.to_pickle(weights['momentum_weights'], file)
else:
    weights['momentum_weights'] = pd.read_pickle(file)

equity_curves: dict = dict()
running_sharpe_ratios: dict = dict()
for approach_name, weights_frame in weights.items():

    if approach_name == 'rsi_weights':
        continue  # fixme something is odd with the rsi results, skip for now

    start = '2011-01-31'
    backtest = backtest_portfolio(df_returns=returns[start:], df_weights=weights_frame[start:], rebal_cost=0)
    equity_curves[approach_name] = backtest['equity']
    running_sharpe_ratios[approach_name] = portfolio_metrics(backtest['equity'])
    # adding 1 year to collect proper running sharpe ratio metrics
    running_sharpe_ratios[approach_name] = running_sharpe_ratios[approach_name]['2012-01-31':]
    portfolio_turnover(weights_frame[start:])

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
plt.savefig('plots/q3/equity_and_sharpe_comparison.png')
plt.show()
