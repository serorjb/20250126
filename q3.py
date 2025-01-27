import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from matplotlib import pyplot as plt

from utils import compute_rsi, compute_momentum

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


def portfolio_metrics(equity_curve: pd.Series, risk_free_rate: float = float()) -> pd.Series:
    # note we assume daily data with 252 business days per year to compute annualized metrics
    daily_returns = equity_curve.pct_change().dropna()
    high_water_mark = equity_curve.cummax()
    drawdowns = (equity_curve - high_water_mark) / high_water_mark
    max_drawdown = drawdowns.min() * 100
    max_drawdown_duration = (drawdowns < 0).astype(int).groupby((drawdowns == 0).cumsum()).cumsum().max()

    excess_returns = daily_returns - risk_free_rate / 252
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.sqrt(252) * (excess_returns.mean() / downside_returns.std())
    annualized_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1
    calmar_ratio = annualized_return / abs(max_drawdown / 100)

    running_sharpe = daily_returns.expanding().apply(lambda x: np.sqrt(252) * x.mean() / x.std())
    daily_returns = equity_curve.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1

    print(f'Max Drawdown: {max_drawdown:.2f}%')
    print(f'Max Drawdown Duration: {max_drawdown_duration} days')
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    print(f'Sortino Ratio: {sortino_ratio:.2f}')
    print(f'Calmar Ratio: {calmar_ratio:.2f}')
    print(f'Volatility: {volatility:.2f}%')
    print(f'CAGR: {cagr:.2%}')

    return running_sharpe


def backtest_portfolio(df_returns: pd.DataFrame, df_weights: pd.DataFrame, rebal_cost: float = float()):
    # the below implementation does daily rebal with elements in scope for this month,
    # one extension could be to use a threshold-based rebal frequency rather than time-based
    symbols = df_returns.columns
    df_backtest = pd.concat([df_returns.add_suffix('_RETURN'), df_weights.add_suffix('_WEIGHT')], axis=1)

    for symbol in symbols:
        df_backtest[symbol] = df_backtest[f'{symbol}_RETURN'] * df_backtest[f'{symbol}_WEIGHT']*100
        if rebal_cost != 0.0:
            df_backtest[symbol] = df_backtest[symbol].mask(
                df_backtest[f'{symbol}_WEIGHT'] != df_backtest[f'{symbol}_WEIGHT'].shift(),
                df_backtest[symbol] - rebal_cost)  # discounts rebal costs every time we change weights

    df_backtest = df_backtest[symbols]
    df_backtest['equity'] = (df_backtest + 1).mean(axis=1).cumprod(axis=0)

    df_backtest['high_water_mark'] = df_backtest['equity'].cummax()
    df_backtest['drawdown'] = (df_backtest['equity'] - df_backtest['high_water_mark']) / df_backtest[
        'high_water_mark']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    # Plot equity curve and high-water mark
    axes[0].plot(df_backtest['equity'], label='Equity Curve', linewidth=2)
    axes[0].plot(df_backtest['high_water_mark'], label='High Water Mark', linestyle='--', color='red')
    axes[0].set_title('Equity Curve and High Water Mark')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Equity')
    axes[0].legend()
    axes[0].grid(True)

    # Plot drawdown
    axes[1].plot(df_backtest['drawdown'], label='Running Drawdown', color='blue')
    axes[1].fill_between(df_backtest.index, df_backtest['drawdown'], color='blue', alpha=0.3)
    axes[1].set_title('Running Drawdown')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Drawdown')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return df_backtest[['equity', 'high_water_mark', 'drawdown']]


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
signal_values = returns.apply(lambda x: -(compute_rsi(x, period=14)-50)/100)
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
