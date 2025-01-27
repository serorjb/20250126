import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from types import FunctionType

# note, given we have a $1 portfolio to start with, I will take the simplifying assumption
# that fractional shares are allowed, which I believe is the spirit of the exercise
# if not, we could use methods like DiscreteAllocation in pypfopt which handle this well


# todo use eligible universe / scope


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



def backtest_portfolio(
        df_returns: pd.DataFrame,
        df_weights: pd.DataFrame,
        rebal_cost: float = 5*0.01*0.01):
    # 5bps rebal cost assumed
    symbols = df_returns.columns
    df_backtest = pd.concat([df_returns.add_suffix('_RETURN'), df_weights.add_suffix('_WEIGHT')], axis=1)
    # print(df_backtest)
    for symbol in symbols:
        df_backtest[symbol] = df_backtest[f'{symbol}_RETURN'] * df_backtest[f'{symbol}_WEIGHT']
        df_backtest[symbol] = df_backtest[symbol].mask(
            df_backtest[f'{symbol}_WEIGHT'] != df_backtest[f'{symbol}_WEIGHT'].shift(),
            df_backtest[symbol] - rebal_cost)  # discounts rebal costs every time we change weights

    df_backtest = df_backtest[symbols]
    df_backtest['portfolio'] = (df_backtest + 1).mean(axis=1).cumprod(axis=0)
    df_backtest['high_water_mark'] = df_backtest['portfolio'].cummax()
    df_backtest['drawdown'] = (df_backtest['portfolio'] - df_backtest['high_water_mark']) / df_backtest['high_water_mark']

    # df_backtest[['portfolio', 'drawdown']].plot()
    # plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    # Plot equity curve and high-water mark
    axes[0].plot(df_backtest['portfolio'], label='Equity Curve', linewidth=2)
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

    return df_backtest[['portfolio', 'high_water_mark', 'drawdown']]


# todo equal_weights = returns.notna().div(returns.notna().sum(axis=1), axis=0)

def signal2weights(signal_values: pd.DataFrame, long_bias: float = 0.6) -> pd.DataFrame:
    upper_thresholds = signal_values.quantile(0.85, axis=1)
    lower_thresholds = signal_values.quantile(0.15, axis=1)
    strong_signal_mask = signal_values.ge(upper_thresholds, axis=0)
    weak_signal_mask = signal_values.le(lower_thresholds, axis=0)

    # scale values linearly, so that we allocate proportionally to the signal strength in the upper quantile
    strong = signal_values.where(strong_signal_mask)
    strong = strong.sub(signal_values.where(strong_signal_mask).min(axis=1), axis=0)
    strong = strong.div(signal_values.where(strong_signal_mask).max(axis=1) - signal_values.where(strong_signal_mask).min(axis=1),axis=0)
    strong_signal_weights = strong.div(strong.sum(axis=1), axis=0).fillna(0)

    weak = signal_values.where(weak_signal_mask)
    weak = weak.sub(signal_values.where(weak_signal_mask).max(axis=1), axis=0)
    weak = weak.div(signal_values.where(weak_signal_mask).min(axis=1) - signal_values.where(weak_signal_mask).max(axis=1), axis=0)
    weak_signal_weights = weak.div(weak.sum(axis=1), axis=0).fillna(0)

    remaining_short_weight = 1 - long_bias
    strong_signal_weights *= long_bias
    weak_signal_weights *= remaining_short_weight

    weights = strong_signal_weights.add(weak_signal_weights, fill_value=0)
    return weights


