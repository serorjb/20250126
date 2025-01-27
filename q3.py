import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from q2 import compute_rsi, compute_momentum

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


returns = pd.read_pickle('hot/returns.pickle')
equal_weighted_portfolio = pd.DataFrame(1/len(returns.columns), index=returns.index, columns=returns.columns)
# clip_returns(df_returns)  # cap outliers


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


def signal2weights(returns: pd.DataFrame, period: int = 14, upper_threshold: int = 65, lower_threshold: int = 35) -> pd.DataFrame:
    """
    Generate a momentum-based signal with weights for stocks based on their RSI.

    Args:
        df_returns (pd.DataFrame): DataFrame of returns with columns as stocks and rows as time.
        rsi_period (int): The period over which to compute the RSI (default 14).
        upper_threshold (int): The RSI value above which to underweight a stock (default 80).
        lower_threshold (int): The RSI value below which to overweight a stock (default 20).

    Returns:
        pd.DataFrame: DataFrame of weights with the same index as df_returns.
    """
    # Initialize weights with equal allocation
    n_stocks = df_returns.shape[1]
    equal_weight = 1 / n_stocks
    weights = pd.DataFrame(equal_weight, index=df_returns.index, columns=df_returns.columns)


    rsi_values = df_returns.apply(lambda x: compute_rsi(x, period))

    # Initialize previous RSI values to track threshold crossings
    prev_rsi_values = rsi_values.shift(1)

    # Store the current weights to track updates
    current_weights = pd.DataFrame(equal_weight, index=df_returns.index, columns=df_returns.columns)

    # Loop through each row (date) and apply the logic
    for date in rsi_values.index:
        # Get the RSI values for the current date and previous date
        rsi_today = rsi_values.loc[date]
        rsi_prev = prev_rsi_values.loc[date]

        # Identify stocks that crossed above 80 or below 20
        overweight_stocks = rsi_today[(rsi_prev < upper_threshold) & (rsi_today > upper_threshold)].index
        underweight_stocks = rsi_today[(rsi_prev > lower_threshold) & (rsi_today < lower_threshold)].index

        # Adjust weights when a crossing happens
        if len(overweight_stocks) > 0 or len(underweight_stocks) > 0:
            # Assign proportional weights
            overweight_weight = 0.3 / len(overweight_stocks) if len(overweight_stocks) > 0 else 0
            underweight_weight = 0.01 / len(underweight_stocks) if len(underweight_stocks) > 0 else 0
            neutral_weight = (1 - (overweight_weight * len(overweight_stocks) + underweight_weight * len(underweight_stocks))) / (n_stocks - len(overweight_stocks) - len(underweight_stocks))

            # Update current weights for the current date
            current_weights.loc[date:, :] = neutral_weight
            current_weights.loc[date:, overweight_stocks] = overweight_weight
            current_weights.loc[date:, underweight_stocks] = underweight_weight

        # Set the new weights for the current date in the weights DataFrame
        weights.loc[date:, :] = current_weights.loc[date:, :]

    return weights