import numpy as np
import pandas as pd


# note, given we have a $1 portfolio to start with, I will take the simplifying assumption
# that fractional shares are allowed, which I believe is the spirit of the exercise
# if not, we could use methods like DiscreteAllocation in pypfopt which handle this well


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
