from types import FunctionType

import pandas as pd
import talib
from statsmodels.tsa.stattools import adfuller
from talib import abstract

# !pip install TA-lib

prices = pd.read_pickle('hot/prices.pickle')
clusters = pd.read_pickle('hot/clusters.pickle')


# Use returns calculated in Q1 to construct at least 2 price momentum signals. You may assume that closing
# price for MARKET_DATE=2016-06-30 is available for signal construction on DATE=2016-06-30.

# Explain if you would favor larger/smaller values of the signal constructed and explain why assets with
# larger (smaller) signal values should outperform those with smaller (larger) values.

# RSI / MOMENTUM
def compute_rsi(series: pd.Series, period: int = 14):
    # c.f. https://en.wikipedia.org/wiki/relative_strength_index
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill()


def compute_momentum(series: pd.Series, period: int = 21):
    return (series / series.shift(period) - 1).bfill()


# rsi = prices.apply(compute_rsi, axis=0)
# momentum = prices.apply(compute_momentum, axis=0)

# the RSI and Momentum are classic momentum-type indicators;
# RSI is traditionally used with a 14 period and Momentum 10.
# for our study here, it is sensible to use 21 business days (i.e. a month) at a minimum
# given that is the fastest rebal frequency allowed by the universe/scope

# the theory is that stocks that outperform (usually within their peer group) will continue to do so,
# and vice-versa for underperformers, hence the idea is to identify them and bet on the favourites
# separately the RSI goal is more to identify overbought/oversold trends that tend to retrace,
# after excessive movement in either direction, usually a cross downwards from 70+ or upwards from 30-


import pandas as pd
import matplotlib.pyplot as plt


def get_signal_avg_returns(
        signal_fct: FunctionType,
        horizon_days: int,
        prices: pd.DataFrame = None,
        returns: pd.DataFrame = None,
        quantile_bins: int = 5,
        beta_adjust: bool = False):
    if returns is None:
        if prices is None:
            raise
        future_returns: pd.DataFrame = prices.pct_change(periods=horizon_days).shift(-horizon_days)
    else:
        future_returns: pd.DataFrame = returns
    if beta_adjust:
        # good, but returns are consistently positive, which is in-line with general stock markets distribution skew
        # an interesting approach would be to do some kind of beta-discounting, let's try using an equal weighted index
        beta_adjustment = prices.mean(axis=1).pct_change(periods=horizon).shift(-horizon)
        future_returns = future_returns.subtract(beta_adjustment, axis=0)
    signal_values: pd.DataFrame = prices.apply(signal_fct, axis=0)
    avg_returns_list: list = list()
    for ticker in prices.columns:
        if signal_values[ticker].isnull().all():
            continue
        quantiles = pd.qcut(signal_values[ticker], quantile_bins, labels=False, duplicates='drop')
        future_returns_col = future_returns[ticker]
        returns_by_quartile = future_returns_col.groupby(quantiles).mean()
        avg_returns_list.append(returns_by_quartile)
    avg_returns_df = pd.concat(avg_returns_list, axis=1).mean(axis=1)
    return avg_returns_df


# time horizons
horizons_results: dict = dict()
horizon_days = (5, 10, 15, 21, 63, 92, 252)
signals = dict(mom=compute_momentum, rsi=compute_rsi, )
beta_adjust = (True, False)


# running a simulation with various parameters
# for beta_adjustment in beta_adjust:
#     for signal_name, signal_fct in signals.items():
#         for horizon in horizon_days:
#             print(f'', horizon)
#             avg_returns_df = get_signal_avg_returns(prices, signal_fct, horizon_days=horizon, beta_adjust=beta_adjustment)
#             horizons_results[horizon] = avg_returns_df
#
#         aggregated_results = pd.concat(horizons_results, axis=1)
#         aggregated_results.columns = [f'Horizon {horizon} days' for horizon in horizons_results.keys()]
#         plt.figure(figsize=(12, 6))
#         for column in aggregated_results.columns:
#             plt.plot(aggregated_results.index, aggregated_results[column], label=column)
#         plt.title("Average Returns Across Horizons")
#         plt.xlabel("Index")
#         plt.ylabel("Average Returns")
#         plt.legend(title="Horizons")
#         plt.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(f'plots/q2/returns_{signal_name}_horizons_beta_adjusted_{beta_adjustment}.png')
#         plt.show()


# so, interestingly both indicators show consistent results, both with and without beta discounting, lower quantiles
# outperform higher quantiles, i.e. the securities presented here seem to exhibit a mean reverting behavior
# fwiw let's do a quick adf test and see what it shows


def adf_with_drift(prices, confidence=0.05, regression='ctt', maxlag=21):
    def mean_reversion_test(series):
        result = adfuller(series, regression=regression, maxlag=maxlag)
        p_value = result[1]
        return p_value < confidence

    results = prices.apply(mean_reversion_test, axis=0)
    mean_reverting_ratio = results.mean() * 100
    return mean_reverting_ratio


# let's use a 200 tickers subset in the universe scope for this test which should be representative enough
# subset = prices['2011-01-31':'2020-12-31'].dropna(axis=1, how='any').sample(n=200, axis=1, random_state=int())
# print(f"{adf_with_drift(subset, regression='ct'):.2f}% of tickers exhibit mean-reverting behavior with drift")
# print(f"{adf_with_drift(subset):.2f}% of tickers exhibit mean-reverting behavior with linear and quadratic trends")

# non-negligible proportion of mean-reverting securities, given that the returns distribution is skewed positively,
# it is not surprising that the lowest quantile (go long oversold stocks) outperforms the rest

# rsi by regime with horizon 3 months
regimes_results: dict = dict()
regimes = pd.read_pickle('hot/regimes.pickle')

prices_low_vol = prices.loc[regimes.index[regimes['Regime'] == 0]]
prices_high_vol = prices.loc[regimes.index[regimes['Regime'] == 1]]
signals = dict(mom=compute_momentum, rsi=compute_rsi, )
regimes = dict(low_vol=prices_low_vol, high_vol=prices_high_vol)

beta_adjustment = True
for signal_name, signal_fct in signals.items():
    for regime_name, regime_prices in regimes.items():
        avg_returns_df = get_signal_avg_returns(signal_fct, 63, prices=regime_prices, beta_adjust=True)
        print(avg_returns_df)

aggregated_results = pd.concat(horizons_results, axis=1)
aggregated_results.columns = [f'Horizon {horizon} days' for horizon in horizons_results.keys()]
plt.figure(figsize=(12, 6))
for column in aggregated_results.columns:
    plt.plot(aggregated_results.index, aggregated_results[column], label=column)
plt.title("Average Returns Across Horizons")
plt.xlabel("Index")
plt.ylabel("Average Returns")
plt.legend(title="Horizons")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'plots/q2/returns_{signal_name}_horizons_beta_adjusted_{beta_adjustment}.png')
plt.show()

# by cluster with horizon 3 months todo
clusterss_results: dict = dict()

# Plot the results (raw and beta-adjusted)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot raw returns
avg_returns.plot(ax=axes[0])
axes[0].set_title('Average Future Returns (Raw)')
axes[0].set_xlabel('Quartile')
axes[0].set_ylabel('Average Return')

# Plot beta-adjusted returns
# Compute beta-adjusted returns by subtracting index average returns (use mean of all columns)
index_returns = prices.mean(axis=1).pct_change().shift(-1)
beta_adjusted_returns = avg_returns.sub(index_returns, axis=0)

beta_adjusted_returns.plot(ax=axes[1])
axes[1].set_title('Average Future Returns (Beta-Adjusted)')
axes[1].set_xlabel('Quartile')
axes[1].set_ylabel('Beta-Adjusted Return')

plt.tight_layout()
plt.show()


# RELATIVE
# another interesting development is to use these indicators for relative comparison within clusters


# REGIME todo
# a possibility is also to use these in combination with a regime detection model,
# usually in high vol markets, investors tend to be more coordinated and momentum is stronger


# CUSTOM
# thinking blue-sky for a minute here, another example of approach we could take is to generate
# many technical indicators and feed them as features to a random forest, and see what's relevant;
# we don't have OHLC dataset but given we have daily close and monthly rebal, we can do a rolling resample

def talib_base_indicators(df: pd.DataFrame, inputs: dict):
    # skipping 'Volume Indicators' as we don't have the volume here
    scope = {'Momentum Indicators', 'Cycle Indicators', 'Volatility Indicators', 'Price Transform'}
    for f, v in abstract.__dict__.items():
        if type(v) == talib._ta_lib.Function:
            if v.info.get('group') in scope:
                name = v.info.get('name')
                output = v(inputs)
                if type(output) == list:
                    for x in range(len(output)):
                        df[f'{name}_x'] = output[x]
                else:
                    df[name] = output


custom: dict = dict()
for col in prices.columns:
    temp = prices[[col]].copy(deep=True)
    talib_base_indicators(temp, inputs=dict(
        # assuming 5 business days per week
        open=temp[col].shift(5),
        high=temp[col].rolling(5).max(),
        low=temp[col].rolling(5).min(),
        close=temp[col],
        # feeding 0 volume for compatibility
        # features will go into a random forest which will ignore irrelevant inputs
        volume=pd.Series(0, index=temp[col].index),
    ))
    custom[col] = temp

# col = prices.columns[0]
# custom = talib_base_indicators(prices[[col]], inputs=dict(
#         open=prices[col].shift(5),
#         high=prices[col].rolling(5).max(),
#         low=prices[col].rolling(5).min(),
#         close=prices[col],
#         # feeding 0 volume for compatibility
#         # features will go into a random forest which will ignore irrelevant inputs
#         volume=pd.Series(0, index=prices[col].index),
# ))

print(custom)

# todo once done run flake8 & mypy
