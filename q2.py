from types import FunctionType

import pandas as pd
import talib
from statsmodels.tsa.stattools import adfuller
from talib import abstract

# !pip install TA-lib


def plot_series(data_dict: dict, title: str, xlabel: str, ylabel: str, save_path: str=None):
    plt.figure(figsize=(10, 6))
    for label, series in data_dict.items():
        plt.plot(series.index, series, label=label, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


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
        future_returns: pd.DataFrame,
        signal_values: pd.DataFrame,
        quantile_bins: int = 5):
    avg_returns_list: list = list()
    for ticker in future_returns.columns:
        if signal_values[ticker].isnull().all():
            continue
        quantiles = pd.qcut(signal_values[ticker], quantile_bins, labels=False, duplicates='drop')
        future_returns_col = future_returns[ticker]
        returns_by_quantile = future_returns_col.groupby(quantiles).mean()
        avg_returns_list.append(returns_by_quantile)
    avg_returns_df = pd.concat(avg_returns_list, axis=1).mean(axis=1)
    return avg_returns_df


# BASELINE - MULTIPLE HORIZONS
horizons_results: dict = dict()
horizon_days = (5, 10, 15, 21, 63, 92, 252)
signals = dict(mom=compute_momentum, rsi=compute_rsi, )
beta_adjustments = (True, False)
prices = pd.read_pickle('hot/prices.pickle')

# running a simulation with various parameters
# for beta_adjustment in beta_adjustments:
#     for signal_name, signal_fct in signals.items():
#         for horizon in horizon_days:
#             print(f'{str(beta_adjustment)=} {signal_name=} {horizon=}')
#             future_returns = prices.pct_change(periods=horizon).shift(-horizon)
#
#             if beta_adjustment:
#                 # results are good, but returns are consistently positive, which is in-line with general stock markets
#                 # distribution skew; an interesting approach would be to do some kind of beta-discounting,
#                 # let's try using an equal weighted index
#                 # note the index composition will naturally vary over time, which is somewhat in line with the reality
#                 beta_adjustment_values = future_returns.mean(axis=1)
#                 future_returns = future_returns.subtract(beta_adjustment_values, axis=0)
#
#             signal_values = prices.apply(signal_fct, axis=0)
#             avg_returns_df = get_signal_avg_returns(future_returns, signal_values)
#             horizons_results[horizon] = avg_returns_df
#
#         aggregated_results = pd.concat(horizons_results, axis=1)
#         aggregated_results.columns = [f'Horizon {horizon} days' for horizon in horizons_results.keys()]
#         plot_series(horizons_results, 'Average Returns Across Horizons', 'Quantile', 'Average Returns',
#             save_path=f'plots/q2/returns_{signal_name}_horizons_beta_adjusted_{beta_adjustment}.png')

# so, interestingly both indicators show consistent results, both with and without beta discounting, lower quantiles
# outperform higher quantiles, i.e. the securities presented here seem to exhibit a mean reverting behavior
# fwiw let's do a quick adf test and see what it shows


def adf_with_drift(prices, confidence=0.05, regression='ctt', maxlag=21):
    def mean_reversion_test(series):
        p_value = adfuller(series, regression=regression, maxlag=maxlag)[1]
        return p_value < confidence

    results = prices.apply(mean_reversion_test, axis=0)
    mean_reverting_ratio = results.mean() * 100
    return mean_reverting_ratio


# using dropna we have a 600+ tickers subset in the universe scope, for this test it should be representative enough
# subset = prices['2011-01-31':'2020-12-31'].dropna(axis=1, how='any')
# print(f"{adf_with_drift(subset, regression='ct'):.2f}% of tickers exhibit mean-reverting behavior with drift")
# print(f"{adf_with_drift(subset):.2f}% of tickers exhibit mean-reverting behavior with linear and quadratic trends")

# non-negligible proportion of mean-reverting securities, given that the returns distribution is skewed positively,
# it is not surprising that the lowest quantile (go long oversold stocks) outperforms the rest

# REGIME
# a possibility is also to use these in combination with a regime detection model, let's tke 3 months horizon
# usually in high vol markets, investors tend to be more coordinated and momentum is stronger
regimes_results: dict = dict()
regimes = pd.read_pickle('hot/regimes.pickle')
regimes_dict = dict(low_vol=0, high_vol=1)

# beta_adjustment = True
# for signal_name, signal_fct in signals.items():
#     future_returns = prices.pct_change(periods=63).shift(-63)
#     beta_adjustment_values = future_returns.mean(axis=1)
#     future_returns = future_returns.subtract(beta_adjustment_values, axis=0)
#     signal_values = prices.apply(signal_fct, axis=0)
#
#     for regime_name, regime_value in regimes_dict.items():
#         regime_future_returns = future_returns.loc[regimes.index[regimes['Regime'] == regime_value]]
#         regime_signal_values = signal_values.loc[regimes.index[regimes['Regime'] == regime_value]]
#         regimes_results[regime_name] = get_signal_avg_returns(regime_future_returns, regime_signal_values)
#
#     # Plotting both series
#     plt.figure(figsize=(10, 6))
#     hv, lv = regimes_results['high_vol'], regimes_results['low_vol']
#     plt.plot(hv.index, hv, label='High Volatility', color='red', alpha=0.7)
#     plt.plot(lv.index, lv, label='Low Volatility', color='black', alpha=0.7)
#     plt.title('High Volatility vs Low Volatility')
#     plt.xlabel('Quantile')
#     plt.ylabel('Value')
#     plt.legend(loc='best')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'plots/q2/returns_{signal_name}_regimes_beta_adjusted_True.png')
#     plt.show()

    # todo switch plotting code to boilerplate plot_series()

# worth noting that naturally the high vol context provides better opportunities to "buy the dip",
# and bounce back comparatively to the low vol regime, the plots display this, which makes sense


# RELATIVE
# another interesting development is to use these indicators for relative comparison within clusters
# clusters_results: dict = dict()
# clusters = pd.read_pickle('hot/clusters.pickle')
# clusters = {f'cluster_{x}': value for x, value in enumerate(clusters.values())}
#
# beta_adjustment = True
# for signal_name, signal_fct in signals.items():
#     future_returns = prices.pct_change(periods=63).shift(-63)
#     beta_adjustment_values = future_returns.mean(axis=1)
#     future_returns = future_returns.subtract(beta_adjustment_values, axis=0)
#     signal_values = prices.apply(signal_fct, axis=0)
#
#     for cluster_name, cluster in clusters.items():
#         clusters_results[cluster_name] = get_signal_avg_returns(future_returns[cluster], signal_values[cluster])
#
#     plot_series(clusters_results, 'Alpha by SOM Cluster', 'Quantile', 'Value',
#                 save_path=f'plots/q2/returns_{signal_name}_clusters_beta_adjusted_True.png')

# worth noting that some clusters react much better than others especially for the RSI signal


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
    temp['OBJECTIVE'] = temp.iloc[:, 0].pct_change(periods=63).shift(-63)
    temp[temp.columns[0]] = str(col)
    custom[col] = temp

# todo revisit once done ith q3/q4, gather all securities, train the forest once a year with a growing window
print(custom)

# todo run flake8 & mypy
