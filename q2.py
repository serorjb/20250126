import pandas as pd
import talib
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


rsi = prices.apply(compute_rsi, axis=0)
momentum = prices.apply(compute_momentum, axis=0)

# the RSI and Momentum are classic momentum-type indicators;
# RSI is traditionally used with a 14 period and Momentum 10.
# for our study here, it is sensible to use 21 business days (i.e. a month) at a minimum
# given that is the fastest rebal frequency allowed by the universe/scope

# the theory is that stocks that outperform (usually within their peer group) will continue to do so,
# and vice-versa for underperformers, hence the idea is to identify them and bet on the favourites
# separately the RSI goal is more to identify overbought/oversold trends that tend to retrace,
# after excessive movement in either direction, usually a cross downwards from 70+ or upwards from 30-




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
