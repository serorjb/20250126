import pandas as pd

prices = pd.read_pickle('hot/prices.pickle')
clusters = pd.read_pickle('hot/clusters.pickle')

# Use returns calculated in Q1 to construct at least 2 price momentum signals. You may assume that closing
# price for MARKET_DATE=2016-06-30 is available for signal construction on DATE=2016-06-30. You are
# free to use your own methodology but if you need a reference, you may use the following formulation for
# asset 𝑖 and pick a suitable 𝑎, 𝑏 lookback window.

# Explain if you would favor larger/smaller values of the signal constructed and explain why assets with
# larger (smaller) signal values should outperform those with smaller (larger) values.

# ok let's try to do momentum and RSI and then momentum by cluster and see how it goes





