import pandas as pd
from matplotlib import pyplot as plt
from pandas import Timestamp
from pypfopt import risk_models, EfficientFrontier
from pypfopt import plotting
# !pip install pyportfolioopt


# returns = pd.read_pickle('hot/returns.pickle')
prices = pd.read_pickle('hot/prices.pickle')
scope = pd.read_pickle('hot/scope.pickle')

# we will take monthly points starting from 2011-01-31 as that's where the eligible universe rebal dates start
df = prices['2010-07-01':].dropna(axis=1, how='all').bfill()
df = df.div(df.iloc[0])

# however for the sake of having a proper matrix, we will use Ledoit-Wolf
# and add 6 months of historical data points for lookback purposes
extra_points = prices[(prices.index.year == 2010) & (prices.index.month > 6)]
monthly_indices = extra_points.resample('M').last().index
rebal_dates = scope.index.to_list()
history = rebal_dates + list(monthly_indices)

df = df[df.index.isin(history)]
print(df)


S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
plotting.plot_covariance(S, plot_correlation=True)
plt.savefig('plots/q4/ledoit_wolf_covariance_matrix.png')
# plt.show()
print(S)


S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
ef = EfficientFrontier(None, S, weight_bounds=(None, None))
ef.min_volatility()
weights = ef.clean_weights()
print(weights)
