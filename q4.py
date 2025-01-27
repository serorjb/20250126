import pandas as pd
from matplotlib import pyplot as plt
from pypfopt import risk_models
from pypfopt import plotting
# !pip install pyportfolioopt

# returns = pd.read_pickle('hot/returns.pickle')
prices = pd.read_pickle('hot/prices.pickle')

# we start from 2011-01-31 as that's where the eligible universe rebal dates start
temp = prices['2011-01-31':].dropna(axis=1, how='all').bfill()
temp = temp.div(temp.iloc[0])


S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
plotting.plot_covariance(S, plot_correlation=True)
# plt.show()  # no point plotting, can't see much as there are too many assets
print(S)


S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

# You don't have to provide expected returns in this case
ef = EfficientFrontier(None, S, weight_bounds=(None, None))
ef.min_volatility()
weights = ef.clean_weights()
weights
