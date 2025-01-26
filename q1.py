import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import RobustScaler


# !pip install scikit-learn minisom matplotlib numpy pandas os

# Use closing price from price.csv to construct returns for each ID in the asset list. Plot the average daily
# returns of assets in the asset list and/or any other useful exploratory data analysis. You may assume that
# the assets does not change between DATEs


# I DATA CLEANING
def get_frames(root: str = 'data'):
    frames: dict = dict()
    for file in os.listdir(root):
        if file.endswith('.csv'):
            print(file[:-4])
            frames[file[:-4]] = pd.read_csv(f'{root}/{file}')
    return frames


frames = get_frames()

fx = frames['fx']
prices = frames['price']
prices.set_index(pd.to_datetime(prices['MARKET_DATE']), inplace=True)
fx.set_index(pd.to_datetime(fx['MARKET_DATE']), inplace=True)

# FX data is monthly with missing bits early in the series and then daily, it is somewhat sufficient for our scope;
# there are 3 days (2007-05-28 / 2015-05-25 / 2020-08-05) where we have a CAD price but the rate is nan,
# I will opt to interpolate rather than ffill for these 3 days
fx = fx[prices.index.min():]
all_dates = prices.index.union(fx.index)
fx = fx.reindex(all_dates)
fx['MARKET_DATE'] = fx.index
fx.drop_duplicates(inplace=True)
fx['TICKER'].fillna('CAD', inplace=True)
fx['RATE'] = fx['RATE'].interpolate(method='linear')

# I wanted to skip the FX conversion part, there is only 1 foreign (CAD) denominated stock (PXID 66309 / ID 17434);
# we have prices back from 2005, so I can't exclude it on the grounds of insufficient lookback,
# and it is in the univ scope in 2016-06-30 and 2016-07-29 rebal periods, so technically eligible.

# we take the simplified approach below as we only have CAD to deal with
prices.update(({'RATE': fx['RATE']}))
prices['CAD'] = fx['RATE']
prices['CLOSE'] = np.where(prices['ISOCURR'] == 'USD', prices['CLOSE'], prices['CLOSE'] * prices['CAD'])
prices = prices[['PXID', 'CLOSE']]
prices = prices.pivot(columns='PXID', values='CLOSE')
# some of these rows are like 99%+ empty, so I will just drop them
prices.dropna(thresh=int(prices.shape[1] / 100), inplace=True)
# I could use parquet but it doesn't support int column names
pd.to_pickle(prices, 'hot/prices.pickle')
print(f'{prices=}')

# now converting the ID map to a PXID map defining the eligible scope
univ = pd.read_csv('data/univ.csv', parse_dates=['DATE'])
univ_map = pd.read_csv('data/univ_map.csv', parse_dates=['EFFFROM', 'EFFTHRU'])
ven_map = pd.read_csv('data/ven_map.csv', parse_dates=['EFFFROM', 'EFFTHRU'])
univ['DATE'] = pd.to_datetime(univ['DATE'], errors='coerce')
univ_map['EFFFROM'] = pd.to_datetime(univ_map['EFFFROM'])
ven_map['EFFFROM'] = pd.to_datetime(ven_map['EFFFROM'])

# coerce and fillna below because pandas can't handle 2900 dates, that workaround is sufficient for our purpose here
univ_map['EFFTHRU'] = pd.to_datetime(univ_map['EFFTHRU'], errors='coerce').fillna(pd.Timestamp('2262-04-11'))
ven_map['EFFTHRU'] = pd.to_datetime(ven_map['EFFTHRU'], errors='coerce').fillna(pd.Timestamp('2262-04-11'))

univ_map_filtered = univ.merge(univ_map, on='ID', how='inner')
univ_map_filtered = univ_map_filtered[
    (univ_map_filtered['DATE'] >= univ_map_filtered['EFFFROM']) &
    (univ_map_filtered['DATE'] <= univ_map_filtered['EFFTHRU'])
    ]

merged = univ_map_filtered.merge(ven_map, on='UID', how='inner')
merged = merged[(merged['DATE'] >= merged['EFFFROM_y']) & (merged['DATE'] <= merged['EFFTHRU_y'])]
scope = merged.groupby('DATE')['PXID'].apply(list).reset_index()
scope = scope.set_index('DATE')
print(f'{scope=}')

# II DATA EXPLORATION
# scope['PXID'].apply(lambda x: len(x)).plot()
# plt.show()

# noticed the scope widens drastically on covid outbreak month March 2020 and then reverts to Feb range in April;
# not sure if genuine or not, I will take the discretion here to set March scope to be Feb union April;
# there is limited value in overhauling the portfolio one month and having to revert it the next
scope.loc['2020-03-31']['PXID'] = list(set(scope.loc['2020-02-28'].values[0] + scope.loc['2020-04-30'].values[0]))
pd.to_pickle(scope, 'hot/scope.pickle')

# returns
returns = prices.pct_change()


def fix_returns(df: pd.DataFrame, quantile=0.001):
    for col in df.columns:
        upper = df[col].quantile(1 - quantile)
        lower = df[col].quantile(quantile)
        # df[col].clip(upper=upper, lower=lower, inplace=True)
        df[col] = df[col].mask((df[col] > upper) | (df[col] < lower), df[col].mean())


fix_returns(returns)


# plotting
# so, let's take a look at the data now, given the universe of portfolio construction starts on 2011-01-31,
# we can take a look at the prices, see if we can cluster them somehow, and see how they performed without look-ahead

temp = prices[:'2011-01-31'].dropna(axis=1, how='all').fillna(method='bfill')
temp = temp.div(temp.iloc[0])
temp.plot()
plt.show()

# let's use self organising maps to take a more structured look
time_series_data = RobustScaler().fit_transform(temp.T)
som = MiniSom(x=3, y=3, input_len=time_series_data.shape[1])
som.pca_weights_init(time_series_data)
som.train_random(time_series_data, 1000)

clusters = {}
for i, stock_data in enumerate(time_series_data):
    cluster = som.winner(stock_data)
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(temp.columns[i])

pd.to_pickle(clusters, 'hot/clusters.pickle')

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()
for i, (cluster, members) in enumerate(clusters.items()):
    cluster_data = temp[members]
    cluster_mean = cluster_data.mean(axis=1)
    for col in cluster_data.columns:
        axes[i].plot(cluster_data.index, cluster_data[col], color='gray', alpha=0.3)
    axes[i].plot(cluster_data.index, cluster_mean, color='red', linewidth=2)  # mean by cluster
    axes[i].set_title(f'cluster {i + 1}')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].text(0.5, -0.3, str(), ha='center', va='center', fontsize=8, transform=axes[i].transAxes)
plt.tight_layout()
plt.show()


# let's take a look at the returns distribution in each cluster
temp_returns = returns[:'2011-01-31'].dropna(axis=1, how='all')
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()
for i, (cluster, members) in enumerate(clusters.items()):
    cluster_returns = temp_returns[members]
    for col in cluster_returns.columns:
        axes[i].hist(cluster_returns[col], bins=20, alpha=0.3, color='gray')
    axes[i].set_title(f'Cluster {i + 1} Returns')
    axes[i].set_xlabel('Return')
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# todo if time allows, revisit to do a hmm regime detection and plot returns by regime
