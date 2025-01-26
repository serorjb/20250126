import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt


def get_frames(root: str = 'data'):
    frames: dict = dict()
    for file in os.listdir(root):
        if file.endswith('.csv'):
            print(file[:-4])
            frames[file[:-4]] = pd.read_csv(f'{root}/{file}')
    return frames


frames = get_frames()
# def fx_conv(prices: pd.DataFrame, fx: pd.DataFrame):
#     for ccy in prices['ISOCURR'].unique():



# df = fx_conv(prices, fx)



# fx['TICKER'].unique()  # only prints CADUSD
# fx.shape[0]
# fx.drop_duplicates().shape[0]  # no duplicates

fx = pd.read_csv('data/fx.csv')
prices = pd.read_csv('data/price.csv')
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
prices['CLOSE'] = np.where(prices['ISOCURR'] == 'USD', prices['CLOSE'], prices['CLOSE']*prices['CAD'])
prices = prices[['PXID', 'CLOSE']]
print(f'{prices=}')

# now converting the ID map to a PXID map defining the eligible scope
import pandas as pd
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

