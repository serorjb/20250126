import pandas as pd


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
