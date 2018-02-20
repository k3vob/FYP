import numpy as np
import pandas as pd
import talib

from Main import dailyMarketData as data

mas = [3, 5, 10]

# % Change
data['%_change'] = data['adj_close'].pct_change()

# Simple Moving Average
for ma in mas:
    data['sma_{}'.format(ma)] = data['adj_close'].rolling(ma).mean()

# SMA Crossover
for ma in mas[1:]:
    data['sma_cross_{}_{}'.format(mas[0], ma)] = (
        data['sma_{}'.format(mas[0])] > data['sma_{}'.format(ma)]
    ).astype(float)

# Exponential Moving Average
for ma in mas:
    data['ema_{}'.format(ma)] = data['adj_close'].ewm(span=ma).mean()

# EMA Crossover
for ma in mas[1:]:
    data['ema_cross_{}_{}'.format(mas[0], ma)] = (
        data['ema_{}'.format(mas[0])] > data['ema_{}'.format(ma)]
    ).astype(float)

# Moving Average Convergence/Divergence
df = pd.DataFrame()
df['ema_26'] = data['adj_close'].ewm(span=26).mean()
df['ema_12'] = data['adj_close'].ewm(span=12).mean()
df['diff'] = (df['ema_12'] - df['ema_26'])
df['signal'] = df['diff'].ewm(span=9).mean()
data['macd'] = df['diff'] - df['signal']

# Parabolic Stop & Reverse

# Momentum
data['momentum_5'] = data['adj_close'].diff(5)
data['momentum_10'] = data['adj_close'].diff(10)

# Relative Strength Index
for ma in mas[1:]:
    delta = data['adj_close'].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(ma).mean()
    RolDown = dDown.rolling(ma).mean().abs()

    RS = RolUp / RolDown
    data['rsi_{}'.format(ma)] = 100.0 - (100.0 / (1.0 + RS))

# On-Balance Volume
priceDiff = data['adj_close'].diff()
prevVol = data['adj_volume'].shift()

data['obv'] = np.select([priceDiff > 0, priceDiff < 0],
                        [prevVol + data['adj_volume'], prevVol - data['adj_volume']],
                        default=prevVol)

print(data)

data['up'] = (data['adj_close'].diff(1) >= 0).astype(float)
# data['down'] = (data['adj_close'].diff(1) < 0).astype(float)
data['up10'] = (data['adj_close'].diff(10) >= 0).astype(float)
# data['down10'] = (data['adj_close'].diff(10) < 0).astype(float)

rowsToDrop = [data.xs(i, drop_level=False).index[j]
              for j in range(10)
              for i in tickers]
data.drop(rowsToDrop, inplace=True)
data.dropna(inplace=True)
for col in data.columns[: -2]:
    if len(col) <= 7:
        print(col + "\t\t", end="")
    else:
        print(col + "\t", end="")
    print("%.6f" % data[col].corr(data['up']), end="\t")
    print("%.6f" % data[col].corr(data['up10']))
