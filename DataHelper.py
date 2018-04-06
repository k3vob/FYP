import numpy as np
import pandas as pd
import talib as ta

movingAverages = [3, 5, 10, 20]


def normalise(data):
    for col in data.columns:
        if data[col].max() >= abs(data[col].min()):
            data[col] = data[col] / data[col].max()
        else:
            data[col] = data[col] / abs(data[col].min())
    return data


def buildTechnicals(rawPrices):
    # TA-Lib requires Numpy arrays
    high = np.array(rawPrices['high'])
    low = np.array(rawPrices['low'])
    close = np.array(rawPrices['close'])
    volume = np.array(rawPrices['volume'])
    sp500 = np.array(rawPrices['sp500'])

    data = pd.DataFrame()
    data['high'] = high
    data['low'] = low
    data['close'] = close
    data['volume'] = volume
    data['sp500'] = sp500

    # Moving Averages
    for ma in movingAverages:
        for col in ['open', 'high', 'low', 'close', 'sp500']:
            # MA trend
            data['{}_{}_ema'.format(col, ma)] = ta.EMA(np.array(rawPrices[col]), timeperiod=ma)
            # MA stationary
            data['{}_{}_ema'.format(col, ma)] = data['{}_{}_ema'.format(col, ma)].diff()

    # Bollinger Bands
    upperband, middleband, lowerband = ta.BBANDS(close)
    data['bbands'] = (close - lowerband) / (upperband - lowerband)

    # Parabolic SAR
    data['psar'] = ta.SAR(high, low)
    data['psar'] = data['psar'].diff()

    # Average Directional Index
    data['adx'] = ta.ADX(high, low, close)
    data['adx'] = data['adx'].diff()

    # AROON Oscillator
    data['aroon osc'] = ta.AROONOSC(high, low)

    # Commodidty Channel Index
    data['cci'] = ta.CCI(high, low, close)

    # Moving Average Convergence/Divergence
    data['macd'], _, _ = ta.MACD(close)

    # Relative Strength Index
    data['rsi'] = ta.RSI(close)

    # Stochastic Oscillator
    _, data['stochastic osc'] = ta.STOCH(high, low, close)

    # Ultimate Oscillator
    data['ultimate osc'] = ta.ULTOSC(high, low, close)

    # Volume
    data['volume'] = volume
    data['volume'] = data['volume'].diff()

    # Chaikin Oscillator
    data['chaikin osc'] = ta.ADOSC(high, low, close, volume)

    # On Balance Volume
    data['obv'] = ta.OBV(close, volume)

    return data
