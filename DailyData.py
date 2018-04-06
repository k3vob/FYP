import datetime as dt
import pickle as pk

import quandl
from alpha_vantage.timeseries import TimeSeries

import Constants
import DataHelper

quandl.ApiConfig.api_key = 'AysSyMCk5fZSBkHA-8_i'
alphaVantageKey = 'XIZOWSOCZRYV23XJ'


def getDailyMarketDataSet():
    """Retrieves market prices from Quandl."""
    df = quandl.get_table(
        'WIKI/PRICES',
        ticker=Constants.ticker,
        date={'gte': Constants.dailyStartDate, 'lte': Constants.dailyEndDate},
        qopts={'columns':
               ['date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']},
        paginate=True
    )
    # Rename cols
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    # Change date + time to  just date
    df['date'] = [day.date() for day in df['date']]
    return df


def getSP500Index(dates):
    """Retrieves SP500 close dataSet from AlphaVantage."""
    av = TimeSeries(key=alphaVantageKey, output_format='pandas')
    SP500, _ = av.get_daily_adjusted(symbol='^GSPC', outputsize='full')
    SP500 = SP500['5. adjusted close'].to_frame()
    # Rename col
    SP500.columns = ['sp500']
    # Strip to date range being used
    SP500 = SP500.loc[str(dates[0]): str(dates[-1])]
    # Convert dates from strings to datetime.dates
    dateStrings = SP500.index.values
    convertedDates = []
    for dateString in dateStrings:
        split = dateString.split('-')
        date = dt.date(int(split[0]), int(split[1]), int(split[2]))
        convertedDates.append(date)
    # Add dates as first col
    SP500.insert(0, 'date', convertedDates)
    # Match dates
    SP500 = SP500.loc[SP500['date'].isin(dates)]
    # Set integer indexes
    SP500.reset_index(inplace=True, drop=True)
    return SP500['sp500']


def generateDataSet():
    """Retrieves and stores dates, raw prices & working dataSet set."""
    rawPrices = getDailyMarketDataSet()
    dates = list(rawPrices['date'])
    rawPrices['sp500'] = getSP500Index(dates)
    rawPrices.drop('date', axis=1, inplace=True)
    stationaryPrices = rawPrices.pct_change()
    stationaryPrices.dropna(inplace=True)
    stationaryPrices.reset_index(drop=True, inplace=True)
    dataSet = DataHelper.buildTechnicals(stationaryPrices)
    dataSet = DataHelper.normalise(dataSet)
    dataSet['label'] = (rawPrices['close'].shift(-Constants.predictionWindow) >= rawPrices['close']).astype(float)
    totalCount = dataSet.shape[0]
    dataSet.dropna(inplace=True)
    dataSet.reset_index(drop=True, inplace=True)
    numToDrop = totalCount - dataSet.shape[0]
    rawPrices = list(rawPrices['close'])
    rawPrices = rawPrices[numToDrop:]   # Don't drop last predictionWindow for demo
    dataSet = dataSet[:-Constants.predictionWindow]
    pk.dump(rawPrices, open(Constants.dataDir + "dailyPrices.p", "wb"))
    pk.dump(dataSet, open(Constants.dataDir + "dailyData.p", "wb"))
