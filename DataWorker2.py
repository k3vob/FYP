import datetime as dt
import fnmatch
import math
import os
import pickle as pk

import bs4 as bs
import pandas as pd
import pandas_market_calendars
import pytz
import quandl
import requests as rq
from alpha_vantage.timeseries import TimeSeries

import Constants

quandl.ApiConfig.api_key = Constants.quandlKey
missingDay = dt.date(2017, 11, 8)   # MISSING FROM QUANDL
# 2017-08-07 missing from 10 tickers (including Apple and Amazon)
#   ABT ATVI ADBE AKAM AMZN AAPL CELG CMCSA COST INTC QCOM
# 2017-04-14 present for FAST and LOW but none other
# TRV and MMM missing ~20 dates


def getDays():
    endDate = dt.date(2017, 12, 31)
    startDate = dt.date(2014, 1, 1)
    #startDate = endDate - dt.timedelta(days=int(365.25 * Constants.years))
    days = pandas_market_calendars.get_calendar('NYSE').valid_days(
        start_date=startDate, end_date=endDate)
    days = [day.date() for day in days]
    days.remove(missingDay)
    return days


def getSP500Tickers():
    html = rq.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(html.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = {}
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        name = row.findAll('td')[1].text
        tickers[ticker] = name
    return tickers


def getSP500Index(tradingDays):
    # Get SP500IndexPricesDF Index adjusted closing price and change col name
    av = TimeSeries(key=Constants.alphaVantageKey, output_format='pandas')
    SP500IndexDF, _ = av.get_daily_adjusted(symbol='^GSPC', outputsize='full')
    SP500IndexDF = SP500IndexDF['5. adjusted close'].to_frame()
    SP500IndexDF.columns = ['SP500_adj_close']

    # Reduce to timeframe to trading days
    SP500IndexDF = SP500IndexDF.loc[str(startDate):str(endDate)]
    SP500IndexDF.drop([str(missingDay)], inplace=True)

    # Convert string index to date
    dateStrings = SP500IndexDF.index.values
    dates = []
    for i in range(len(tradingDays)):
        parts = dateStrings[i].split('-')
        date = dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
        dates.append(date)
    SP500IndexDF.set_index([dates], inplace=True)

    return SP500IndexDF


def getPricesForTickers(tickers, SP500Index, tradingDays):
    pricesDF = None

    # All tickers available at startDate
    startTickers = quandl.get_table(
        'WIKI/PRICES',
        date=startDate,
        paginate=True
    )['ticker'].as_matrix()

    # Remove all S&P500 tickers not available on Quandl
    unavailableTickers = list(set(tickers).difference(set(startTickers)))
    unavailableTickers = unavailableTickers + Constants.ignoredTickers
    tickers = {key: tickers[key] for key in tickers if key not in unavailableTickers}
    unavailableTickers = []

    for i, ticker in enumerate(tickers):
        print(i + 1, "/", len(tickers))

        tickerDF = quandl.get_table(
            'WIKI/PRICES',
            ticker=ticker,
            date={'gte': startDate,
                  'lte': endDate},
            paginate=True
        )

        # If any days missing, skip
        if tickerDF.shape[0] != len(tradingDays):
            unavailableTickers.append(ticker)
            continue

        # Change date to YYYY-MM-DD
        tickerDF['date'] = [day.date() for day in tickerDF['date']]

        tickerDF.set_index('date', inplace=True)

        # Unneeded columns
        # tickerDF.drop(['ticker', 'open', 'high', 'low', 'close', 'ex-dividend',
        #                'volume', 'split_ratio'], axis=1, inplace=True)
        tickerDF.drop(['ticker'], axis=1, inplace=True)

        # Adds SP500 Index column
        tickerDF = pd.concat([tickerDF, SP500Index], axis=1)

        # Add all price moving averages
        for ma in Constants.movingAverages:
            tickerDF['{}_ma'.format(ma)] = tickerDF['adj_close'].rolling(ma).mean()
        for ma in Constants.movingAverages:
            tickerDF['SP500_{}_ma'.format(ma)] = tickerDF['SP500_adj_close'].rolling(ma).mean()

        # Add daily % changes
        tickerDF['%_change'] = tickerDF['adj_close'].pct_change()
        tickerDF['SP500_%_change'] = tickerDF['SP500_adj_close'].pct_change()

        # Adds second index level (ticker) for concatenation
        tickerDF = pd.concat([tickerDF], keys=[ticker], names=['ticker'])

        # Concatenates tickerDF to SP500DF
        if pricesDF is None:
            pricesDF = tickerDF
        else:
            pricesDF = pd.concat([pricesDF, tickerDF])

    # Remove all unavailable tickers
    tickers = {key: tickers[key] for key in tickers if key not in unavailableTickers}

    return pricesDF, tickers


def stripName(name):
    return (" " + name.lower()
            .replace("-", " ").replace("(", " ").replace(")", " ").replace("the ", "")
            .replace(" and ", "").replace(".", "").replace(",", "").replace("&", "")
            .replace("\'", "").replace("!", "").replace("*", "") + " ").replace(" ", "")


def getKnowsisLinks(tickers):
    with open(Constants.dataDir + "Knowsis/_CompanyList.txt") as knowsis:
        knowsisFile = knowsis.readlines()

    for ticker, name in tickers.items():
        name = stripName(name)

        bestMatch = (0, "")
        for line in knowsisFile[1:]:
            parts = line.split(",")
            knowsisName = stripName(parts[0])
            knowsisLink = parts[1]

            if (ticker in Constants.hardcodedTickers.keys()
                    and knowsisName == Constants.hardcodedTickers[ticker]):
                bestMatch = (99, knowsisLink)
                continue

            length = min(len(name), len(knowsisName))
            for i in range(length):
                if name[i] != knowsisName[i]:
                    if i > bestMatch[0]:
                        bestMatch = (i, knowsisLink)
                    break
                if i == length - 1 and i > bestMatch[0]:
                    bestMatch = (i, knowsisLink)

        tickers[ticker] = bestMatch[1]

    return tickers


def downloadKnowsisFiles(tickers):
    # Change working directory
    os.chdir(Constants.dataDir + "/Knowsis/")
    files = os.listdir('.')

    for ticker, url in tickers.items():
        # File names
        tarFile = ticker + ".tar.gz"
        csvFile = ticker + ".csv"

        if csvFile in files:
            continue

        # Download zipped file and set file name
        os.system("wget -O {} {}".format(tarFile, url))
        # Unzip file
        os.system("tar -xvzf {}".format(tarFile))

        # Find and rename unzipped file
        # (only file with lower case)
        for file in files:
            if fnmatch.fnmatch(file, '[a-z]*'):
                os.rename(file, csvFile)

        # Delete zipped file
        os.remove(tarFile)

    # Change to project's root directory
    os.chdir(Constants.projectDir)


def getKnowsisSentimentData(tradingDays, tickers, pricesDF):
    os.chdir(Constants.dataDir + "/Knowsis/")

    thresholdTime = dt.time(17, 30)

    sentimentDF = pd.DataFrame()

    for i, ticker in enumerate(tickers):
        print(i + 1, "/", len(tickers))
        with open(ticker + ".csv") as file:
            sentimentData = file.readlines()

        day_sentimentScores = {}
        for day in tradingDays:
            day_sentimentScores[day] = [0, 0, 0, 0]

        for line in sentimentData[1:]:
            lineParts = line.split(",")

            # Scores as [pos, neg, neu, tot]
            scores = [int(score.strip()) for score in lineParts[-4:]]

            # Convert date & time string to datetime object in NYSE timezone
            dateTimeString = lineParts[4]
            dateTime = dt.datetime.strptime(dateTimeString, '%Y-%m-%d %H:%M:%S')
            dateTime = dateTime.replace(tzinfo=pytz.utc)
            dateTime = dateTime.astimezone(pytz.timezone('America/New_York'))

            if dateTime.time() > thresholdTime:
                dateTime = dateTime + dt.timedelta(days=1)

            endReached = False
            while dateTime.date() not in day_sentimentScores:
                dateTime = dateTime + dt.timedelta(days=1)
                if dateTime.date() > tradingDays[-1]:
                    endReached = True
                    break

            if not endReached:
                day_sentimentScores[dateTime.date()] = [
                    a + b for
                    a, b in zip(day_sentimentScores[dateTime.date()], scores)
                ]

        for date, scores in day_sentimentScores.items():
            if scores[-1] == 0:
                day_sentimentScores[date] = 0
            else:           # Bigger influence from smaller total amounts
                day_sentimentScores[date] = ((scores[0] - scores[1]) / scores[-1]) * 100

        # Adds second index level (ticker) for concatenation
        tickerSentimentDF = pd.concat(
            [pd.DataFrame.from_dict(day_sentimentScores, orient='index')],
            keys=[ticker],
            names=['ticker']
        )

        sentimentDF = pd.concat([sentimentDF, tickerSentimentDF])

    prices['sentiment'] = sentimentDF

    os.chdir(Constants.projectDir)
    return pricesDF


def normalise(pricesDF):
    minDF = pricesDF.min()
    maxDF = pricesDF.max()
    pricesDF = (pricesDF - minDF) / (maxDF - minDF)
    return pricesDF, minDF['adj_close'], maxDF['adj_close']


def denormalise(price, min, max):
    return price * (max - min) + min


# def addLabels(pricesDF):
#     pricesDF['up'] = (pricesDF['adj_close'].diff() >= 0).astype(float)
#     pricesDF['down'] = (pricesDF['adj_close'].diff() < 0).astype(float)
#
#     # # Drop 1st row of each index
#     # # (label and % change refers to last row of previous index)
#     # rowsToDrop = [SP500Prices.xs(i, drop_level=False).index[0] for i in SP500Tickers]
#     # return SP500Prices.drop(rowsToDrop)
#
#     # Drop all nulls (label and % change of 1st row per ticker refers to previous ticker)
#     pricesDF.dropna(inplace=True)
#     return pricesDF


def addLabels(pricesDF, tickers):
    window = 10
    pricesDF['up1'] = (pricesDF['adj_close'].diff(1) >= 0).astype(float)
    pricesDF['down1'] = (pricesDF['adj_close'].diff(1) < 0).astype(float)
    pricesDF['up10'] = (pricesDF['adj_close'].diff(window) >= 0).astype(float)
    pricesDF['down10'] = (pricesDF['adj_close'].diff(window) < 0).astype(float)

    # Drop 1st row of each index
    # (label and % change refers to last row of previous index)
    rowsToDrop = [pricesDF.xs(i, drop_level=False).index[j] for j in range(window) for i in tickers]
    return pricesDF.drop(rowsToDrop)

    # Drop all nulls (label and % change of 1st row per ticker refers to previous ticker)
    # pricesDF.dropna(inplace=True)
    return pricesDF


# ##############################################################################
# EXECUTION
# ##############################################################################

tradingDays = getDays()
startDate = tradingDays[0]
endDate = tradingDays[-1]

################################################################################
# RETRIEVE AND STORE DATA
################################################################################

tickers = getSP500Tickers()
SP500Index = getSP500Index(tradingDays)
prices, tickers = getPricesForTickers(tickers, SP500Index, tradingDays)
tickers = getKnowsisLinks(tickers)
downloadKnowsisFiles(tickers)
prices = getKnowsisSentimentData(tradingDays, tickers, prices)
# prices = addLabels(prices)
tickers = list(tickers.keys())
pk.dump(tickers, open(Constants.dataDir + "tickers.p", "wb"))
pk.dump(prices, open(Constants.dataDir + "prices.p", "wb"))

# ##############################################################################
# LOAD, NORMALISE & LABEL DATA
# ##############################################################################

# tickers = pk.load(open(Constants.dataDir + "tickers.p", "rb"))
#
# prices = pk.load(open(Constants.dataDir + "prices.p", "rb"))
# prices = addLabels(prices, tickers)
# prices, minPrice, maxPrice = normalise(prices)
#
# numDays = prices.loc[tickers[0]].shape[0]
# numTrainingDays = int(numDays * Constants.trainingPercentage)
# numFeatures = prices.shape[1] - Constants.numLabels
# numSlices = math.ceil(len(tickers) / Constants.batchSize)
