import datetime as dt
import math
import os
import pickle as pk

import bs4 as bs
import numpy as np
import pandas as pd
import pandas_market_calendars
import pytz
import quandl
import requests as rq
from alpha_vantage.timeseries import TimeSeries

import Constants

quandl.ApiConfig.api_key = Constants.quandlKey


def getTradingDates():
    """Get all dates in the range that the NYSE was trading on."""

    print("\nRETRIEVING TRADING DATES...\n")
    # Days NYSE was open
    trainingDates = (pandas_market_calendars.get_calendar('NYSE').valid_days(
        start_date=Constants.trainingStartDate,
        end_date=Constants.trainingEndDate)
    )
    testingDates = (pandas_market_calendars.get_calendar('NYSE').valid_days(
        start_date=Constants.testingStartDate,
        end_date=Constants.testingEndDate)
    )

    # Just save date (YYYY-MM-DD)
    trainingDates = [day.date() for day in trainingDates]
    testingDates = [day.date() for day in testingDates]

    if Constants.missingDate in trainingDates:
        trainingDates.remove(Constants.missingDate)
    if Constants.missingDate in testingDates:
        testingDates.remove(Constants.missingDate)

    pk.dump(trainingDates, open(Constants.dataDir + "trainingDates.p", "wb"))
    pk.dump(testingDates, open(Constants.dataDir + "testingDates.p", "wb"))
    return trainingDates, testingDates


def getTickers_Names(tradingDates):
    """Gets all SP500 Tickers that are also available on Quandl from startDate."""

    print("\nRETRIEVING SP500 TICKERS...\n")
    # All Quandl tickers available at startDate
    startTickers = quandl.get_table(
        'WIKI/PRICES',
        date=tradingDates[0],
        paginate=True
    )['ticker'].as_matrix()

    # Get all SP500 tickers from Wikipedia
    html = rq.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(html.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    # Store intersection of SP500 & Quandl tickers
    tickers_dict = {}
    print("Tickers mising from Quandl on start date:")
    missing, total = 0, 0
    for row in table.findAll('tr')[1:]:
        total += 1
        ticker = row.findAll('td')[0].text
        name = row.findAll('td')[1].text
        if ticker in startTickers:
            tickers_dict[ticker] = name
        else:
            print(ticker)
            missing += 1
    print("\n{} -> {}".format(total, total - missing))

    return tickers_dict


def getSP500Index(dates):
    """Get SP500 Index closing prices for specified dates."""

    print("\nRETRIEVING SP500 INDEX...\n")
    # Get SP500 Index adjusted closing price and change col name
    av = TimeSeries(key=Constants.alphaVantageKey, output_format='pandas')
    SP500Index, _ = av.get_daily_adjusted(symbol='^GSPC', outputsize='full')
    SP500Index = SP500Index['5. adjusted close'].to_frame()
    SP500Index.columns = ['SP500']

    # Reduce to timeframe to trading dates
    SP500Index = SP500Index.loc[str(dates[0]): str(dates[-1])]
    SP500Index.drop([str(Constants.missingDate)], inplace=True)

    # Convert string index to date (YYYY-MM-DD) & set as index
    dateStrings = SP500Index.index.values
    convertedDates = []
    for i in range(len(dates)):
        split = dateStrings[i].split('-')
        date = dt.date(int(split[0]), int(split[1]), int(split[2]))
        convertedDates.append(date)
    SP500Index.set_index([convertedDates], inplace=True)

    pk.dump(SP500Index, open(Constants.dataDir + "SP500Index.p", "wb"))
    return SP500Index


def getDailyMarketData(dates, tickers_dict):
    """Store daily market data for all tickers in concatenated DataFrame."""

    print("\nRETRIEVING DAILY MARKET DATA...\n")
    # Master DF
    dailyMarketData = pd.DataFrame()
    # Tickers to be removed
    unavailableTickers = []

    for i, ticker in enumerate(tickers_dict.keys()):
        print("Retrieving Prices: {} / {} ({})"
              .format(i + 1, len(tickers_dict), ticker),
              end="")

        tickerData = quandl.get_table(
            'WIKI/PRICES',
            ticker=ticker,
            date={'gte': dates[0],
                  'lte': dates[-1]},
            paginate=True
        )

        # If any dates missing, skip
        if tickerData.shape[0] != len(dates):
            print("\tDates Missing")
            unavailableTickers.append(ticker)
            continue
        print("")

        # Change date to YYYY-MM-DD and set as index
        tickerData['date'] = [day.date() for day in tickerData['date']]
        tickerData.set_index('date', inplace=True)

        # Drop unneeded columns
        tickerData.drop(['ticker', 'open', 'high', 'low', 'close', 'volume',
                         'ex-dividend', 'split_ratio'], axis=1, inplace=True)

        # Move volume column to front
        volume = tickerData['adj_volume']
        tickerData.drop(labels=['adj_volume'], axis=1, inplace=True)
        tickerData.insert(0, 'adj_volume', volume)

        # Adds second index level (ticker) for concatenation
        tickerData = pd.concat([tickerData], keys=[ticker], names=['ticker'])

        # Add to master DF
        dailyMarketData = pd.concat([dailyMarketData, tickerData])

    # Remove all unavailable tickers
    print("\n\n{} -> {}\n"
          .format(len(tickers_dict), len(tickers_dict) - len(unavailableTickers)))
    tickers_dict = {key: tickers_dict[key]
                    for key in tickers_dict if key not in unavailableTickers}

    pk.dump(dailyMarketData, open(Constants.dataDir + "dailyMarketData.p", "wb"))
    pk.dump(tickers_dict, open(Constants.dataDir + "tickers_dict.p", "wb"))
    pk.dump(list(tickers_dict.keys()), open(Constants.dataDir + "tickers.p", "wb"))
    return dailyMarketData, tickers_dict


def getSentimentURLs(tickers_dict):
    """Match stored company names with Knowsis's names and store corresponding URL."""

    print("\nRETRIEVING SENTIMENT URLs...\n")
    # Store each knowsisName, knowsisURL
    with open(Constants.dataDir + "Knowsis/_CompanyList.txt") as knowsis:
        knowsisFile = knowsis.readlines()

    # Find best match between tickerNames and knowsisNames
    # Best match = longest [0: n] matching chars
    for ticker, name in tickers_dict.items():
        name = stripName(name)

        bestMatch_URL = (0, "")
        for line in knowsisFile[1:]:
            parts = line.split(",")
            knowsisName = stripName(parts[0])
            knowsisURL = parts[1]

            if (ticker in Constants.hardcodedTickers.keys()
                    and knowsisName == Constants.hardcodedTickers[ticker]):
                bestMatch_URL = (99, knowsisURL)
                continue

            shortestLength = min(len(name), len(knowsisName))
            for i in range(shortestLength):
                if name[i] != knowsisName[i]:
                    if i > bestMatch_URL[0]:
                        bestMatch_URL = (i, knowsisURL)
                    break
                if i == (shortestLength - 1) and i > bestMatch_URL[0]:
                    bestMatch_URL = (i, knowsisURL)

        tickers_dict[ticker] = (tickers_dict[ticker], bestMatch_URL[0], bestMatch_URL[1])

    pk.dump(tickers_dict, open(Constants.dataDir + "tickers_dict.p", "wb"))
    return tickers_dict


def stripName(name):
    """Changes company name format to all lowercase, no spaces, no punctuation."""

    return (" " + name.lower()
            .replace("-", " ").replace("(", " ").replace(")", " ").replace("the ", "")
            .replace(" and ", "").replace(".", "").replace(",", "").replace("&", "")
            .replace("\'", "").replace("!", "").replace("*", "") + " ").replace(" ", "")


def downloadSentimentFiles(tickers_dict):
    """Downloads sentiment CSV files for all tickers."""

    print("\nDOWNLOADING SENTIMENT FILES...\n")
    # Change to download dir and get existing filenames
    os.chdir(Constants.dataDir + "/Knowsis/")

    i = 0
    for ticker, data in tickers_dict.items():
        print("Retrieving Sentiment Files: {} / {} ({})"
              .format(i + 1, len(tickers_dict), ticker), end="")
        i += 1

        knowsisURL = data[2]
        zippedFile = ticker + ".tar.gz"
        csvFile = ticker + ".csv"

        # Download zipped file and set file name
        os.system("wget -O {} {}".format(zippedFile, knowsisURL))
        # Unzip file
        os.system("tar -xvzf {}".format(zippedFile))

        # Delete zipped file
        os.remove(zippedFile)

        for file in os.listdir('.'):
            if file[0].islower() or file[0].isnumeric():
                os.rename(file, csvFile)

    # Change back to project's root directory
    os.chdir(Constants.projectDir)


def getSentimentData(dates, tickers):
    """Parses sentiment scores from CSV files into concatenated DataFrame."""

    print("\nPARSING SENTIMENT DATA...\n")
    # Change to download dir and get existing filenames
    os.chdir(Constants.dataDir + "/Knowsis/")

    sentimentScores = pd.DataFrame()

    for i, ticker in enumerate(tickers):
        print("Parsing Sentiment Scores: {} / {} ({})"
              .format(i + 1, len(tickers), ticker))

        # Open and store ticker's corresponding CSV file
        with open(ticker + ".csv") as sentimentFile:
            sentimentData = sentimentFile.readlines()

        # {YYYY-MM-DD: [POS, NEG, NEU, TOT]}
        date_sentimentScores = {}

        # Init all to 0
        for date in dates:
            date_sentimentScores[date] = [0, 0, 0, 0]

        for line in sentimentData[1:]:
            split = line.split(",")

            # [POS, NEG, NEU, TOT] per line
            scores = [int(score.strip()) for score in split[-4:]]

            # Convert date & time string to datetime object in NYSE timezone
            dateTimeString = split[4]
            dateTime = dt.datetime.strptime(dateTimeString, '%Y-%m-%d %H:%M:%S')
            dateTime = dateTime.replace(tzinfo=pytz.utc)
            dateTime = dateTime.astimezone(pytz.timezone('America/New_York'))

            # Iterate until sentiment time stamp is within first trading date threshold
            if dateTime.date() < dates[0] - dt.timedelta(days=1):
                continue
            if (dateTime.date() == dates[0] - dt.timedelta(days=1)
                    and dateTime.time() < Constants.sentimentThresholdTime):
                continue

            # If passed threshold time, move to next day
            if dateTime.time() > Constants.sentimentThresholdTime:
                dateTime = dateTime + dt.timedelta(days=1)

            # Iterate through dates until trading day found
            # or end of trading dates range reached
            endReached = False
            while dateTime.date() not in date_sentimentScores:
                dateTime = dateTime + dt.timedelta(days=1)
                if dateTime.date() > dates[-1]:
                    endReached = True
                    break

            if not endReached:
                date_sentimentScores[dateTime.date()] = [
                    a + b for
                    a, b in zip(date_sentimentScores[dateTime.date()], scores)
                ]
            else:
                break

        for date, scores in date_sentimentScores.items():
            # # If no scores for date, set normalised score to 0
            # if scores[-1] < 50:
            #     date_sentimentScores[date] = 0
            # # else normalise score
            # else:             # ########## BIGGER INFLUENCE FROM SMALLER TOTALS ##################
            #     date_sentimentScores[date] = ((scores[0] - scores[1]) / scores[-1]) * 100
            date_sentimentScores[date] = math.log(scores[0] + 0.5, 2) - math.log(scores[1] + 0.5, 2)

        # Creates DataFrame from date_sentiment dictionary
        # and adds second index level (ticker) for concatenation
        tickerSentimentScores = pd.concat(
            [pd.DataFrame.from_dict(date_sentimentScores, orient='index')],
            keys=[ticker],
            names=['ticker']
        )

        # Concatenate into master DF
        sentimentScores = pd.concat([sentimentScores, tickerSentimentScores])

    # Change dir back to project root
    os.chdir(Constants.projectDir)

    pk.dump(sentimentScores, open(Constants.dataDir + "sentimentScores.p", "wb"))
    return sentimentScores


def addTechnicalFeatures(dailyMarketData):
    """Adds technical indeicators as features."""

    print("\nADDING TECHNICAL FEATURES...\n")
    # % Change
    dailyMarketData['%_change'] = dailyMarketData['adj_close'].pct_change()

    # Simple Moving Average
    for ma in Constants.movingAverages:
        dailyMarketData['sma_{}'.format(ma)] = dailyMarketData['adj_close'].rolling(ma).mean()

    # SMA Crossover
    for ma in Constants.movingAverages[1:]:
        dailyMarketData['sma_cross_{}_{}'.format(Constants.movingAverages[0], ma)] = (
            dailyMarketData['sma_{}'.format(Constants.movingAverages[0])]
            > dailyMarketData['sma_{}'.format(ma)]
        ).astype(float)

    # Exponential Moving Average
    for ma in Constants.movingAverages:
        dailyMarketData['ema_{}'.format(ma)] = dailyMarketData['adj_close'].ewm(span=ma).mean()

    # EMA Crossover
    for ma in Constants.movingAverages[1:]:
        dailyMarketData['ema_cross_{}_{}'.format(Constants.movingAverages[0], ma)] = (
            dailyMarketData['ema_{}'.format(Constants.movingAverages[0])]
            > dailyMarketData['ema_{}'.format(ma)]
        ).astype(float)

    # Moving Average Convergence/Divergence
    df = pd.DataFrame()
    df['ema_26'] = dailyMarketData['adj_close'].ewm(span=26).mean()
    df['ema_12'] = dailyMarketData['adj_close'].ewm(span=12).mean()
    df['diff'] = (df['ema_12'] - df['ema_26'])
    df['signal'] = df['diff'].ewm(span=9).mean()
    dailyMarketData['macd'] = df['diff'] - df['signal']

    # Parabolic Stop & Reverse

    # Momentum
    dailyMarketData['momentum_5'] = dailyMarketData['adj_close'].diff(5)
    dailyMarketData['momentum_10'] = dailyMarketData['adj_close'].diff(10)

    # Relative Strength Index
    for ma in Constants.movingAverages[1:]:
        delta = dailyMarketData['adj_close'].diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0

        RolUp = dUp.rolling(ma).mean()
        RolDown = dDown.rolling(ma).mean().abs()

        RS = RolUp / RolDown
        dailyMarketData['rsi_{}'.format(ma)] = 100.0 - (100.0 / (1.0 + RS))

    # On-Balance Volume
    priceDiff = dailyMarketData['adj_close'].diff()
    prevVol = dailyMarketData['adj_volume'].shift()

    dailyMarketData['obv'] = np.select([priceDiff > 0, priceDiff < 0],
                                       [prevVol + dailyMarketData['adj_volume'],
                                        prevVol - dailyMarketData['adj_volume']],
                                       default=prevVol)

    return dailyMarketData


def addSP500Index(data, SP500, tickers):
    """Adds daily SP500 Index to each ticker in DataFrame"""

    print("\nADDING SP500 INDEX...\n")
    fullSP500 = pd.DataFrame()
    for ticker in tickers:
        individualSP500 = pd.concat(
            [SP500],
            keys=[ticker],
            names=['ticker']
        )

        fullSP500 = pd.concat([fullSP500, individualSP500])

    data['SP500'] = fullSP500
    return data


def addSentimentScores(data, sentimentScores):
    """Adds all sentiment scores to DataFrame."""

    print("\nADDING SENTIMENT SCORES...\n")
    data['sentiment'] = sentimentScores
    return data


def addLabels(data, tickers):
    """Adds training labels as last columns of DataFrame."""

    print("\nADDING LABELS...\n")

    data['down'] = (data['adj_close'].shift(-Constants.predictionWindow) <
                    data['adj_close']).astype(float)
    data['up'] = (data['adj_close'].shift(-Constants.predictionWindow) >=
                  data['adj_close']).astype(float)

    # Drop last 'predictionWindow' dates
    rowsToDrop = [data.xs(i, drop_level=False).index[data.loc[tickers[0]].shape[0] - 1 - j]
                  for j in range(Constants.predictionWindow)
                  for i in tickers]

    data.drop(rowsToDrop, inplace=True)
    data.dropna(inplace=True)

    pk.dump(data, open(Constants.dataDir + "fullData.p", "wb"))
    return data


def normalise(data):
    minDF = data.min()
    maxDF = data.max()
    data = (data - minDF) / (maxDF - minDF)
    return data, minDF['adj_close'], maxDF['adj_close']


def denormalise(price, min, max):
    return price * (max - min) + min


# ##############################################################################
# LOAD STORED DATA
# ##############################################################################

# trainingDates = pk.load(open(Constants.dataDir + "trainingDates.p", "rb"))
# testingDates = pk.load(open(Constants.dataDir + "testingDates.p", "rb"))
# trainingDates = trainingDates[Constants.movingAverages[-1]:]
# testingDates = testingDates[:-Constants.predictionWindow]
# allDates = trainingDates + testingDates
# tickers = pk.load(open(Constants.dataDir + "tickers.p", "rb"))
#
# data = pk.load(open(Constants.dataDir + "fullData.p", "rb"))
#
# # SP500Index = pk.load(open(Constants.dataDir + "SP500Index.p", "rb"))
# # dailyMarketData = pk.load(open(Constants.dataDir + "dailyMarketData.p", "rb"))
# # tickers_dict = pk.load(open(Constants.dataDir + "tickers_dict.p", "rb"))
# # sentimentScores = pk.load(open(Constants.dataDir + "sentimentScores.p", "rb"))
# # data = addTechnicalFeatures(dailyMarketData)
# # data = addSP500Index(data, SP500Index, tickers)
# # data = addSentimentScores(data, sentimentScores)
# # data = addLabels(data, tickers)
#
# data, minPrice, maxPrice = normalise(data)
# numFeatures = data.shape[1] - Constants.numLabels
# numSlices = math.ceil(len(tickers) / Constants.batchSize)
#
# print("Number of Companies:\t", len(tickers))
# print("Training Dates:\t\t", trainingDates[0], "-", trainingDates[-1])
# print("Testing Dates:\t\t", testingDates[0], "-", testingDates[-1])

# ##############################################################################
# RETRIEVE AND STORE DATA
# ##############################################################################

trainingDates, testingDates = getTradingDates()
allDates = trainingDates + testingDates
tickers_dict = getTickers_Names(allDates)
dailyMarketData, tickers_dict = getDailyMarketData(allDates, tickers_dict)
tickers = list(tickers_dict.keys())
SP500Index = getSP500Index(allDates)
tickers_dict = getSentimentURLs(tickers_dict)
downloadSentimentFiles(tickers_dict)
sentimentScores = getSentimentData(allDates, tickers)
