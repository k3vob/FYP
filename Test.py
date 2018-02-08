import datetime as dt
import os

import Constants

os.chdir(Constants.dataDir + "/Knowsis/")

startTime = dt.time(9, 30)
endTime = dt.time(16, 0)


for file in os.listdir('.'):
    if file[-4:] != '.csv':
        continue

    with open(file) as sentimentFile:
        sentimentData = sentimentFile.readlines()

    dates = {}

    for line in sentimentData[1:]:     # #######
        parts = line.split(",")
        dateTimeString = parts[4]
        sentimentScores = [int(score.strip()) for score in parts[-4:]]
        dateTime = dt.datetime.strptime(dateTimeString, '%Y-%m-%d %H:%M:%S')
        if dateTime.time() < startTime:
            date = dateTime.date()
            if date not in dates:
                dates[date] = sentimentScores
            else:
                dates[date] = [a + b for a, b in zip(dates[date], sentimentScores)]
        else:
            date = dateTime.date() + dt.timedelta(days=1)
            if date not in dates:
                dates[date] = sentimentScores
            else:
                dates[date] = [a + b for a, b in zip(dates[date], sentimentScores)]

    print(dates)

    break                               # ######

os.chdir(Constants.projectDir)
