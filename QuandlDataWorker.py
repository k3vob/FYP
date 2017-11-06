import Constants
import quandl
import pandas as pd
import numpy as np

# First Date:   1999-11-18
# Num Tickers:  3194

quandl.ApiConfig.api_key = '53psox6QtoBdzpgXYs75'
# quandl.get_table('WIKI/PRICES', ticker='A', date='1999-11-18,1999-11-19,1999-11-22')

tickers = np.load(Constants.dataDir + "Quandl/tickers.npy")
cols = np.load(Constants.dataDir + "Quandl/columns.npy")


table = quandl.get_table('WIKI/PRICES', ticker=tickers[0])
print(table)
