import Constants
from Test2 import data

for col in data.columns:
    if len(col) <= 7:
        print(col + "\t\t", end="")
    else:
        print(col + "\t", end="")
    print("%.6f" % data[col].corr(data['up']))
