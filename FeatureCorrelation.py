import Constants
from DataWorker import data

print(data)

for col in data.columns[:-Constants.numLabels]:
    if len(col) <= 7:
        print(col + "\t\t", end="")
    else:
        print(col + "\t", end="")
    print("%.6f" % data[col].corr(data['up']))

# 0   -> 0.121258
# 50  -> 0.087134
# 100 -> 0.065487
# log2(P + 0.5) - log2(P + 0.5)
#     -> 0.13
