import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import Constants
import DataWorker

df = pd.DataFrame(index=DataWorker.IDs, columns=DataWorker.TSs)

for i, ID in enumerate(DataWorker.IDs):
    print(i + 1, "/", len(DataWorker.IDs))
    bools = []
    for i in DataWorker.TSs:
        if i in DataWorker.ID_TS_dict[ID]:
            bools.append(True)
        else:
            bools.append(False)
    df.loc[ID] = bools

colours = mcolors.LinearSegmentedColormap.from_list(
    "n", ['#49585B', '#7FE6DA'])  # [No, Yes
chart = sb.heatmap(
    df,
    cmap=colours,
    square=True,
    cbar=False,
    xticklabels=False,
    yticklabels=False)
chart.set(xlabel='Timestamps', ylabel="IDs")
plt.title("Sorted By _____")
plt.savefig(Constants.dataDir + "BatchingAlgos/SortedBy_____")
