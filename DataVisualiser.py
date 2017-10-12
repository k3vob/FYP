import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import DataWorker
import Constants


df = pd.DataFrame(index=DataWorker.IDs, columns=DataWorker.TSs)

# df = pd.DataFrame(index=DataWorker.ID_TS_dict.keys(), columns=DataWorker.TSs)

for i, ID in enumerate(DataWorker.IDs):
    print(i + 1, "/", len(DataWorker.IDs))
    bools = []
    for i in DataWorker.TSs:
        if i in DataWorker.ID_TS_dict[ID]:
            bools.append(True)
        else:
            bools.append(False)
    df.loc[ID] = bools

colours = mcolors.LinearSegmentedColormap.from_list("n", ['#EF5350', '#4CAF50'])  # [No, Yes]
image = sb.heatmap(df, cmap=colours, square=True, cbar=False, xticklabels=False, yticklabels=False)
image.set(xlabel='Timestamps', ylabel="IDs")
plt.title("Sorted By Overlap Ratio")
plt.savefig(Constants.data_dir + "BatchingAlgos/SortedByOverlapRatio")
