### Contents
- [Data](#data)
  - [Raw Dataset](#raw-dataset)
  - [Pre-Processing](#pre-processing)
    - [Dealing With Nulls](#dealing-with-nulls)
    - [Sorting](#sorting)
    - [Normalisation](#normalisation)
- [Batching](#batching)
    - [Sorting Prior to Batching](#sorting-prior-to-batching)

# Data

## Raw Dataset

* 1,710,756 rows x 111 columns
* Financial portfolio of 1424 financial instrument IDs and 1813 timestamps
* IDs: [0, 6, 7, ..., 2156, 2158]
* Timestamps: [0, ..., 1812]
* ID Lifespan: Each ID exists in the portfolio across a single continuous subset of the 1813 timestamps
* 108 features and 1 label per ID for each timestamp that the ID exists in the portfolio
* Each row corresponds to the 108 features and single label for an ID at a single timestamp
* Rows are sorted by ascending timestamp and then by ascending ID
* There are plenty of null values in the dataset

| id | timestamp | derived_0 | ... | derived_4 | fundamental_0 | ... | fundamental_63 | technical_0 | ... | technical_3 | technical_5 | ... | technical_44 | y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0 |...|...|...|...|...|...|...|...|...|...|...|...|...|
| 11 | 0 |...|...|...|...|...|...|...|...|...|...|...|...|...|
| 12 | 0 |...|...|...|...|...|...|...|...|...|...|...|...|...|
| 25 | 0 |...|...|...|...|...|...|...|...|...|...|...|...|...|
| ... | ... |...|...|...|...|...|...|...|...|...|...|...|...|...|
| 10 | 1 |...|...|...|...|...|...|...|...|...|...|...|...|...|
| 11 | 1 |...|...|...|...|...|...|...|...|...|...|...|...|...|
| 12 | 1 |...|...|...|...|...|...|...|...|...|...|...|...|...|
| 25 | 1 |...|...|...|...|...|...|...|...|...|...|...|...|...|
| ... | ... |...|...|...|...|...|...|...|...|...|...|...|...|...|


## Pre-Processing

### Dealing With Nulls

Initially, to facilitate creating the most simple Neural Network possible, I decided to simply fill in all null values with the mean value of its column, thus preventing the program from attempting to make calculations with nulls and crashing execution.

This filling of nulls with the mean soon changed to filling them with the value of 0 for reasons outlined when it came to [normalising the data](#normalisation).

### Sorting

To facilitate my [batching](#batching) algorithm, I had to re-order the dataset by grouping by ID and then ordering them by ID exit, ID entry, timestamp giving the following structure:

| id | timestamp | ... |
|----|-----------|-----|
| 1314 | 0 | ... |
| 1314 | 1 | ... |
| 1314 | 2 | ... |
| 899 | 0 | ... |
| 899 | 1 | ... |
| 899 | 2 | ... |
| 899 | 3 | ... |
| 899 | 4 | ... |
| 1063 | 1 | ... |
| 1063 | 2 | ... |
| 1063 | 3 | ... |
| 1063 | 4 | ... |
| 430 | 0 | ... |
| 430 | 1 | ... |
| 430 | 2 | ... |
| 430 | 3 | ... |
| 430 | 4 | ... |
| 430 | 5 | ... |
| ... | ... | ... |

### Normalisation

**Feature Normalisation**

Due to the large amount of null values in the dataset, I came to the conclusion that normalising all features to 0 mean would then allow me to fill in all null values with 0. My reasoning behind this was that, firstly the mean, variance, etc. of the data would remain unaffected, and secondly, when feeding the data through the network, all null values now set to 0 would force the internal matrix multiplications of these values also to equate to 0, thus not effecting the output. **(Is this the case???)**

In order to achieve this normalisation to 0 mean, my first choice was to use a Z-Score normalisation:

<p align="center"> 
<img src="http://mathurl.com/y9pw597o.png" alt="Z-Score Formula">
</p>

Z-Score Normalisation produces 0 mean I desired, while also aligning the 1st standard deviation to the range [-1, 1]. However, I quickly realised that, due to some values in the dataset being so large (max value is 1e+18) and a great distance away from the 1st standard deviation, these values would then still take a substantial value when normalised with Z-Score.

To combat these still very large values after Z-Score normalisation, I decided to keep the 0 mean normalisation, but not continue divide each value by the features standard deviation, and instead devised the following replacement:

* If the absolute value of the feature's maximum value is greater than that of the feature's minimum value, I would then divide each value the absolute value of feature's max
* Otherwise, if the opposite is true, I divide it by the absolute value of the feature's min

<p align="center"> 
<img src="http://mathurl.com/yd4bxhcm.png" alt="Normalise to [-1,1]">
</p>

This gives a normalisation where the value of the feature at the greatest distance away from the mean takes the value of 1 if it is positive and -1 if negative. All other values in the feature are then squashed proportionally.

The overall result of this normalisation is a 0 mean and all values within a distance of 1 from this mean.

**Label Normalisation**

All label values in the 'y' column of the dataset are originally within the range [-0.01, +0.01]. These will always need to be normalised to the output range of whatever [activation function](#) is being used within the network.

I originally chose to use the [ReLU activation function](#) which produces output in the range [0, inf], and so I normalised the labels to the range [0, 1] so that my network would learn to improve its predictions output from the ReLU activations of range [0, inf] to valid predictions in the range [0, 1].

# Batching

## Sorting Prior to Batching