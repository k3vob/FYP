### Contents
- [Data](#data)
  - [Raw Dataset](#raw-dataset)
  - [Pre-Processing](#pre-processing)
    - [Dealing With Nulls](#dealing-with-nulls)
    - [Sorting](#sorting)
    - [Normalisation](#pre-processing)
- [Batching](#batching)
    - [Sorting](#sorting-prior-to-batching)

# Data

## Raw Dataset

* 1,710,756 rows x 111 columns
* Financial portfolio of 1424 financial instrument IDs and 1813 timestamps
* IDs: [0, 6, 7, ..., 2156]
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



# Batching

## Sorting Prior to Batching