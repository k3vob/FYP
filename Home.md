### Contents
- [Data](#data)
  - [Raw Dataset](#raw-dataset)
  - [Pre-Processing](#pre-processing)
- [Batching](#batching)

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

## Sorting

The dataset is re-ordered by grouping by ID and then ordering them by ID exit, ID entry, timestamp for [batching](#sorting) reasons, giving the following structure:

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



# Batching

## Sorting