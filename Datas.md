## Raw Dataset

* 1,710,756 rows x 111 columns
* Financial portfolio of 1424 financial instrument IDs and 1813 timestamps
* IDs: [0, 6, 7, ..., 2156]
* Timestamps: [0, ..., 1812]
* ID Lifespan: Each ID exists in the portfolio across a single continuous subset of the 1813 timestamps
* 108 features and 1 label per ID for each timestamp that the ID exists in the portfolio
* Columns:

| id | timestamp | derived_0 | ... | derived_4 | fundamental_0 | ... | fundamental_63 | technical_0 | ... | technical_3 | technical_5 | ... | technical_44 | y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

* Each row corresponds to the 108 features and single label for an ID at a single timestamp
* Rows sorted by ascending timestamp and then by ascending ID

## Pre-Processing