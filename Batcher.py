import Constants


def getNextTrainBatch(data, cursor):
    x = [data.iloc[cursor: cursor + Constants.sequenceLength, : -1].as_matrix()]
    y = [data.iloc[cursor: cursor + Constants.sequenceLength, -1:].as_matrix()]
    cursor += 1
    if cursor + Constants.sequenceLength > data.shape[0]:
        cursor = 0
    return x, y, cursor


def getNextOnlineBatch(data, cursor, predict=False):
    if predict:
        cursor += 1
    x = [data.iloc[cursor: cursor + Constants.sequenceLength, : -1].as_matrix()]
    y = [data.iloc[cursor: cursor + Constants.sequenceLength, -1:].as_matrix()]
    if predict:
        if cursor + Constants.sequenceLength >= data.shape[0]:
            cursor = 0
        return x, y, cursor
    return x, y
