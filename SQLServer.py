import sqlalchemy as sqla
from pandas import read_sql
import urllib
import sys

driver = 'FreeTDS'
server = '127.0.0.1'
port = '1433'
username = 'sa'
password = 'Obrien55'
db = 'TwoSigma'
connection = None

for i in range(1, 10):
    print(i)

conn_str = urllib.parse.quote_plus(
    r'Driver=' + driver + ';'
    r'Server=' + server + ';'
    r'port=' + port + ';'
    r'uid=' + username + ';'
    r'pwd=' + password + ';'
    r'Database=' + db + ';'
)

try:
    engine = sqla.create_engine('mssql+pyodbc:///?odbc_connect={}'.format(conn_str))
    connection = engine.connect()
    print("SUCCESS: Connection established to SQL Server")
except:
    input("ERROR: Failed to connect to SQL Server.")
    sys.exit()


def disconnect():
    try:
        connection.close()
        print("SUCCESS: Connection closed to SQL Server")
    except:
        input("ERROR: Failed to disconnect from SQL Server.")
        sys.exit()


def execute(query):
    connection.execute(query)


def createTable(df, tableName):
    batchSize = 100000
    if engine.has_table(tableName):
        dropTable(tableName)
    for i in range(int(df.shape[0] / batchSize) + 1):
        start = (batchSize * i)
        end = min((batchSize * i) + (batchSize), df.shape[0])
        print("Importing rows " + str(start) + " - " + str(end - 1) + " / " + str(df.shape[0] - 1))
        dfTemp = df[start:end]
        dfTemp.to_sql(tableName, connection, index=False, chunksize=10000, if_exists='append')


def exportTable(tableName):
    return read_sql('SELECT * FROM ' + tableName, connection)


def dropTable(tableName):
    execute('DROP TABLE ' + tableName)
