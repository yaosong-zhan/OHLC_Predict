import pandas as pd

class QueryData(object):

    def __init__(self):
        pass

    def login(self, name, passwd, dbName):
        pass

    def download(self, sqll):
        pass

    def readlocal(self, filename, filetype = 'csv'):
        if filetype == 'csv':
            return pd.read_csv(filename)

