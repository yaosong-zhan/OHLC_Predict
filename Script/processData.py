import pandas as pd
import numpy as np

class ProcessData(object):

    def __init__(self, data, split, cols):

        iSplit = int(len(data)*split)
        self.dataTrain = data.get(cols).values[:iSplit]
        self.dataTest = data.get(cols).values[iSplit:]
        self.trainLen = len(self.dataTrain)
        self.testLen = len(self.dataTest)

    def getTestData(self, seqLen, normalize):

        dataX = []
        dataY = []

        for i in range(self.testLen - seqLen):
            windows = self.dataTest[i:i+seqLen]
            x = windows[:, :-1]
            y = windows[-1, -1]
            dataX.append(x)
            dataY.append(y)
        
        return np.array(dataX), np.array(dataY)


    def getTrainData(self, seqLen, normalise):
        dataX = []
        dataY = []
        for i in range(self.trainLen - seqLen):
            windows = self.dataTrain[i:i+seqLen]
            x = windows[:, :-1]
            y = windows[-1, -1]
            dataX.append(x)
            dataY.append(y)
        
        return np.array(dataX), np.array(dataY)
