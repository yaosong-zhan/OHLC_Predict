import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def loadModel(self, filename):
        print('[Model] Loading model from file %s' % filename)
        self.model = load_model(filename)
    
    def buildModel(self, configs):

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropoutRate = layer['dropoutRate'] if 'dropoutRate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            returnSeq = layer['returnSeq'] if 'returnSeq' in layer else None
            inputTimesteps = layer['inputTimesteps'] if 'inputTimesteps' in layer else None
            inputDim = layer['inputDim'] if 'inputDim' in layer else None

            if layer['type'] == 'dense':
                self.model(Dense(neurons, activation = activation))
            elif layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(inputTimesteps, inputDim), return_sequences=returnSeq))
            elif layer['type'] == 'dropout':
                self.model.add(Dropout(dropoutRate))

        self.model.compile(loss = configs['model']['loss'], optimizer = configs['model']['optimizer'])

        print('[Model] Model Compiled')

    def train(self, x, y,epochs, batchSize, saveDir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batchSize))

        saveName = os.path.join(saveDir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ]

        self.model.fit(x, y, epochs = epochs, batch_size = batchSize, callbacks = callbacks)
        self.model.save(saveName)
    
    def predictPoint(self, data):
        prediction = self.model.predict(data)
        prediction = np.reshape(prediction, (prediction.size,))
        return prediction



