'''
Implementing LRCN structure
References:
    http://cs231n.stanford.edu/reports2016/221_Report.pdf
    https://arxiv.org/pdf/1411.4389v3.pdf
'''
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM

N_CLASSES = 101
SequenceLength = 10


def RNN(weights_dir, CNN_output):
    model = Sequential()

    model.add(LSTM(256, return_sequences=True, input_shape=(SequenceLength, CNN_output)))
    model.add(Dropout(0.9))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.9))
    model.add(Dense(N_CLASSES, activation='softmax'))

    if os.path.exists(weights_dir):
        model.load_weights(weights_dir)

    return model
