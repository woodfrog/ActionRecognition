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
from keras.optimizers import SGD, Adam

N_CLASSES = 101
SequenceLength = 10


def RNN(weights_dir, CNN_output):
    model = Sequential()

    model.add(LSTM(256, return_sequences=True, input_shape=(SequenceLength, CNN_output)))
    model.add(Dropout(0.9))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.9))
    model.add(Dense(N_CLASSES, activation='softmax'))
    # sgd = SGD(lr=0.001, decay=1e-5, momentum=0.5, nesterov=True, clipnorm=1.)
    adam = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # model structure summary
    print(model.summary())

    if os.path.exists(weights_dir):
        model.load(weights_dir)

    return model
