'''
Implementing LRCN structure
References:
    http://cs231n.stanford.edu/reports2016/221_Report.pdf
    https://arxiv.org/pdf/1411.4389v3.pdf
'''
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Reshape, Permute, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

N_CLASSES = 101
IMSIZE = (216, 216)
SequenceLength = 10
BatchSize = 30
CNN_output = 154880


def RNN():
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(SequenceLength, CNN_output)))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))
    sgd = SGD(lr=0.001, decay=1e-5, momentum=0.5, nesterov=True, clipnorm=1.)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # model structure summary
    print(model.summary())

    return model
