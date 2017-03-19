'''
Implementing LRCN structure
References:
    http://cs231n.stanford.edu/reports2016/221_Report.pdf
    https://arxiv.org/pdf/1411.4389v3.pdf
'''

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Reshape, Permute, Activation
from keras.layers import Convolution2D, MaxPooling3D, ConvLSTM2D
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
import keras.callbacks
import os, random
import UCF_utils as util

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


def fit_model(model, train_data, test_data):
    weights_dir = 'RNN_weights.h5'
    try:
        if os.path.exists(weights_dir):
            model.load_weights(weights_dir)
            print('Load weights')
        train_generator = util.video_generator(train_data, BatchSize, SequenceLength, CNN_output, N_CLASSES)
        test_generator = util.video_generator(test_data, BatchSize, SequenceLength, CNN_output, N_CLASSES)
        print('Start fitting model')
        checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_weights_only=True)
        model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=200,
            validation_data=test_generator,
            validation_steps=50,
            verbose=2,
            callbacks=[checkpointer]
        )
    except KeyboardInterrupt:
        print('Training time:')


if __name__ == '__main__':
    data_dir = '/home/changan/ActionRocognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'CNN_Processed')

    train_data, test_data, class_index = util.get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    model = RNN()
    fit_model(model, train_data, test_data)

