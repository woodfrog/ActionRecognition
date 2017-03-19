'''
Implementing LRCN structure
References:
    http://cs231n.stanford.edu/reports2016/221_Report.pdf
    https://arxiv.org/pdf/1411.4389v3.pdf
'''

import os
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Reshape, Permute, Activation
from keras.layers import Convolution2D, MaxPooling3D, MaxPooling2D, ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers

N_CLASSES = 101
IMSIZE = (216, 216)
SequenceLength = 10
BatchSize = 30


def CNN(include_top=True):
    # use simple CNN structure
    in_shape = (IMSIZE[0], IMSIZE[1], 3)
    model = Sequential()
    model.add(Convolution2D(64, kernel_size=(7, 7), input_shape=in_shape, data_format='channels_last'))
    print('Output shape:', model.output_shape)
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Convolution2D(96, kernel_size=(5, 5), data_format='channels_last'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    print('Output shape:', model.output_shape)

    model.add(Convolution2D(128, kernel_size=(3, 3), data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, kernel_size=(3, 3), data_format='channels_last'))
    model.add(Activation('relu'))
    print('Output shape:', model.output_shape)
    model.add(Convolution2D(196, kernel_size=(3, 3), data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dense(320))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    print('Output shape:', model.output_shape)

    if include_top:
        model.add(Dense(N_CLASSES, activation='softmax',
                        kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    weights_dir = 'CNN_weights.h5'
    if os.path.exists(weights_dir):
        model.load_weights(weights_dir)
        print('===Load weights')

    # model structure summary
    print(model.summary())

    return model
