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
import keras.callbacks
import os, random
from UCF_utils import video_generator, get_data_list

N_CLASSES = 101
IMSIZE = (216, 216)
SequenceLength = 10
BatchSize = 2


def load_model():
    # use simple CNN structure
    in_shape = (SequenceLength, IMSIZE[0], IMSIZE[1], 3)
    model = Sequential()
    model.add(ConvLSTM2D(32, kernel_size=(7, 7), padding='valid', return_sequences=True, input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(ConvLSTM2D(64, kernel_size=(5, 5), padding='valid', return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
    model.add(Activation('relu'))
    model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
    model.add(Activation('relu'))
    model.add(ConvLSTM2D(96, kernel_size=(3, 3), padding='valid', return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dense(320))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    out_shape = model.output_shape
    # print('====Model shape: ', out_shape)
    model.add(Reshape((SequenceLength, out_shape[2] * out_shape[3] * out_shape[4])))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # model structure summary
    print(model.summary())

    return model


def fit_model(model, train_data, test_data):
    try:
        if os.path.exists('LRCN_keras.h5'):
            model.load_weights('LRCN_keras.h5')
            print('Load weights')
        train_generator = video_generator(train_data, BatchSize, seq_len=SequenceLength, img_size=IMSIZE,
                                          num_classes=101)
        test_generator = video_generator(test_data, BatchSize, seq_len=SequenceLength, img_size=IMSIZE, num_classes=101)
        print('Start fitting model')
        checkpointer = keras.callbacks.ModelCheckpoint('LRCN_weights.h5', save_weights_only=True)
        model.fit_generator(
            train_generator,
            steps_per_epoch=4500,
            epochs=20,
            validation_data=test_generator,
            validation_steps=300,
            verbose=2,
            callbacks=[checkpointer]
        )
        model.save_weights('LRCN_keras.h5')
    except KeyboardInterrupt:
        print('Training time:')


if __name__ == '__main__':
    data_dir = '/home/changan/ActionRocognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed')

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    model = load_model()
    fit_model(model, train_data, test_data)
