'''
Implementing simple LRCN structure in paper https://arxiv.org/pdf/1411.4389v3.pdf
'''

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Reshape, Permute, Activation
from keras.layers import Convolution2D, MaxPooling3D, ConvLSTM2D
from keras.layers.recurrent import LSTM
import keras.callbacks
import os, random

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
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))  # channel last
    model.add(ConvLSTM2D(64, kernel_size=(5, 5), padding='valid', return_sequences=True))
    model.add(Activation('relu'))
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


def video_generator(data, batch_size):
    x_shape = (batch_size, SequenceLength, IMSIZE[0], IMSIZE[1], 3)
    y_shape = (batch_size, 101)
    index = 0
    while True:
        batch_x = np.ndarray(x_shape)
        batch_y = np.zeros(y_shape)
        for i in range(batch_size):
            index = (index + 1) % len(data)
            clip_dir, clip_class = data[index]
            batch_y[i, clip_class - 1] = 1
            clip_dir = os.path.splitext(clip_dir)[0] + '.npy'
            while not os.path.exists(clip_dir):
                index = (index + 1) % len(data)
                clip_dir, class_idx = data[index]
            clip_data = np.load(clip_dir)
            batch_x[i] = clip_data
        yield batch_x, batch_y


def fit_model(model, train_data, test_data):
    try:
        if os.path.exists('LRCN_keras.h5'):
            model.load_weights('LRCN_keras.h5')
            print('Load weights')
        train_generator = video_generator(train_data, BatchSize)
        test_generator = video_generator(test_data, BatchSize)
        print('Start fitting model')
        weigts_dir = './weights'
        model.fit_generator(
            train_generator,
            samples_per_epoch=9000,
            nb_epoch=2000,
            validation_data=test_generator,
            nb_val_samples=300,
            verbose=2,
            callbacks=keras.callbacks.ModelCheckpoint('LRCN_weights.h5', save_weights_only=True)
        )
        model.save_weights('LRCN_keras.h5')
    except KeyboardInterrupt:
        print('Training time:')


def get_data_list(list_dir, video_dir):
    train_dir = os.path.join(video_dir, 'train')
    test_dir = os.path.join(video_dir, 'test')
    testlisttxt = 'testlist01.txt'
    trainlisttxt = 'trainlist01.txt'

    testlist = []
    txt_path = os.path.join(list_dir, testlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            testlist.append(line.rstrip())

    trainlist = []
    txt_path = os.path.join(list_dir, trainlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            trainlist.append(line[:line.rfind(' ')])

    class_index = dict()
    class_dir = os.path.join(list_dir, 'classInd.txt')
    with open(class_dir) as fo:
        for line in fo:
            class_number, class_name = line.split()
            class_number = int(class_number)
            class_index[class_name] = class_number

    train_data = []
    for i, clip in enumerate(trainlist):
        clip_class = os.path.dirname(clip)
        dst_dir = os.path.join(train_dir, clip)
        train_data.append((dst_dir, class_index[clip_class]))
    random.shuffle(train_data)

    test_data = []
    for i, clip in enumerate(testlist):
        clip_class = os.path.dirname(clip)
        dst_dir = os.path.join(test_dir, clip)
        test_data.append((dst_dir, class_index[clip_class]))
    random.shuffle(train_data)

    return train_data, test_data, class_index


if __name__ == '__main__':
    data_dir = 'data_dir'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed')

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    model = load_model()
    fit_model(model, train_data, test_data)

