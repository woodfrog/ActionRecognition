'''
Implementing LRCN structure
References:
    http://cs231n.stanford.edu/reports2016/221_Report.pdf
    https://arxiv.org/pdf/1411.4389v3.pdf
'''

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Reshape, Permute, Activation
from keras.layers import Convolution2D, MaxPooling3D, MaxPooling2D, ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import keras.callbacks
import os, random

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


def video_generator(data, batch_size):
    x_shape = (batch_size, IMSIZE[0], IMSIZE[1], 3)
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
            batch_x[i] = clip_data[random.randrange(SequenceLength)]
        yield batch_x, batch_y


def fit_model(model, train_data, test_data):
    weights_dir = 'CNN_weights.h5'
    try:
        train_generator = video_generator(train_data, BatchSize)
        test_generator = video_generator(test_data, BatchSize)
        print('Start fitting model')
        checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_weights_only=True)
        model.fit_generator(
            train_generator,
            steps_per_epoch=300,
            epochs=200,
            validation_data=test_generator,
            validation_steps=100,
            verbose=2,
            callbacks=[checkpointer]
        )
        model.save_weights(weights_dir)
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
    random.shuffle(test_data)

    return train_data, test_data, class_index


if __name__ == '__main__':
    data_dir = '/home/changan/ActionRocognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed')

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    model = CNN()
    fit_model(model, train_data, test_data)

