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
from UCF_utils import video_image_generator
from resnet50 import ResNet50
from keras.optimizers import SGD
from inception_v3 import InceptionV3

N_CLASSES = 101
IMSIZE = (216, 216, 3)
SequenceLength = 10
BatchSize = 30


def CNN(include_top=True):
    # use simple CNN structure
    model = Sequential()
    model.add(Convolution2D(96, kernel_size=(7, 7), strides=(2, 2), input_shape=IMSIZE, data_format='channels_last'))
    print('Output shape:', model.output_shape)
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Convolution2D(256, kernel_size=(5, 5), strides=(2, 2), data_format='channels_last'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    print('Output shape:', model.output_shape)

    model.add(Convolution2D(512, kernel_size=(3, 3), data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, kernel_size=(3, 3), data_format='channels_last'))
    model.add(Activation('relu'))
    print('Output shape:', model.output_shape)
    model.add(Convolution2D(512, kernel_size=(3, 3), data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
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


def fit_model(model, train_data, test_data):
    weights_dir = 'Inception_weights.h5'
    if os.path.exists(weights_dir):
        model.load_weights(weights_dir)
    try:
        train_generator = video_image_generator(train_data, BatchSize, seq_len=SequenceLength, img_size=IMSIZE,
                                                num_classes=101)
        test_generator = video_image_generator(test_data, BatchSize, seq_len=SequenceLength, img_size=IMSIZE,
                                               num_classes=101)
        print('Start fitting model')
        checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
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


def load_model():
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=IMSIZE)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    data_dir = '/home/changan/ActionRocognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed')

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    model = load_model()
    fit_model(model, train_data, test_data)

