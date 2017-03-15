import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Reshape, Permute, Activation
from keras.layers import Convolution2D, MaxPooling3D, ConvLSTM2D
from keras.layers.recurrent import LSTM
#import inception_v3 as inception
import os, random, cv2


N_CLASSES = 101
IMSIZE = (216, 216)
SequenceLength = 10
BatchSize = 1

def load_model():
    '''
    # Start with an Inception V3 model, not including the final softmax layer.
    base_model = inception.InceptionV3(weights='imagenet', input_tensor=())
    print('Loaded Inception model')

    # Turn off training on base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add on new fully connected layers for the output classes.
    x = LSTM(32, name='lstm1', return_sequences=True)(base_model.get_layer('flatten').output)
    x = LSTM(32, name='lstm2', return_sequences=False)(x)
    predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=predictions)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    '''
    # use simple CNN structure
    in_shape = (SequenceLength, IMSIZE[0], IMSIZE[1], 3)
    model = Sequential()
    model.add(ConvLSTM2D(2, 3, 3, border_mode='valid', return_sequences=True, input_shape=in_shape))
    model.add(Activation('relu'))
    # model.add(ConvLSTM2D(2, 3, 3, border_mode='valid', return_sequences=True))
    # model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 4, 4)))# channel last
    model.add(Dropout(0.25))

    # model.add(ConvLSTM2D(2, 3, 3, border_mode='valid', return_sequences=True))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    # model.add(Dropout(0.25))
    # model.add(ConvLSTM2D(2, 3, 3, border_mode='valid', return_sequences=True))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(ConvLSTM2D(2, 3, 3, border_mode='valid', return_sequences=True))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    # model.add(Dropout(0.25))
    # model.add(ConvLSTM2D(2, 3, 3, border_mode='valid', return_sequences=True))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    # model.add(Dropout(0.25))

    out_shape = model.output_shape
    print('====Model shape: ', out_shape)
    # model.add(Flatten())
    model.add(Reshape((SequenceLength, out_shape[2]*out_shape[3]*out_shape[4])))
    # model.add(LSTM(1, return_sequences=True))
    model.add(LSTM(1, return_sequences=False))
    model.add(Dense(N_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Show some debug output
    print (model.summary())

    return model

def video_generator(data, batch_size):
    x_shape = (batch_size, SequenceLength, IMSIZE[0], IMSIZE[1], 3)
    y_shape = (batch_size, 101)
    index = 0
    while(True):
        batch_x = np.ndarray(x_shape)
        batch_y = np.zeros(y_shape)
        for i in range(batch_size):
            index = (index + 1) % len(data)
            clip_dir, clip_class = data[index]
            batch_y[i, clip_class-1] = 1
            while not os.path.exists(clip_dir):
                index = (index+1) % len(data)
                clip_dir, class_idx = data[index]
            cap = cv2.VideoCapture(clip_dir)
            j = 0
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if ret:
                        batch_x[i][j] = frame
                    else:
                        break
            cap.release()
        yield batch_x, batch_y


def fit_model(model, train_data, test_data):
    try:
        train_generator = video_generator(train_data, BatchSize)
        test_generator = video_generator(test_data, BatchSize)
        model.fit_generator(
            train_generator,
            samples_per_epoch=1,
            nb_epoch=10,
            validation_data=test_generator,
            nb_val_samples=32,
            verbose=2,
        )
        model.save_weights('inception_lstm_weights.h5')
    except KeyboardInterrupt:
        print('Training time:')


def get_data_list(list_dir, video_dir):
    train_dir = os.path.join(video_dir, 'train')
    test_dir = os.path.join(video_dir, 'test')
    testlisttxt = ['testlist01.txt', 'testlist02.txt', 'testlist03.txt']
    trainlisttxt = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt']

    testlist = []
    for fname in testlisttxt:
        txt_path = os.path.join(list_dir, fname)
        with open(txt_path) as fo:
            for line in fo:
                testlist.append(line)

    trainlist = []
    for fname in trainlisttxt:
        txt_path = os.path.join(list_dir, fname)
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
    data_dir = '/home/changan/ActionRocognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed')

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    model = load_model()
    fit_model(model, train_data, test_data)



'''
img_path = '/home/changan/419/sport3/validation/hockey/img_2997.jpg'
img = image.load_img(img_path, target_size=IMSIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = inception.preprocess_input(x)

preds = model.predict(x)
print('Predicted:', preds)
'''