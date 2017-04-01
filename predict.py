import os
import numpy as np
from utils.UCF_utils import two_stream3_generator, two_stream18_generator
from models.two_stream import two_stream_model

N_CLASSES = 101
BatchSize = 32


def predict_two_stream3_test():
    spatial_weights_dir = '/home/changan/ActionRecognition/models/finetuned_resnet_RGB_65.h5'
    temporal_weights_dir = '/home/changan/ActionRecognition/models/finetuned_resnet_flow.h5'
    model = two_stream_model(spatial_weights_dir=spatial_weights_dir, temporal_weights_dir=temporal_weights_dir)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    print('Start to predict two stream model')
    data_dir = '/home/changan/ActionRecognition/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    test_list = os.path.join(list_dir, 'testlist.txt')
    frames_dir = '/home/changan/ActionRecognition/data/frames'
    flow_images_dir = '/home/changan/ActionRecognition/data/flow_images'
    input_shape = (216, 216, 3)
    generator = two_stream3_generator(test_list, frames_dir, flow_images_dir, 1, input_shape, N_CLASSES,
                                      mean_sub=True, normalization=True, random_crop=False, horizontal_flip=False)
    steps = 300
    correct_num = 0
    for i in range(steps):
        x, y = next(generator)
        prediction = model.predict(x)
        if y[0][np.argmax(prediction)] == 1:
            correct_num += 1
    print('test accuracy on', steps, 'examples is', float(correct_num)/steps)


def predict_two_stream18_test():
    spatial_weights_dir = '/home/changan/ActionRecognition/models/finetuned_resnet_RGB_65.h5'
    temporal_weights_dir = '/home/changan/ActionRecognition/models/temporal_cnn_42.h5'
    model = two_stream_model(spatial_weights_dir=spatial_weights_dir, temporal_weights_dir=temporal_weights_dir)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    print('Start to predict two stream model')
    data_dir = '/home/changan/ActionRecognition/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    test_list = os.path.join(list_dir, 'testlist.txt')
    frames_dir = '/home/changan/ActionRecognition/data/UCF-Preprocessed-OF/test'
    flow_dir = '/home/changan/ActionRecognition/data/OF_data/test'
    N_CLASSES = 101
    spatial_shape = (216, 216, 3)
    temporal_shape = (216, 216, 18)
    generator = two_stream18_generator(test_list, frames_dir, flow_dir, 1,
                                       spatial_shape, temporal_shape, N_CLASSES)
    steps = 3000
    correct_num = 0
    for i in range(steps):
        x, y = next(generator)
        prediction = model.predict(x)
        if y[0][np.argmax(prediction)] == 1:
            correct_num += 1
    print('test accuracy on', steps, 'examples is', float(correct_num)/steps)


if __name__ == '__main__':
    data_dir = '/home/changan/ActionRecognition/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'OF_data')
    weights_dir = '/home/changan/ActionRecognition/models'


    # fine tune resnet50
    # train_data = os.path.join(list_dir, 'trainlist.txt')
    # test_data = os.path.join(list_dir, 'testlist.txt')
    # input_shape = (216, 216, 3)
    # weights_dir = os.path.join(weights_dir, 'finetuned_resnet_flow.h5')
    # model = finetuned_resnet(include_top=True, weights_dir=weights_dir)
    # fit_model(model, train_data, test_data, weights_dir, input_shape)

    # train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    #
    # training CNN using optical flow as input
    # weights_dir = os.path.join(weights_dir, 'temporal_cnn.h5')
    # input_shape = (216, 216, 18)
    # model = temporal_CNN(input_shape, N_CLASSES, weights_dir, include_top=True)
    # fit_model(model, train_data, test_data, weights_dir, input_shape, optical_flow=True)

    # predict test dataset using two stream
    predict_two_stream18_test()
