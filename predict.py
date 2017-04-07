import os
import numpy as np
from .utils.UCF_utils import two_stream3_generator, two_stream18_generator
from .models.two_stream import two_stream_model
from .utils.OF_utils import stack_optical_flow
import cv2
import random
from scipy.misc import imresize
from keras.applications.imagenet_utils import preprocess_input

N_CLASSES = 101
BatchSize = 32
IMSIZE = (216, 216, 3)


def predict_two_stream3_test():
    spatial_weights_dir = '/home/changan/ActionRecognition/models/finetuned_resnet_RGB_65.h5'
    temporal_weights_dir = '/home/changan/ActionRecognition/models/finetuned_resnet_flow.h5'
    model = two_stream_model(spatial_weights_dir=spatial_weights_dir, temporal_weights_dir=temporal_weights_dir)

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
    print('test accuracy on', steps, 'examples is', float(correct_num) / steps)


def predict_two_stream18_test():
    spatial_weights_dir = '/home/changan/ActionRecognition/models/finetuned_resnet_RGB_65.h5'
    temporal_weights_dir = '/home/changan/ActionRecognition/models/temporal_cnn_42.h5'
    model = two_stream_model(spatial_weights_dir=spatial_weights_dir, temporal_weights_dir=temporal_weights_dir)

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
    print('test accuracy on', steps, 'examples is', float(correct_num) / steps)


def predict_single_video(model, video_path, top_num):
    cap = cv2.VideoCapture(video_path)
    video = list()
    while cap.isOpened():
        succ, frame = cap.read()
        if not succ:
            break
        # append frame that is not all zeros
        if frame.any():
            frame = imresize(frame, IMSIZE)
            video.append(frame)

    frames = _pick_frames(video, num_frame=10)

    of_input = stack_optical_flow(frames, mean_sub=True)
    of_input = np.expand_dims(of_input, axis=0)

    single_frame = frames[random.randint(0, len(frames) - 1)]
    single_frame = np.expand_dims(single_frame, axis=0)
    single_frame = preprocess_single_frame(single_frame)

    two_stream_input = [single_frame, of_input]
    preds = model.predict(two_stream_input)
    return decode_prediction(preds, top=top_num)


def _pick_frames(video_sequence, num_frame):
    i = 0
    if num_frame > len(video_sequence):
        raise ValueError('Input video is too short and cannot provide enough frames for optical flow...')
    frames_shape = (num_frame,) + video_sequence[0].shape
    start = random.randint(0, len(video_sequence) - num_frame)
    result = np.zeros(frames_shape)

    for i in range(num_frame):
        result[i] = video_sequence[i + start]

    return result


def decode_prediction(preds, top=3):
    index_dir = '/home/changan/ActionRecognition/data/ucfTrainTestlist/classInd.txt'
    class_dict = dict()
    with open(index_dir) as fo:
        for line in fo:
            class_index, class_name = line.split()
            class_dict[int(class_index)-1] = class_name
    top = np.argsort(preds)[0][-top:][::-1]
    print(preds)
    print(top)
    return [(class_dict[x], preds[0][x]) for x in top]


def preprocess_single_frame(frame):
    frame = preprocess_input(frame)
    frame /= 255
    return frame


if __name__ == '__main__':
    data_dir = '/home/changan/ActionRecognition/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'OF_data')
    weights_dir = '/home/changan/ActionRecognition/models'

    # predict overall testing set using two-stream model
    # predict_two_stream18_test()

    # predict single video
    # predict_single_video(video_path='/Users/cjc/cv/ActionRecognition/data/v_BabyCrawling_g01_c01.mp4')
