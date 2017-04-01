import numpy as np
import cv2
import os
import warnings
from collections import OrderedDict
import shutil


def optical_flow_prep(src_dir, dest_dir, mean_sub=True, overwrite=False):
    train_dir = os.path.join(src_dir, 'train')
    test_dir = os.path.join(src_dir, 'test')

    # create dest directory
    if os.path.exists(dest_dir):
        if overwrite:
            shutil.rmtree(dest_dir)
        else:
            raise IOError(dest_dir + ' already exists')
    os.mkdir(dest_dir)
    print(dest_dir, 'created')

    # create directory for training data
    dest_train_dir = os.path.join(dest_dir, 'train')
    if os.path.exists(dest_train_dir):
        print(dest_train_dir, 'already exists')
    else:
        os.mkdir(dest_train_dir)
        print(dest_train_dir, 'created')

    # create directory for testing data
    dest_test_dir = os.path.join(dest_dir, 'test')
    if os.path.exists(dest_test_dir):
        print(dest_test_dir, 'already exists')
    else:
        os.mkdir(dest_test_dir)
        print(dest_test_dir, 'created')

    dir_mapping = OrderedDict(
        [(train_dir, dest_train_dir), (test_dir, dest_test_dir)])  # the mapping between source and dest

    print('Start computing optical flows ...')
    for dir, dest_dir in dir_mapping.items():
        print('Processing data in {}'.format(dir))
        for index, class_name in enumerate(os.listdir(dir)):  # run through every class of video
            class_dir = os.path.join(dir, class_name)
            dest_class_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(dest_class_dir):
                os.mkdir(dest_class_dir)
                # print(dest_class_dir, 'created')
            for filename in os.listdir(class_dir):  # process videos one by one
                file_dir = os.path.join(class_dir, filename)
                frames = np.load(file_dir)
                # note: store the final processed data with type of float16 to save storage
                processed_data = stack_optical_flow(frames, mean_sub).astype(np.float16)
                dest_file_dir = os.path.join(dest_class_dir, filename)
                np.save(dest_file_dir, processed_data)
            # print('No.{} class {} finished, data saved in {}'.format(index, class_name, dest_class_dir))
    print('Finish computing optical flows')


def stack_optical_flow(frames, mean_sub=False):
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
        warnings.warn('Warning! The data type has been changed to np.float32 for graylevel conversion...')
    frame_shape = frames.shape[1:-1]  # e.g. frames.shape is (10, 216, 216, 3)
    num_sequences = frames.shape[0]
    output_shape = frame_shape + (2 * (num_sequences - 1),)  # stacked_optical_flow.shape is (216, 216, 18)
    flows = np.ndarray(shape=output_shape)

    for i in range(num_sequences - 1):
        prev_frame = frames[i]
        next_frame = frames[i + 1]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = _calc_optical_flow(prev_gray, next_gray)
        flows[:, :, 2 * i:2 * i + 2] = flow

    if mean_sub:
        flows_x = flows[:, :, 0:2 * (num_sequences - 1):2]
        flows_y = flows[:, :, 1:2 * (num_sequences - 1):2]
        mean_x = np.mean(flows_x, axis=2)
        mean_y = np.mean(flows_y, axis=2)
        for i in range(2 * (num_sequences - 1)):
            flows[:, :, i] = flows[:, :, i] - mean_x if i % 2 == 0 else flows[:, :, i] - mean_y

    return flows


def _calc_optical_flow(prev, next_):
    flow = cv2.calcOpticalFlowFarneback(prev, next_, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    return flow


if __name__ == '__main__':
    src_dir = '/home/changan/ActionRecognition/data/UCF-Preprocessed-OF'
    dest_dir = '/home/changan/ActionRecognition/data/OF_data'
    optical_flow_prep(src_dir, dest_dir, mean_sub=True, overwrite=True)
