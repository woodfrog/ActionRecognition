import os
import numpy as np
from collections import OrderedDict


def model_processing(model, src_dir, dest_dir, timeseq_len):
    """
    :param model: the model for processing input data
    :param src_dir: the path of the directory containing numpy arrays of input data
    :param dest_dir: the path for saving processed data
    :param timeseq_len: the length of time sequence for videos
    """
    train_dir = os.path.join(src_dir, 'train')
    test_dir = os.path.join(src_dir, 'test')

    # create dest directory
    if os.path.exists(dest_dir):
        print(dest_dir, 'already exists')
    else:
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

    for dir, dest_dir in dir_mapping.items():
        print('Processing data in {}'.format(dir))
        for index, class_name in enumerate(os.listdir(dir)):  # run through every class
            class_dir = os.path.join(dir, class_name)
            dest_class_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(dest_class_dir):
                os.mkdir(dest_class_dir)
                print(dest_class_dir, 'created')
            for filename in os.listdir(class_dir):  # process files one by one
                file_dir = os.path.join(class_dir, filename)
                clip_data = np.load(file_dir)
                processed_data = model.predict(clip_data, batch_size=timeseq_len)
                dest_file_dir = os.path.join(dest_class_dir, filename)
                np.save(dest_file_dir, processed_data)
            print('No.{} class {} finished, data saved in {}'.format(index, class_name, dest_class_dir))