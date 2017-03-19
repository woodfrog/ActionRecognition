import os
from utils.UCF_utils import video_generator, get_data_list
import keras.callbacks
from models import RNN

N_CLASSES = 101
IMSIZE = (216, 216)
SequenceLength = 10
BatchSize = 30
CNN_output = 154880


def fit_RNN_model(model, train_data, test_data):
    weights_dir = 'RNN_weights.h5'
    try:
        if os.path.exists(weights_dir):
            model.load_weights(weights_dir)
            print('Load weights')
        train_generator = video_generator(train_data, BatchSize, SequenceLength, CNN_output, N_CLASSES)
        test_generator = video_generator(test_data, BatchSize, SequenceLength, CNN_output, N_CLASSES)
        print('Start fitting model')
        checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_weights_only=True)
        model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=200,
            validation_data=test_generator,
            validation_steps=50,
            verbose=2,
            callbacks=[checkpointer]
        )
    except KeyboardInterrupt:
        print('Training time:')


if __name__ == '__main__':
    data_dir = '/home/changan/ActionRocognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed')

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    RNN_model = RNN.RNN()
    fit_RNN_model(RNN_model, train_data, test_data)
