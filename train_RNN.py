import os
from utils.UCF_utils import sequence_generator, get_data_list
import keras.callbacks
from models import RNN
from keras.optimizers import SGD, Adam

N_CLASSES = 101
BatchSize = 30


def fit_model(model, train_data, test_data, weights_dir, input_shape):
    try:
        if os.path.exists(weights_dir):
            model.load_weights(weights_dir)
            print('Load weights')
        train_generator = sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
        test_generator = sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)
        print('Start fitting model')
        checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
        adam = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit_generator(
            train_generator,
            steps_per_epoch=300,
            epochs=200,
            validation_data=test_generator,
            validation_steps=100,
            verbose=2,
            callbacks=[checkpointer]
        )
    except KeyboardInterrupt:
        print('Training time:')


if __name__ == '__main__':
    data_dir = '/home/changan/ActionRecognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'CNN_Predicted')
    weights_dir = '/home/changan/ActionRecognition_rnn/models'

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    CNN_output = 1024
    input_shape = (10, CNN_output)
    rnn_weights_dir = os.path.join(weights_dir, 'rnn.h5')
    RNN_model = RNN.RNN(rnn_weights_dir, CNN_output)
    fit_model(RNN_model, train_data, test_data, rnn_weights_dir, input_shape)
