import os
import keras.callbacks
from utils.UCF_utils import image_generator, get_data_list, sequence_generator
from models.finetuned_resnet import finetuned_resnet
from models.temporal_CNN import temporal_CNN

N_CLASSES = 101
BatchSize = 30


def fit_model(model, train_data, test_data, weights_dir, input_shape, optical_flow=False):
    try:
        if optical_flow:
            train_generator = sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
            test_generator = sequence_generator(test_data, BatchSize, input_shape, N_CLASSES)
        else:
            train_generator = image_generator(train_data, BatchSize, input_shape, N_CLASSES)
            test_generator = image_generator(test_data, BatchSize, input_shape, N_CLASSES)
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


if __name__ == '__main__':
    data_dir = '/home/changan/ActionRecognition/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'OF_data')
    weights_dir = '/home/changan/ActionRecognition/models'

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)

    # input_shape = (10, 216, 216, 3)
    # resnet_weights_dir = os.path.join(weights_dir, 'finetuned_resnet.h5')
    # resnet_model = finetuned_resnet(include_top=True, weights_dir=resnet_weights_dir)
    # fit_model(resnet_model, train_data, test_data, input_shape=IMSIZE, resnet_weights_dir)

    weights_dir = os.path.join(weights_dir, 'temporal_cnn.h5')
    input_shape = (216, 216, 18)
    model = temporal_CNN(input_shape, N_CLASSES, weights_dir, include_top=True)
    fit_model(model, train_data, test_data, weights_dir, input_shape, optical_flow=True)
