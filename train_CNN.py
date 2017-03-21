import os
import keras.callbacks
from utils.UCF_utils import image_generator, get_data_list
from models.finetuned_resnet import finetuned_resnet

N_CLASSES = 101
IMSIZE = (216, 216, 3)
SequenceLength = 10
BatchSize = 30


def fit_model(model, train_data, test_data, weights_dir):
    try:
        train_generator = image_generator(train_data, BatchSize, SequenceLength, IMSIZE, N_CLASSES)
        test_generator = image_generator(test_data, BatchSize, SequenceLength, IMSIZE, N_CLASSES)
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
    data_dir = '/home/changan/ActionRecognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed')
    weights_dir = '/home/changan/ActionRecognition_rnn/models'

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)

    resnet_weights_dir = os.path.join(weights_dir, 'finetuned_resnet.h5')
    resnet_model = finetuned_resnet(include_top=True, weights_dir=resnet_weights_dir)
    fit_model(resnet_model, train_data, test_data, resnet_weights_dir)
