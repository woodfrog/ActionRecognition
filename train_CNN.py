import os, time
import keras.callbacks
from utils.UCF_utils import image_generator, get_data_list, sequence_generator
from models.finetuned_resnet import finetuned_resnet
from models.temporal_CNN import temporal_CNN
from utils.UCF_preprocessing import preprocessing
from utils.OF_utils import optical_flow_prep

N_CLASSES = 101
BatchSize = 30


def regenerate_data():
    start_time = time.time()
    sequence_length = 10
    image_size = (216, 216, 3)

    data_dir = '/home/changan/ActionRecognition/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    UCF_dir = os.path.join(data_dir, 'UCF-101')
    dest_dir = os.path.join(data_dir, 'UCF-Preprocessed-OF')

    # generate sequence for optical flow
    preprocessing(list_dir, UCF_dir, dest_dir, sequence_length, image_size, overwrite=True, normalization=False,
                  mean_subtraction=False, horizontal_flip=False, random_crop=True, consistent=True, continuous_seq=True)

    # compute optical flow data
    src_dir = '/home/changan/ActionRecognition/data/UCF-Preprocessed-OF'
    dest_dir = '/home/changan/ActionRecognition/data/OF_data'
    optical_flow_prep(src_dir, dest_dir, mean_sub=True, overwrite=True)

    elapsed_time = time.time() - start_time
    print('Regenerating data takes:', int(elapsed_time / 60), 'minutes')


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
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=2,
                                                      mode='auto')
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        while True:
            model.fit_generator(
                train_generator,
                steps_per_epoch=300,
                epochs=50,
                validation_data=test_generator,
                validation_steps=100,
                verbose=2,
                callbacks=[checkpointer, earlystopping]
            )
            regenerate_data()

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
