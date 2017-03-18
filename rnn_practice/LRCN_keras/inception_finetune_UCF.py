from inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, GlobalAveragePooling2D
from UCF_utils import get_data_list, video_image_generator
import os

N_CLASSES = 101
batch_size = 64
SequenceLength = 10
IMSIZE = (216, 216)


def inception_finetune_UCF():
    base_model = InceptionV3(weights='imagenet', include_top=False)
    print('Inception_v3 loaded')

    # freeze the top layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    data_dir = '/Users/cjc/cv/ActionRecognition_rnn/data/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed')

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    train_generator = video_image_generator(train_data, batch_size, seq_len=SequenceLength, img_size=IMSIZE,
                                            num_classes=101)
    test_generator = video_image_generator(test_data, batch_size, seq_len=SequenceLength, img_size=IMSIZE,
                                           num_classes=101)

    model.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        nb_epoch=20,
        validation_data=test_generator,
        nb_val_samples=300,
        verbose=2,
    )

if __name__ == '__main__':
    inception_finetune_UCF()