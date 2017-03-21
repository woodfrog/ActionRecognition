from inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout, Flatten
from UCF_utils import get_data_list, video_image_generator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import os

N_CLASSES = 101
batch_size = 30
SequenceLength = 10
IMSIZE = (216, 216, 3)


def inception_finetune_UCF():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=IMSIZE)
    print('Inception_v3 loaded')

    # freeze the top layers
    # for layer in base_model.layers[:172]:
    #     layer.trainable = False
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='relu')(x)
    predictions = Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    data_dir = '/home/changan/ActionRocognition_rnn/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')
    video_dir = os.path.join(data_dir, 'UCF-Preprocessed')

    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    print('Train data size: ', len(train_data))
    print('Test data size: ', len(test_data))

    train_generator = video_image_generator(train_data, batch_size, seq_len=SequenceLength, img_size=IMSIZE,
                                            num_classes=101)
    test_generator = video_image_generator(test_data, batch_size, seq_len=SequenceLength, img_size=IMSIZE,
                                           num_classes=101)
    weights_dir = 'inception_finetune.h5'
    if os.path.exists(weights_dir):
        model.load_weights(weights_dir)
        print('weights loaded')
    checkpointer = ModelCheckpoint(weights_dir, save_weights_only=True)
    model.fit_generator(
        train_generator,
        steps_per_epoch=30,
        epochs=200,
        validation_data=test_generator,
        validation_steps=100,
        verbose=2,
        callbacks=[checkpointer]
    )

if __name__ == '__main__':
    inception_finetune_UCF()