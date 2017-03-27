from finetuned_resnet import finetuned_resnet
from temporal_CNN import temporal_CNN
from keras.layers.merge import Average
from keras.models import Model


def two_stream_model():
    '''
    The simple two-stream model, it simply takes an average on the outputs of two streams and regards it as
    the final output
    :return: The two stream model that fuses the output of spatial and temporal streams
    '''
    # the models of different stream
    spatial_stream = finetuned_resnet(include_top=True, weights_dir='')
    temporal_stream = temporal_CNN(include_top=True, input_shape=(216, 216, 18), weights_dir='', classes=101)

    # freeze all weights, the two models have been trained separately
    for layer in spatial_stream.layers:
        layer.trainable = False
    for layer in temporal_stream.layers:
        layer.trainable = False

    # extract their output
    spatial_output = spatial_stream.output
    temporal_output = temporal_stream.output

    fused_output = Average(name='fusion_layer')([spatial_output, temporal_output])

    model = Model(inputs=[spatial_stream.input, temporal_stream.input], outputs=fused_output, name='two_stream')

    return model


if __name__ == '__main__':
    model = two_stream_model()
    print(model.summary())
