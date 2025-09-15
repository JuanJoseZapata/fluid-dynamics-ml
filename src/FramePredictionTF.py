import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Conv2DTranspose, concat
from tensorflow.keras.models import Model

def create_cnn_model(
    input_shape,
    filters_list,
    kernel_size=(3, 3)
):
    """
    Creates a TensorFlow Keras CNN model for predicting the next frame from a single input frame.
    
    Parameters:
    ----------
    input_shape: tuple
        Shape of the input frame (height, width, channels).
    filters_list: list of int
        A list specifying the number of filters for each convolutional layer.
    kernel_size: tuple
        Size of the convolutional kernel.
        
    Returns:
    -------
    tf.keras.Model
        The compiled Keras model.
    """
    # The input layer now takes a single frame.
    inputs = Input(shape=input_shape)
    x = inputs

    # Stack the convolutional layers as defined by filters_list.
    for filters in filters_list:
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu'
        )(x)
        x = BatchNormalization()(x)

    # Final convolutional layer to produce the output frame.
    # The number of filters must match the number of channels in the input frame.
    outputs = Conv2D(
        filters=input_shape[-1],
        kernel_size=kernel_size,
        padding='same',
        activation='sigmoid'
    )(x)

    # Create and return the model.
    model = Model(inputs=inputs, outputs=outputs)

    return model


def create_unet_model(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    c1 = BatchNormalization()(c1)
    c2 = Conv2D(128, (3, 3), padding='same', activation='relu', strides=(2, 2))(c1)
    c2 = BatchNormalization()(c2)
    c3 = Conv2D(256, (3, 3), padding='same', activation='relu', strides=(2, 2))(c2)
    c3 = BatchNormalization()(c3)
    
    # Decoder
    up1 = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2), activation='relu')(c3)
    up1 = concat([up1, c2]) # Skip connection
    up1 = BatchNormalization()(up1)
    
    up2 = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2), activation='relu')(up1)
    up2 = concat([up2, c1]) # Skip connection
    up2 = BatchNormalization()(up2)
    
    # Output layer
    outputs = Conv2D(input_shape[-1], (3, 3), padding='same', activation='sigmoid')(up2)

    model = Model(inputs=inputs, outputs=outputs)
    return model