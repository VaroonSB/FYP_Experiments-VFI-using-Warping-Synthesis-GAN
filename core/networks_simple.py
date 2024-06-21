# -*- coding: utf-8 -*-
# Model architectures
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, concatenate, Activation
from tensorflow.keras.models import Model
from core.activation import Swish

def UNET(input_shape, activation = 'relu'):
    '''
    Modified U-net architecture with some extra blocks (+Conv and BatchNorm layer)

    # Arguments
    input shape: Input size (a tuple) of the network. Format: (2*channels, height, width)
    because it will get 2 images.

    # Output
    model: The created U-net network with output_size = (channels, height, width)
    '''

    inputs = Input(input_shape)

    conv1 = Conv2D(32, (3, 3), activation=activation, padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    conv2 = Conv2D(64, (3, 3), activation=activation, padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(128, (3, 3), activation=activation, padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(256, (3, 3), activation=activation, padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)


    conv5 = Conv2D(512, (3, 3), activation=activation, padding='same')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)


    conv5_2 = Conv2D(512, (3, 3), activation=activation, padding='same')(pool5)    
    up5_2 = concatenate([UpSampling2D(size=(2, 2))(conv5_2), conv5], axis=1)


    conv6_2 = Conv2D(512, (3, 3), activation=activation, padding='same')(up5_2)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv4], axis=1)


    conv6 = Conv2D(256, (3, 3), activation=activation, padding='same')(up6)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    
    
    conv7 = Conv2D(128, (3, 3), activation=activation, padding='same')(up7)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    
    conv8 = Conv2D(64, (3, 3), activation=activation, padding='same')(up8)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    
    
    conv9 = Conv2D(32, (3, 3), activation=activation, padding='same')(up9)
    
    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model