from keras.models import Sequential
from keras.layers import Reshape
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Concatenate, ReLU, DepthwiseConv2D, add
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.optimizers import Adam , SGD
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.regularizers import l2

from FSM import fsm


def conv2d_bn_relu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def depth_conv_bn_relu(input, filters, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1),
                   initializer='he_normal', regularizer=l2(1e-5)):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding,
                        depthwise_initializer=initializer, use_bias=False, depthwise_regularizer=regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding=padding,
               kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def x_block(x, channels):
    res = conv2d_bn_relu(x, filters=channels, kernel_size=(1, 1))
    x = depth_conv_bn_relu(x, filters=channels, kernel_size=(3, 3))
    x = depth_conv_bn_relu(x, filters=channels, kernel_size=(3, 3))
    x = depth_conv_bn_relu(x, filters=channels, kernel_size=(3, 3))
    x = add([x, res])
    return x
