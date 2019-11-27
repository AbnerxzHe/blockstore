from keras.layers.convolutional import Conv3D
from keras.layers import Dropout, Input,UpSampling3D,concatenate,multiply,UpSampling3D,AveragePooling3D
from keras.layers import Flatten, Add
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.
Learn more or give us feedback
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
from keras.layers import Activation, Reshape, Lambda, dot, add, Input, BatchNormalization, ReLU, DepthwiseConv2D, Concatenate
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2

#-----------------------------------------------------------Conv--------------------------------------------------------


def Conv2d_bn_relu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
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


#-----------------------------------------------------------Res--------------------------------------------------------

#https://www.cnblogs.com/ariel-dreamland/p/10569968.html
#desnet18/34

def Conv2d_BN(x, nb_filter, kernel_size, strides, padding='same', name=None):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides,
               activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def Res_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,strides=strides, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides,
                             kernel_size=kernel_size)
        x = Dropout(0.2)(x)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

    
    
#psp





#FPN




#-----------------------------------------------------------GCN--------------------------------------------------------
#Global Convolutional Network
def GCN(x,k):
    x1 = x
    nb_filter=x.shape[-1]
    X1 =  Conv2D(nb_filter, kernel_size=(3,1), padding='same', strides=(k,k))(x)
    X1 =  Conv2D(nb_filter, kernel_size=(1,3), padding='same', strides=(k,k))(x)
    X2 =  Conv2D(nb_filter, kernel_size=(3,1), padding='same', strides=(k,k))(x)
    X2 =  Conv2D(nb_filter, kernel_size=(1,3), padding='same', strides=(k,k))(x)
    X3 =  Conv2D(nb_filter, kernel_size=(3,1), padding='same', strides=(k,k))(x)
    X3 =  Conv2D(nb_filter, kernel_size=(1,3), padding='same', strides=(k,k))(x)
    out = Add([x1,x2,x3],axis=-1)
    return out

#-----------------------------------------------------------BR--------------------------------------------------------
#Boundary Refinement
def BR(x):
    x1 = x
    nb_filter=x.shape[-1]
    x = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(x)
    x = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2))(x)
    out = Add([x1,x],axis=-1)
    return out

#-----------------------------------------------------------RCU--------------------------------------------------------
#RCU:Residual Conv Unit
def RCU(x):
    x1 = x
    x = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(x)
    x = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(x)
    out = Add([x1,x],axis=-1)
    return out

#-----------------------------------------------------------MRF--------------------------------------------------------
#Multi-resolution Fusion
def MRF(a,b):
    x1 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(a)
    x1 = UpSampling2D((2,2))(x1)
    x2 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(b)
    x2 = UpSampling2D((4,4))(x2)
    x3 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(b)
    x3 = UpSampling2D((8,8))(x2)
    x4 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(b)
    x4 = UpSampling2D((16,16))(x2)
    out = Add([x1,x2,x3,x4],axis=-1)
    return out


#-----------------------------------------------------------CRP--------------------------------------------------------
#Chained Residual Pooling
def CRP(x):
    x = Activation('ReLU')(x)
    x1 = MaxPooling(pool_size=(5,5))(x)
    x1 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(x1)
    x = Add([x,x1],axis=-1)
    x2 = MaxPooling2D(pool_size=(5,5))(x1)
    x2 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(x2)
    x = Add([x,x2],axis=-1)
    x3 = MaxPooling2D(pool_size=(5,5))(x2)
    x3 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(x3)
    x = Add([x,x3],axis=-1)
    return x


#-----------------------------------------------------------Decoder--------------------------------------------------------
def Dencoder(x,nb_filter,c):
    print(x.shape)  
#output shape 8倍
    x1 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(x)
    x2 = Conv2D(c, kernel_size=(3,3), padding='same', strides=(1,1),activation='relu')(x1)
   
    x1 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(x1)
    x3 = Conv2D(c, kernel_size=(3,3), padding='same', strides=(1,1),activation='relu')(x1)
    
    x1 = Conv2D(nb_filter, kernel_size=(3,3), padding='same', strides=(2,2),activation='relu')(x1)
    x4 = Conv2D(c, kernel_size=(3,3), padding='same', strides=(1,1),activation='relu')(x1)
    x4 = UpSampling2D(size=(2,2))(x4)
    
    x3 = concatenate([x3, x4],axis=-1)
    x3 = UpSampling2D(size=(2,2))(x3)
    x2 = concatenate([x2,x3],axis=-1)
    x2 = UpSampling2D(size=(2,2))(x2)
    
    x5 = Conv2D(c, kernel_size=(3,3), padding='same', strides=(1,1),activation='relu')(x)
#     print(x.shape)
    #x5 = multiply([x2,x5])
    x5 = concatenate([x2,x5],axis=-1)
    x6 = AveragePooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
#     print(x6.shape)
    x6 = Conv2D(c, kernel_size=(3,3), padding='same', strides=(1,1),activation='relu')(x6)
    x6 = UpSampling2D(size=(2,2))(x6)
    
    out = concatenate([x5,x6],axis=-1)
    finaloutput = UpSampling2D(size=(2,2))(out)
    
    finaloutput =  Conv2D(c, kernel_size=(3,3), padding='same', strides=(1,1),activation='relu')(finaloutput)
    
    print('finaloutput')
    print(finaloutput.shape)
    return finaloutput


#-----------------------------------------------------------RP--------------------------------------------------------

from keras.engine.topology import Layer
import keras.backend as K


class RoiPooling(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list
        self.num_rois = num_rois

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.nb_channels * self.num_outputs_per_channel

    def get_config(self):
        config = {'pool_list': self.pool_list, 'num_rois': self.num_rois}
        base_config = super(RoiPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = [w / i for i in self.pool_list]
            col_length = [h / i for i in self.pool_list]

            if self.dim_ordering == 'th':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * col_length[pool_num]
                            x2 = x1 + col_length[pool_num]
                            y1 = y + jy * row_length[pool_num]
                            y2 = y1 + row_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')

                            new_shape = [input_shape[0], input_shape[1],
                                         y2 - y1, x2 - x1]
                            x_crop = img[:, :, y1:y2, x1:x2]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(2, 3))
                            outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x + ix * col_length[pool_num]
                            x2 = x1 + col_length[pool_num]
                            y1 = y + jy * row_length[pool_num]
                            y2 = y1 + row_length[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')
                            x2 = K.cast(K.round(x2), 'int32')
                            y1 = K.cast(K.round(y1), 'int32')
                            y2 = K.cast(K.round(y2), 'int32')

                            new_shape = [input_shape[0], y2 - y1,
                                         x2 - x1, input_shape[3]]
                            x_crop = img[:, y1:y2, x1:x2, :]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(1, 2))
                            outputs.append(pooled_val)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.nb_channels * self.num_outputs_per_channel))

        return final_output
    
    
    

#-----------------------------------------------------------RPC--------------------------------------------------------    
    

Learn more or give us feedback
from keras.engine.topology import Layer
import keras.backend as K


class RoiPoolingConv(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    """

    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            if self.dim_ordering == 'th':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')

                        dx = K.maximum(1, x2 - x1)
                        x2 = x1 + dx

                        dy = K.maximum(1, y2 - y1)
                        y2 = y1 + dy

                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]
                        x_crop = img[:, y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

#-----------------------------------------------------------SPP--------------------------------------------------------    
    
    
from keras.engine.topology import Layer
import keras.backend as K


class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'th':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            num_rows = input_shape[1]
            num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        if self.dim_ordering == 'th':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]
                        x_crop = x[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

        elif self.dim_ordering == 'tf':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]

                        x_crop = x[:, y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        if self.dim_ordering == 'th':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'tf':
            #outputs = K.concatenate(outputs,axis = 1)
            outputs = K.concatenate(outputs)
            #outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            #outputs = K.permute_dimensions(outputs,(3,1,0,2))
            #outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs
    
 
#-----------------------------------------------------------xblock--------------------------------------------------------   
    

def conv2d_bn_relu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
def conv2d_bn_relu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def fsm(x):
    channel_num = x.shape[-1]

    res = x

    x = conv2d_bn_relu(x, filters=int(channel_num // 8), kernel_size=(3, 3))

    # x = non_local_block(x, compression=2, mode='dot')

    ip = x
    ip_shape = K.int_shape(ip)
    batchsize, dim1, dim2, channels = ip_shape
    intermediate_dim = channels // 2
    rank = 4
    if intermediate_dim < 1:
        intermediate_dim = 1

    # theta path
    theta = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    theta = Reshape((-1, intermediate_dim))(theta)

    # phi path
    phi = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    phi = Reshape((-1, intermediate_dim))(phi)

    # dot
    f = dot([theta, phi], axes=2)
    size = K.int_shape(f)
    # scale the values to make it size invariant
    f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    # g path
    g = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    g = Reshape((-1, intermediate_dim))(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])
    y = Reshape((dim1, dim2, intermediate_dim))(y)
    y = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-5))(y)
    y = add([ip, y])

    x = y
    x = conv2d_bn_relu(x, filters=int(channel_num), kernel_size=(3, 3))
    print(x)

    x = add([x, res])
    return x


def create_fsm_model(input_shape=(224, 192, 1)):
    input_img = Input(shape=input_shape)
    x = fsm(input_img)
    model = Model(input_img, x)
    model.summary()
    return model


if __name__ == '__main__':
    create_fsm_model(input_shape=(14, 12, 1024))

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

#----------------------------------------------------dual attenation---------------------------------------------
from Vnet.layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add, upsample3d,
                        weight_xavier_init, bias_variable, save_images)
import tensorflow as tf
import numpy as np
import os



def positionAttentionblock(x, inputfilters, outfilters, kernal_size=1, scope=None):
    """
    Position attention module
    :param x:
    :param inputfilters:inputfilter number
    :param outfilters:outputfilter number
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        m_batchsize, Z, H, W, C = x.get_shape().as_list()

        kernalquery = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wquery = weight_xavier_init(shape=kernalquery,
                                    n_inputs=kernalquery[0] * kernalquery[1] * kernalquery[2] * kernalquery[3],
                                    n_outputs=kernalquery[-1], activefunction='relu',
                                    variable_name=scope + 'conv_Wquery')
        Bquery = bias_variable([kernalquery[-1]], variable_name=scope + 'conv_Bquery')
        query_conv = conv3d(x, Wquery) + Bquery
        query_conv_new = tf.reshape(query_conv, [-1, Z * H * W])

        kernalkey = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wkey = weight_xavier_init(shape=kernalkey, n_inputs=kernalkey[0] * kernalkey[1] * kernalkey[2] * kernalkey[3],
                                  n_outputs=kernalkey[-1], activefunction='relu', variable_name=scope + 'conv_Wkey')
        Bkey = bias_variable([kernalkey[-1]], variable_name=scope + 'conv_Bkey')
        key_conv = conv3d(x, Wkey) + Bkey
        key_conv_new = tf.reshape(key_conv, [-1, Z * H * W])

        # OOM,such as 512x512x32 then matric is 8388608x8388608
        # key_conv_new = tf.transpose(key_conv_new, [0, 2, 1])
        # (2,2,2,3)*(2,2,3,4)=(2,2,2,4),(2,2,3)*(2,3,4)=(2,2,4)
        # energy = tf.matmul(query_conv_new, key_conv_new)  # (m_batchsize,Z*H*W,Z*H*W)

        energy = tf.multiply(query_conv_new, key_conv_new)
        attention = tf.nn.sigmoid(energy)

        kernalproj = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wproj = weight_xavier_init(shape=kernalproj,
                                   n_inputs=kernalproj[0] * kernalproj[1] * kernalproj[2] * kernalproj[3],
                                   n_outputs=kernalproj[-1], activefunction='relu', variable_name=scope + 'conv_Wproj')
        Bproj = bias_variable([kernalproj[-1]], variable_name=scope + 'conv_Bproj')
        proj_value = conv3d(x, Wproj) + Bproj
        proj_value_new = tf.reshape(proj_value, [-1, Z * H * W])

        out = tf.multiply(attention, proj_value_new)
        out_new = tf.reshape(out, [-1, Z, H, W, C])

        out_new = resnet_Add(out_new, x)
        return out_new


def channelAttentionblock(x, scope=None):
    """
    Channel attention module
    :param x:input
    :param scope: scope name
    :return:channelattention result
    """
    with tf.name_scope(scope):
        m_batchsize, Z, H, W, C = x.get_shape().as_list()

        proj_query = tf.reshape(x, [-1, C])
        proj_key = tf.reshape(x, [-1, C])
        proj_query = tf.transpose(proj_query, [1, 0])

        energy = tf.matmul(proj_query, proj_key)  # (C,C)
        attention = tf.nn.sigmoid(energy)

        proj_value = tf.reshape(x, [-1, C])
        proj_value = tf.transpose(proj_value, [1, 0])
        out = tf.matmul(attention, proj_value)  # (C,-1)

        out = tf.reshape(out, [-1, Z, H, W, C])
        out = resnet_Add(out, x)
        return out


def conv_bn_relu_drop(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    """
    :param x:
    :param kernal:
    :param phase:
    :param drop:
    :param image_z:
    :param height:
    :param width:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv3d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def down_sampling(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W, 2) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def deconv_relu(x, kernal, samefeture=False, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-2]], variable_name=scope + 'B')
        conv = deconv3d(x, W, samefeture, True) + B
        conv = tf.nn.relu(conv)
        return conv
    
    

#-----------------------------------------------------------fsm--------------------------------------------------------    
    
def conv2d_bn_relu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def fsm(x):
    channel_num = x.shape[-1]

    res = x

    x = conv2d_bn_relu(x, filters=int(channel_num // 8), kernel_size=(3, 3))

    # x = non_local_block(x, compression=2, mode='dot')

    ip = x
    ip_shape = K.int_shape(ip)
    batchsize, dim1, dim2, channels = ip_shape
    intermediate_dim = channels // 2
    rank = 4
    if intermediate_dim < 1:
        intermediate_dim = 1

    # theta path
    theta = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    theta = Reshape((-1, intermediate_dim))(theta)

    # phi path
    phi = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    phi = Reshape((-1, intermediate_dim))(phi)

    # dot
    f = dot([theta, phi], axes=2)
    size = K.int_shape(f)
    # scale the values to make it size invariant
    f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    # g path
    g = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-5))(ip)
    g = Reshape((-1, intermediate_dim))(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])
    y = Reshape((dim1, dim2, intermediate_dim))(y)
    y = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-5))(y)
    y = add([ip, y])

    x = y
    x = conv2d_bn_relu(x, filters=int(channel_num), kernel_size=(3, 3))
    print(x)

    x = add([x, res])
    return x


def create_fsm_model(input_shape=(224, 192, 1)):
    input_img = Input(shape=input_shape)
    x = fsm(input_img)
    model = Model(input_img, x)
    model.summary()
    return model

def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)
    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    
    y_pool = Conv2D(filters=128, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d')(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)
    
    #y_pool = UpSample.bilinear_upsample(tensor=y_pool, size=[dims[1], dims[2]])
    
    y_pool = UpSampling2D((dims[1], dims[2]))(y_pool)
    y_1 = Conv2D(filters=128, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1')(tensor)
    y_1 = BatchNormalization(name=f'bn_2')(y_1)
    y_1 = Activation('relu', name=f'relu_2')(y_1)
    print(y_1.shape)
    y_6 = Conv2D(filters=128, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6')(tensor)
    y_6 = BatchNormalization(name=f'bn_3')(y_6)
    y_6 = Activation('relu', name=f'relu_3')(y_6)
    print(y_6.shape)
    y_12 = Conv2D(filters=128, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12')(tensor)
    y_12 = BatchNormalization(name=f'bn_4')(y_12)
    y_12 = Activation('relu', name=f'relu_4')(y_12)
    print(y_12.shape)
    y_18 = Conv2D(filters=128, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18')(tensor)
    y_18 = BatchNormalization(name=f'bn_5')(y_18)
    y_18 = Activation('relu', name=f'relu_5')(y_18)
    print(y_18.shape)
    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')
    print(y.shape)
    y = Conv2D(filters=128, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final')(y)
    y = BatchNormalization(name=f'bn_final')(y)
    y = Activation('relu', name=f'relu_final')(y)
    return y


if __name__ == '__main__':
    create_fsm_model(input_shape=(14, 12, 1024))
