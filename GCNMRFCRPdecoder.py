from keras.layers.convolutional import Conv3D
from keras.layers import Dropout, Input,UpSampling3D,concatenate,multiply,UpSampling3D,AveragePooling3D
from keras.layers import Flatten, Add
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.

#https://www.cnblogs.com/ariel-dreamland/p/10569968.html
#desnet18/34

def Conv3d_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
    x = Conv3D(nb_filter, kernel_size, padding=padding, strides=strides,
               activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
    x = Conv3d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv3d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv3d_BN(inpt, nb_filter=nb_filter, strides=strides,
                             kernel_size=kernel_size)
        x = Dropout(0.2)(x)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

    
    
#psp





#FPN




#Global Convolutional Network
def GCN(x,k):
    x1 = x
    nb_filter=x.shape[-1]
    X1 =  Conv3D(nb_filter, kernel_size=(3,1,1), padding='same', strides=(k,k,k))(x)
    X1 =  Conv3D(nb_filter, kernel_size=(1,3,1), padding='same', strides=(k,k,k))(x)
    X1 =  Conv3D(nb_filter, kernel_size=(1,1,3), padding='same', strides=(k,k,k))(x)
    X2 =  Conv3D(nb_filter, kernel_size=(3,1,1), padding='same', strides=(k,k,k))(x)
    X2 =  Conv3D(nb_filter, kernel_size=(1,3,1), padding='same', strides=(k,k,k))(x)
    X2 =  Conv3D(nb_filter, kernel_size=(1,1,3), padding='same', strides=(k,k,k))(x)
    X3 =  Conv3D(nb_filter, kernel_size=(3,1,1), padding='same', strides=(k,k,k))(x)
    X3 =  Conv3D(nb_filter, kernel_size=(1,3,1), padding='same', strides=(k,k,k))(x)
    X3 =  Conv3D(nb_filter, kernel_size=(1,1,3), padding='same', strides=(k,k,k))(x)
    out = Add([x1,x2,x3],axis=-1)
    return out

#Boundary Refinement
def BR(x):
    x1 = x
    nb_filter=x.shape[-1]
    x = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(x)
    x = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2))(x)
    out = Add([x1,x],axis=-1)
    return out

#RCU:Residual Conv Unit
def RCU(x):
    x1 = x
    x = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(x)
    x = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(x)
    out = Add([x1,x],axis=-1)
    return out

#Multi-resolution Fusion
def MRF(a,b):
    x1 = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(a)
    x1 = UpSampling3D((k,k,k))(x1)
    x2 = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(b)
    x2 = UpSampling3D((k,k,k))(x2)
    out = Add([x1,x2],axis=-1)
    return out

#Chained Residual Pooling
def CRP(x):
    x = Activation('ReLU')(x)
    x1 = MaxPooling(pool_size=(5,5,5))(x)
    x1 = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(x1)
    x = Add([x,x1],axis=-1)
    x2 = MaxPooling3D(pool_size=(5,5,5))(x1)
    x2 = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(x2)
    x = Add([x,x2],axis=-1)
    x3 = MaxPooling3D(pool_size=(5,5,5))(x2)
    x3 = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(x3)
    x = Add([x,x3],axis=-1)
    return x


def Dencode(x,nb_filter,c):
    print(x.shape)  
#output shape 8倍
    x1 = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(x)
    x2 = Conv3D(c, kernel_size=(3,3,3), padding='same', strides=(1,1,1),activation='relu')(x1)
   
    x1 = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(x1)
    x3 = Conv3D(c, kernel_size=(3,3,3), padding='same', strides=(1,1,1),activation='relu')(x1)
    
    x1 = Conv3D(nb_filter, kernel_size=(3,3,3), padding='same', strides=(2,2,2),activation='relu')(x1)
    x4 = Conv3D(c, kernel_size=(3,3,3), padding='same', strides=(1,1,1),activation='relu')(x1)
    x4 = UpSampling3D(size=(2,2,2))(x4)
    
    x3 = concatenate([x3, x4],axis=-1)
    x3 = UpSampling3D(size=(2,2,2))(x3)
    x2 = concatenate([x2,x3],axis=-1)
    x2 = UpSampling3D(size=(2,2,2))(x2)
    
    x5 = Conv3D(c, kernel_size=(3,3,3), padding='same', strides=(1,1,1),activation='relu')(x)
#     print(x.shape)
    #x5 = multiply([x2,x5])
    x5 = concatenate([x2,x5],axis=-1)
    x6 = AveragePooling3D(pool_size=(3,3,3),strides=(2,2,2),padding='same')(x)
#     print(x6.shape)
    x6 = Conv3D(c, kernel_size=(3,3,3), padding='same', strides=(1,1,1),activation='relu')(x6)
    x6 = UpSampling3D(size=(2,2,2))(x6)
    
    out = concatenate([x5,x6],axis=-1)
    finaloutput = UpSampling3D(size=(2,2,2))(out)
    
    finaloutput =  Conv3D(c, kernel_size=(3,3,3), padding='same', strides=(1,1,1),activation='relu')(finaloutput)
    
    print('finaloutput')
    print(finaloutput.shape)
    return finaloutput
    
