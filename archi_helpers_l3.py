import keras
import keras.optimizers
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten,   Dropout, Lambda, Reshape
from keras.activations import relu, elu, softmax
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import Constant
from keras.layers.normalization import BatchNormalization
from keras.layers import  concatenate, add
from keras.layers.convolutional import Conv3D,Conv2D, MaxPooling3D, MaxPooling2D, ZeroPadding3D
from keras.regularizers import l1_l2,l2#, activity_l2
from keras.constraints import maxnorm
import keras.losses as losses
from keras import backend as K

def fCreateMNet_Block(input_t, channels, kernel_size=(3,3), type=1, forwarding=True,l1_reg=0.0, l2_reg=1e-6 ):
    tower_t = Conv2D(channels,
                     kernel_size=kernel_size,
                     kernel_initializer='he_normal',
                     weights=None,
                     padding='same',
                     strides=(1, 1),
                     kernel_regularizer=l1_l2(l1_reg, l2_reg),
                     )(input_t)
    tower_t = Activation('relu')(tower_t)
    for counter in range(1, type):
        tower_t = Conv2D(channels,
                         kernel_size=kernel_size,
                         kernel_initializer='he_normal',
                         weights=None,
                         padding='same',
                         strides=(1, 1),
                         kernel_regularizer=l1_l2(l1_reg, l2_reg),
                         )(tower_t)
        tower_t = Activation('relu')(tower_t)
    if (forwarding):
        tower_t = concatenate([tower_t, input_t], axis=1)
    return tower_t

def fCreateMNet_Block_BatchNorm(input_t, channels, kernel_size=(3,3), type=1, forwarding=True,l1_reg=0.0, l2_reg=1e-6 ):
    tower_t = Conv2D(channels,
                     kernel_size=kernel_size,
                     kernel_initializer='he_normal',
                     weights=None,
                     padding='same',
                     strides=(1, 1),
                     kernel_regularizer=l1_l2(l1_reg, l2_reg),
                     )(input_t)
    tower_t = BatchNormalization(axis=1, scale=False)(tower_t)
    tower_t = Activation('relu')(tower_t)
    for counter in range(1, type):
        tower_t = Conv2D(channels,
                         kernel_size=kernel_size,
                         kernel_initializer='he_normal',
                         weights=None,
                         padding='same',
                         strides=(1, 1),
                         kernel_regularizer=l1_l2(l1_reg, l2_reg),
                         )(tower_t)
        tower_t = BatchNormalization(axis=1, scale=False)(tower_t)
        tower_t = Activation('relu')(tower_t)
    if (forwarding):
        tower_t = concatenate([tower_t, input_t], axis=1)
    return tower_t



def fGetOptimizerAndLoss(optimizer,learningRate=0.001, loss='categorical_crossentropy'):
    if optimizer not in ['Adam', 'SGD', 'Adamax', 'Adagrad', 'Adadelta', 'Nadam', 'RMSprop']:
        print 'this optimizer does not exist!!!'
        return None
    loss='categorical_crossentropy'

    if optimizer == 'Adamax':  # leave the rest as default values
        opti = keras.optimizers.Adamax(lr=learningRate)
        loss = 'categorical_crossentropy'
    elif optimizer == 'SGD':
        opti = keras.optimizers.SGD(lr=learningRate, momentum=0.9, decay=5e-5)
        loss = 'categorical_crossentropy'
    elif optimizer == 'Adagrad':
        opti = keras.optimizers.Adagrad(lr=learningRate)
    elif optimizer == 'Adadelta':
        opti = keras.optimizers.Adadelta(lr=learningRate)
    elif optimizer == 'Adam':
        opti = keras.optimizers.Adam(lr=learningRate, decay=5e-5)
        loss = 'categorical_crossentropy'
    elif optimizer == 'Nadam':
        opti = keras.optimizers.Nadam(lr=learningRate)
        loss = 'categorical_crossentropy'
    elif optimizer == 'RMSprop':
        opti = keras.optimizers.RMSprop(lr=learningRate)
    return opti, loss

def fCreateVNet_Block( input_t, channels, type=1, kernel_size=(3,3,3),l1_reg=0.0, l2_reg=1e-6, iPReLU=0, dr_rate=0):
    tower_t= Dropout(dr_rate)(input_t)
    tower_t = Conv3D(channels,
                           kernel_size=kernel_size,
                           kernel_initializer='he_normal',
                           weights=None,
                           padding='same',
                           strides=(1, 1, 1),
                           kernel_regularizer=l1_l2(l1_reg, l2_reg),
                           )(tower_t)

    tower_t = fGetActivation(tower_t, iPReLU=iPReLU)
    for counter in range(1, type):
        tower_t = Dropout(dr_rate)(tower_t)
        tower_t = Conv3D(channels,
                           kernel_size=kernel_size,
                           kernel_initializer='he_normal',
                           weights=None,
                           padding='same',
                           strides=(1, 1, 1),
                           kernel_regularizer=l1_l2(l1_reg, l2_reg),
                           )(tower_t)
        tower_t = fGetActivation(tower_t, iPReLU=iPReLU)
    tower_t = concatenate([tower_t, input_t], axis=1)
    return tower_t

def fGetActivation(input_t,  iPReLU=0):
    init=0.25
    if iPReLU == 1:  # one alpha for each channel
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4])(input_t)
    elif iPReLU == 2:  # just one alpha for each layer
        output_t = PReLU(alpha_initializer=Constant(value=init), shared_axes=[2, 3, 4, 1])(input_t)
    else:
        output_t = Activation('relu')(input_t)
    return output_t


def fCreateVNet_Block_ElemSum(input_t, channels, chan_match=False,type=1, kernel_size=(3,3,3), l1=0, l2=1e-6, iPReLU=0):

    if (chan_match):#channels input will be ignored, if chan_match is set
        channels=input_t._keras_shape[1]

    tower_t = Conv3D(channels,
                     kernel_size=kernel_size,
                     kernel_initializer='he_normal',
                     weights=None,
                     padding='same',
                     strides=(1, 1, 1),
                     kernel_regularizer=l1_l2(l1, l2),
                     )(input_t)
    tower_t = fGetActivation(tower_t, iPReLU=iPReLU)
    for counter in range(1, type):
        tower_t = Conv3D(channels,
                         kernel_size=kernel_size,
                         kernel_initializer='he_normal',
                         weights=None,
                         padding='same',
                         strides=(1, 1, 1),
                         kernel_regularizer=l1_l2(l1,l2),
                         )(tower_t)
        tower_t = fGetActivation(tower_t, iPReLU=iPReLU)
    if (not chan_match):
        if input_t._keras_shape[1]>tower_t._keras_shape[1]:
            print "input channels higher than output channels-> no ElemSum merging possible"
        toPadd=tower_t._keras_shape[1]-input_t._keras_shape[1]
        input_t=Reshape(target_shape=(#permute dimensions...
                                 input_t._keras_shape[4],
                                 input_t._keras_shape[2],input_t._keras_shape[3],
                                 input_t._keras_shape[1]))(input_t)

        input_t= ZeroPadding3D(padding=((0,0),(0,0),(0,toPadd)) )(input_t)
        input_t=Reshape(target_shape=(#permute dimensions...
                                 input_t._keras_shape[4],
                                 input_t._keras_shape[2],input_t._keras_shape[3],
                                 input_t._keras_shape[1]))(input_t)

    tower_t = add([tower_t, input_t])
    return tower_t

def fCreateVNet_DownConv_Block(input_t,channels, stride, l1_reg=0.0, l2_reg=1e-6, iPReLU=0, dr_rate=0):
    output_t=Dropout(dr_rate)(input_t)
    output_t=Conv3D(channels,
                    kernel_size=stride,
                    strides=stride,
                    weights=None,
                    padding='valid',
                    kernel_regularizer=l1_l2(l1_reg, l2_reg),
                    kernel_initializer='he_normal'
                    )(output_t)
    output_t=fGetActivation(output_t,iPReLU=iPReLU)
    return output_t

def fCreateMaxPooling3D(input_t, stride=(2,2,2)):
    output_t=MaxPooling3D(pool_size=stride,
                            strides=stride,
                            padding='valid')(input_t)
    return output_t

def fCreateMaxPooling2D(input_t,stride=(2,2)):
    output_t=MaxPooling2D(pool_size=stride,
                          strides=stride,
                          padding='valid')(input_t)
    return output_t