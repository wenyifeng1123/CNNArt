###datenverlust


import keras
import keras.optimizers
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten,   Dropout, Lambda, Reshape, Permute
from keras.activations import relu, elu, softmax
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import Constant
from keras.layers.normalization import BatchNormalization
from keras.layers import  concatenate
from keras.layers.convolutional import Conv3D,Conv2D, MaxPooling3D, MaxPooling2D
from keras.regularizers import l2#, activity_l2
from keras.constraints import maxnorm
import archi_helpers_l3 as h






def fCreateModel(patchSize,learningRate=1e-3, optimizer='SGD', architecture='Layers3',
                 dr_rate=0.0, input_dr_rate=0.0, max_norm=5,  iPReLU=0, l2_reg=1e-6):

    print 'selected architecture is {}'.format(architecture)
    print 'fCreateModel>,ps:{}x{}x{} lr:{}, opt:{}, drrate:{}, maxNorm:{}'.format(
        int(patchSize[0,0]), int(patchSize[0,1]), int(patchSize[0,2]), learningRate,
        optimizer, dr_rate,max_norm)


    if architecture=='Layers3':
        # change to functional API
        input_t = Input(shape=(1, int(patchSize[0, 0]), int(patchSize[0, 1]), int(patchSize[0, 2])))
        seq_t= Dropout(dr_rate)(input_t)
        seq_t = Conv3D(32,  # numChans
                       kernel_size=(14, 14, 5),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l2(l2_reg),
                       input_shape=(1, int(patchSize[0, 0]), int(patchSize[0, 1]), int(patchSize[0, 2]))
                       )(seq_t)
        seq_t = h.fGetActivation(seq_t, iPReLU=iPReLU)

        seq_t = Dropout(dr_rate)(seq_t)
        seq_t = Conv3D(64,
                       kernel_size=(7, 7, 3),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l2(l2_reg))(seq_t)

        seq_t = h.fGetActivation(seq_t, iPReLU=iPReLU)

        seq_t = Dropout(dr_rate)(seq_t)
        seq_t = Conv3D(128,
                       kernel_size=(3, 3, 2),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l2(l2_reg))(seq_t)

        seq_t = h.fGetActivation(seq_t, iPReLU=iPReLU)

        seq_t = Flatten()(seq_t)

        seq_t = Dropout(dr_rate)(seq_t)
        seq_t = Dense(units=2,
                      kernel_initializer='normal',
                      kernel_regularizer=l2(l2_reg))(seq_t)
        output_t = Activation('softmax')(seq_t)

        opti, loss = h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)  # loss cat_crosent default

        cnn = Model(inputs=[input_t], outputs=[output_t])
        cnn.compile(loss=loss, optimizer=opti, metrics=['accuracy'])
        sArchiSpecs = '_l2{}'.format(l2_reg)


    elif architecture=='Layers3_BN':
        #for patchsize 20x20x20: val_loss: 0.3522 - val_acc: 0.8561


        print "no net fertig"
        a=1
        dense_a=1

        input_t=Input(shape=(1,int(patchSize[0,0]), int(patchSize[0,1]), int(patchSize[0,2])))
        tower_t=Conv3D(32,
                       kernel_size=(6, 6, 6),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l2(1e-6))(input_t)
        # normalization has #params=4xInput_shape(axis)<- one for beta, one for gamma, ?var and mean?(nottrainable!)
        #setting center=False does not work... theano backend does weird things...
        tower_t=BatchNormalization(axis=a, scale=False )(tower_t)
        tower_t=Activation('relu')(tower_t)

        tower_t=Conv3D(64,
                       kernel_size=(5, 5, 5),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l2(1e-6))(tower_t)
        tower_t = BatchNormalization(axis=a,scale=False)(tower_t)
        tower_t=Activation('relu')(tower_t)

        tower_t=Conv3D(128,
                       kernel_size=(4, 4, 4),
                       kernel_initializer='he_normal',
                       weights=None,
                       padding='valid',
                       strides=(1, 1, 1),
                       kernel_regularizer=l2(1e-6))(tower_t)
        tower_t = BatchNormalization(axis=a,scale=False)(tower_t)
        tower_t=Activation('relu')(tower_t)

        flat_t=Flatten()(tower_t)

        after_dense_t=Dense(units=2,
                      kernel_initializer='normal',
                      kernel_regularizer=l2(1e-6))(flat_t)
        #insert batch normalization
        norm_t=BatchNormalization(axis=dense_a, scale=False)(after_dense_t)
        output_t=Activation('softmax')(norm_t)

        opti, loss = h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)  # loss cat_crosent default
        cnn=Model(inputs=[input_t], outputs=[output_t])
        cnn.compile(loss=loss, optimizer=opti, metrics=['accuracy'])
        sArchiSpecs = '_PRElu_alpha:{}'.format(activation_alpha)

    elif architecture=='Layers4_dropout':
        cnn = Sequential()
        cnn.add(Dropout(input_shape=(1, int(patchSize[0, 0]), int(patchSize[0, 1]), int(patchSize[0, 2])),rate=input_dr_rate,
                        ))
        cnn.add(Conv3D(32,  # number of channels should be #wo_dr *(1/p) =>leave them how they are
                              kernel_size=(6, 6, 6),
                              kernel_initializer='he_normal',
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(l2),
                              ))
        cnn.add(Activation('relu'))

        cnn.add(Dropout(rate=dr_rate))
        cnn.add(Conv3D(64,
                              kernel_size=(5, 5, 5),
                              kernel_initializer='he_normal',
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(l2)))
        cnn.add(Activation('relu'))

        cnn.add(Dropout(rate=dr_rate))
        cnn.add(Conv3D(128,
                              kernel_size=(4, 4, 4),
                              kernel_initializer='he_normal',
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(l2)))
        cnn.add(Activation('relu'))

        cnn.add(Dropout(rate=dr_rate))
        cnn.add(Conv3D(128,
                              kernel_size=(3, 3, 3),
                              kernel_initializer='he_normal',
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(l2)))
        cnn.add(Activation('relu'))

        cnn.add(Flatten())

        cnn.add(Dropout(rate=dr_rate))
        cnn.add(Dense(units=2,
                      kernel_initializer='normal',
                      kernel_regularizer='l2'))
        cnn.add(Activation('softmax'))

        opti, loss = h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)  # loss categ_crosent default

        cnn.compile(loss=loss, optimizer=opti, metrics=['accuracy'])

    elif architecture=='Layers4':
        cnn = Sequential()
        cnn.add(Conv3D(32,  # numChans
                              kernel_size=(6, 6, 6),
                              kernel_initializer='he_normal',
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(1e-6),
                              # tricked input shape, patchsize is stored as float...
                              input_shape=(1, int(patchSize[0, 0]), int(patchSize[0, 1]), int(patchSize[0, 2]))
                              ))
        cnn.add(Activation('relu'))

        cnn.add(Conv3D(64,
                              kernel_size=(5, 5, 5),
                              kernel_initializer='he_normal',
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(1e-6)))
        cnn.add(Activation('relu'))

        cnn.add(Conv3D(128,
                              kernel_size=(4, 4, 4),
                              kernel_initializer='he_normal',
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(1e-6)))
        cnn.add(Activation('relu'))

        cnn.add(Conv3D(128,
                              kernel_size=(3, 3, 3),
                              kernel_initializer='he_normal',
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(1e-6)))
        cnn.add(Activation('relu'))

        cnn.add(Flatten())

        cnn.add(Dense(units=2,
                      kernel_initializer='normal',
                      kernel_regularizer='l2'))
        cnn.add(Activation('softmax'))

        opti, loss = h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)  # loss cat_crosent default

        cnn.compile(loss=loss, optimizer=opti, metrics=['accuracy'])

    elif architecture=='Layers4_dropout_maxnorm':
        cnn = Sequential()
        cnn.add(Dropout(rate=input_dr_rate,
                        input_shape=(1, int(patchSize[0, 0]), int(patchSize[0, 1]), int(patchSize[0, 2]))))
        cnn.add(Conv3D(64,  # numChans
                              kernel_size=(6, 6, 6),
                              kernel_initializer='he_normal',
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_constraint=maxnorm(max_value=max_norm),#see dropout reference
                              kernel_regularizer=l2(1e-6),
                              ))
        cnn.add(Activation('relu'))

        cnn.add(Dropout(rate=dr_rate))
        cnn.add(Conv3D(128,
                              kernel_size=(5, 5, 5),
                              kernel_initializer='he_normal',
                              kernel_constraint=maxnorm(max_value=max_norm),  # see dropout reference
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(1e-6)))
        cnn.add(Activation('relu'))

        cnn.add(Dropout(rate=dr_rate))
        cnn.add(Conv3D(256,
                              kernel_size=(4, 4, 4),
                              kernel_initializer='he_normal',
                              kernel_constraint=maxnorm(max_value=max_norm),  # see dropout reference
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(1e-6)))
        cnn.add(Activation('relu'))

        cnn.add(Dropout(rate=dr_rate))
        cnn.add(Conv3D(256,
                              kernel_size=(3, 3, 3),
                              kernel_initializer='he_normal',
                              kernel_constraint=maxnorm(max_value=max_norm),  # see dropout reference
                              weights=None,
                              padding='valid',
                              strides=(1, 1, 1),
                              kernel_regularizer=l2(1e-6)))
        cnn.add(Activation('relu'))

        cnn.add(Flatten())

        cnn.add(Dropout(rate=dr_rate))
        cnn.add(Dense(units=2,
                      kernel_initializer='normal',
                      kernel_constraint=maxnorm(max_value=max_norm),  # see dropout reference
                      kernel_regularizer='l2'))
        cnn.add(Activation('softmax'))

        opti, loss = h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)  # loss cat_crosent default

        cnn.compile(loss=loss, optimizer=opti, metrics=['accuracy'])

    elif architecture=='Layers5':
        print "implement it first"
        #not invented right now
        cnn=None

    elif architecture=='VNet':
        l2_reg=1e-4
        #using SGD lr 0.001
        #motion_head:unkorrigierte Version 3steps with only type(1,1,1)(149K params)--> val_loss: 0.2157 - val_acc: 0.9230
        #motion_head:korrigierte Version type(1,2,2)(266K params) --> val_loss: 0.2336 - val_acc: 0.9149 nach abbruch...
        #double_#channels(type 122) (870,882 params)>
        #functional api...
        input_t=Input(shape=(1, int(patchSize[0, 0]), int(patchSize[0, 1]), int(patchSize[0, 2])))

        after_res1_t=h.fCreateVNet_Block(input_t, 32, type=2, iPReLU=iPReLU,dr_rate=dr_rate,l2_reg=l2_reg)
        after_DownConv1_t=h.fCreateVNet_DownConv_Block(after_res1_t,after_res1_t._keras_shape[1], (2,2,2), iPReLU=iPReLU, dr_rate=dr_rate,l2_reg=l2_reg)

        after_res2_t=h.fCreateVNet_Block(after_DownConv1_t,64, type=2,  iPReLU=iPReLU, dr_rate=dr_rate, l2_reg=l2_reg)
        after_DownConv2_t=h.fCreateVNet_DownConv_Block(after_res2_t, after_res2_t._keras_shape[1],(2,2,1), iPReLU=iPReLU,l2_reg=l2_reg, dr_rate=dr_rate)

        after_res3_t=h.fCreateVNet_Block(after_DownConv2_t, 128, type=2, iPReLU=iPReLU,dr_rate=dr_rate,l2_reg=l2_reg)
        after_DownConv3_t=h.fCreateVNet_DownConv_Block(after_res3_t,after_res3_t._keras_shape[1], (2,2,1), iPReLU=iPReLU,l2_reg=l2_reg, dr_rate=dr_rate)

        after_flat_t=Flatten()(after_DownConv3_t)
        after_dense_t= Dropout(dr_rate)(after_flat_t)
        after_dense_t=Dense(units=2,
                       kernel_initializer='normal',
                       kernel_regularizer=l2(l2_reg))(after_dense_t)
        output_t=Activation('softmax')(after_dense_t)

        cnn=Model(inputs=[input_t], outputs=[output_t])

        opti, loss = h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)  # loss cat_crosent default
        cnn.compile(optimizer=opti, loss=loss, metrics=['accuracy'])
        sArchiSpecs='_t222_l2{}_dr{}'.format(l2_reg, dr_rate)

    elif architecture=='VNet_2':
        l2_reg=1e-4
        #motion_head: with SGD, lr0.001,... --> val_loss: 0.1482 - val_acc: 0.9348
        #187K params
        #use maxpooling instead of DownConv

        input_t = Input(shape=(1, int(patchSize[0, 0]), int(patchSize[0, 1]), int(patchSize[0, 2])))

        after_res1_t = h.fCreateVNet_Block(input_t, 32, type=2, l2_reg=l2_reg)
        after_MaxPool1_t = h.fCreateMaxPooling3D(after_res1_t, stride=(2,2,2))

        after_res2_t = h.fCreateVNet_Block(after_MaxPool1_t, 64, type=2, l2_reg=l2_reg)
        after_MaxPool2_t = h.fCreateMaxPooling3D(after_res2_t, (2, 2, 1))

        after_res3_t = h.fCreateVNet_Block(after_MaxPool2_t, 128, type=2, l2_reg=l2_reg)
        after_MaxPool3_t = h.fCreateMaxPooling3D(after_res3_t,  stride=(2, 2, 1))

        after_flat_t = Flatten()(after_MaxPool3_t)

        after_dense_t = Dense(units=2,
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg))(after_flat_t)
        output_t = Activation('softmax')(after_dense_t)

        cnn = Model(inputs=[input_t], outputs=[output_t])

        opti, loss=h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)#loss cat_crosent default
        cnn.compile(optimizer=opti, loss=loss, metrics=['accuracy'])
        sArchiSpecs = '_t222'

    elif architecture=='VNet_3':#only on linse 9
        l2_reg = 1e-4

        #a VNet with ElementWise Sum like as described, using DownConvolution, matching Layers
        input_t = Input(shape=(1, int(patchSize[0, 0]), int(patchSize[0, 1]), int(patchSize[0, 2])))

        after_res1_t = h.fCreateVNet_Block_ElemSum(input_t, 32, type=1, l2=l2_reg)
        after_DownConv1_t = h.fCreateVNet_DownConv_Block(after_res1_t, after_res1_t._keras_shape[1], (2, 2, 2), l2_reg=l2_reg)

        after_res2_t = h.fCreateVNet_Block_ElemSum(after_DownConv1_t, 64, type=2, l2=l2_reg)
        after_DownConv2_t = h.fCreateVNet_DownConv_Block(after_res2_t, after_res2_t._keras_shape[1], (2, 2, 1), l2_reg=l2_reg)

        after_res3_t = h.fCreateVNet_Block_ElemSum(after_DownConv2_t, 128, type=2, chan_match=False, l2=l2_reg)
        after_DownConv3_t = h.fCreateVNet_DownConv_Block(after_res3_t, after_res3_t._keras_shape[1], (2, 2, 1), l2_reg=l2_reg)

        after_flat_t = Flatten()(after_DownConv3_t)
        after_dense_t = Dense(units=2,
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg))(after_flat_t)
        output_t = Activation('softmax')(after_dense_t)

        cnn = Model(inputs=[input_t], outputs=[output_t])

        opti, loss = h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)  # loss cat_crosent default
        cnn.compile(optimizer=opti, loss=loss, metrics=['accuracy'])
        sArchiSpecs = '_3steps'



    elif architecture=='MNet':#basically a 2D Network
        l2_reg=1e-4

        #(4 stages-each 2 convs)(378,722 params)(for 40x40x10)
        input_t=Input(shape=(1,int(patchSize[0, 0]),int(patchSize[0, 1]), int(patchSize[0, 2])))
        input2D_t=Permute((4,1,2,3))(input_t)
        input2D_t=Reshape(target_shape=(int(patchSize[0, 2]),int(patchSize[0, 0]), int(patchSize[0, 1])))(
            input2D_t)
        #use zDimension as number of channels
        twoD_t=Conv2D(16,
                      kernel_size=(7,7),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      strides=(1,1)
                      )(input2D_t)
        twoD_t = Activation('relu')(twoD_t)

        l_w2_t = h.fCreateMaxPooling2D(twoD_t, stride=(2, 2))
        l_w3_t = h.fCreateMaxPooling2D(l_w2_t, stride=(2, 2))
        l_w4_t = h.fCreateMaxPooling2D(l_w3_t, stride=(2, 2))

        stage1_res1_t=h.fCreateMNet_Block(twoD_t,16,kernel_size=(3,3), forwarding=True, l2_reg=l2_reg)
        stage1_res2_t=h.fCreateMNet_Block(stage1_res1_t,32,kernel_size=(3,3), forwarding=False, l2_reg=l2_reg)

        stage2_inp_t=h.fCreateMaxPooling2D(stage1_res2_t, stride=(2,2))
        stage2_inp_t=concatenate([stage2_inp_t,l_w2_t], axis=1)
        stage2_res1_t=h.fCreateMNet_Block(stage2_inp_t,32,l2_reg=l2_reg)
        stage2_res2_t=h.fCreateMNet_Block(stage2_res1_t,48, forwarding=False)

        stage3_inp_t=h.fCreateMaxPooling2D(stage2_res2_t, stride=(2,2))
        stage3_inp_t=concatenate([stage3_inp_t,l_w3_t], axis=1)
        stage3_res1_t=h.fCreateMNet_Block(stage3_inp_t,48,l2_reg=l2_reg)
        stage3_res2_t = h.fCreateMNet_Block(stage3_res1_t, 64, forwarding=False,l2_reg=l2_reg)

        stage4_inp_t = h.fCreateMaxPooling2D(stage3_res2_t, stride=(2, 2))
        stage4_inp_t = concatenate([stage4_inp_t, l_w4_t], axis=1)
        stage4_res1_t = h.fCreateMNet_Block(stage4_inp_t, 64,l2_reg=l2_reg)
        stage4_res2_t = h.fCreateMNet_Block(stage4_res1_t, 128, forwarding=False,l2_reg=l2_reg)

        after_flat_t = Flatten()(stage4_res2_t)

        after_dense_t = Dense(units=2,
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg))(after_flat_t)
        output_t = Activation('softmax')(after_dense_t)

        cnn = Model(inputs=[input_t], outputs=[output_t])

        opti, loss = h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)
        cnn.compile(optimizer=opti, loss=loss, metrics=['accuracy'])
        sArchiSpecs = '3stages_l2{}'.format(l2_reg)

    elif architecture=='MNet_BN':
        input_t = Input(shape=(1, int(patchSize[0, 0]), int(patchSize[0, 1]), int(patchSize[0, 2])))
        input2D_t = Permute((4, 1, 2, 3))(input_t)
        input2D_t = Reshape(target_shape=(int(patchSize[0, 2]), int(patchSize[0, 0]), int(patchSize[0, 1])))(
            input2D_t)  # use zDimension as number of channels
        twoD_t = Conv2D(16,
                        kernel_size=(7, 7),
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-6),
                        strides=(1, 1)
                        )(input2D_t)
        twoD_t= Activation('relu')(twoD_t)
        l_w2_t = h.fCreateMaxPooling2D(twoD_t, stride=(2, 2))
        l_w3_t = h.fCreateMaxPooling2D(l_w2_t, stride=(2, 2))
        l_w4_t = h.fCreateMaxPooling2D(l_w3_t, stride=(2, 2))

        stage1_res1_t = h.fCreateMNet_Block_BatchNorm(twoD_t, 16, kernel_size=(3, 3), forwarding=True)
        stage1_res2_t = h.fCreateMNet_Block_BatchNorm(stage1_res1_t, 32, kernel_size=(3, 3), forwarding=False)

        stage2_inp_t = h.fCreateMaxPooling2D(stage1_res2_t, stride=(2, 2))
        stage2_inp_t = concatenate([stage2_inp_t, l_w2_t], axis=1)
        stage2_res1_t = h.fCreateMNet_Block_BatchNorm(stage2_inp_t, 32)
        stage2_res2_t = h.fCreateMNet_Block_BatchNorm(stage2_res1_t, 48, forwarding=False)

        stage3_inp_t = h.fCreateMaxPooling2D(stage2_res2_t, stride=(2, 2))
        stage3_inp_t = concatenate([stage3_inp_t, l_w3_t], axis=1)
        stage3_res1_t = h.fCreateMNet_Block_BatchNorm(stage3_inp_t, 48)
        stage3_res2_t = h.fCreateMNet_Block_BatchNorm(stage3_res1_t, 64, forwarding=False)

        stage4_inp_t = h.fCreateMaxPooling2D(stage3_res2_t, stride=(2, 2))
        stage4_inp_t = concatenate([stage4_inp_t, l_w4_t], axis=1)
        stage4_res1_t = h.fCreateMNet_Block_BatchNorm(stage4_inp_t, 64)
        stage4_res2_t = h.fCreateMNet_Block_BatchNorm(stage4_res1_t, 128, forwarding=False)

        after_flat_t = Flatten()(stage4_res2_t)

        after_dense_t = Dense(units=2,
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(1e-6))(after_flat_t)
        norm_t=BatchNormalization(axis=1, scale=False)(after_dense_t)
        output_t = Activation('softmax')(norm_t)

        cnn = Model(inputs=[input_t], outputs=[output_t])

        opti, loss = h.fGetOptimizerAndLoss(optimizer, learningRate=learningRate)
        cnn.compile(optimizer=opti, loss=loss, metrics=['accuracy'])
        sArchiSpecs = '_4Stages'





    else:
        print 'invalid architecure!'

    cnn.summary()
    sModelName=architecture+sArchiSpecs+'_ps:{}{}{}_lr:{:.1e}_o:{}_dr:{}'.format(
        int(patchSize[0,0]), int(patchSize[0,1]), int(patchSize[0,2]),learningRate, optimizer, dr_rate)
    return cnn, sModelName

