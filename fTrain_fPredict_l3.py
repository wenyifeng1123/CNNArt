# -*- coding: utf-8 -*-

import os.path
import scipy.io as sio
import numpy as np                  # for algebraic operations, matrices
import keras
import theano
import functools
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten #, Layer  Dropout, Flatten
from keras.layers import merge
from keras.models import model_from_json
#from hyperas.distributions import choice, uniform, conditional
import patch_generator as gen
from keras.layers.convolutional import Conv3D
#from keras.layers.convolutional import MaxPooling2D as pool2
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
#from keras.layers.convolutional import ZeroPadding2D as zero2d
from keras.regularizers import l2#, activity_l2
#from theano import function

from keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.model_selection._split import   BaseShuffleSplit

#from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe
from hyperopt import fmin, hp
import pickle

from sklearn.utils.validation import _num_samples




def fTrain3D(sOutPath, model, sModelName, patchSize=None, sInPaths=None, sInPaths_valid=None, X_train=None, Y_train=None, X_test=None, Y_test=None,  batchSize=64, iEpochs=299, CV_Patient=0):
    '''train a model with training data X_train with labels Y_train. Validation Data should get the keywords Y_test and X_test'''

    print 'Training CNN'
    print 'with '  + 'batchSize = ' + str(batchSize)

    # save names
    _, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    model_name = sPath + '/' + sModelName + '_bs:{}'.format(batchSize)
    if CV_Patient != 0: model_name = model_name +'_'+ 'CV' + str(CV_Patient)# determine if crossValPatient is used...
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if (os.path.isfile(model_mat)):  # no training if output file exists
        print '----------already trained->go to next----------'
        return


    callbacks = [EarlyStopping(monitor='val_acc', patience=10, verbose=1)]
    callbacks.append(ModelCheckpoint('/home/s1222/no_backup/s1222/checkpoints/checker.hdf5', monitor='val_acc', verbose=0,
        period=5, save_best_only=True))# overrides the last checkpoint, its just for security
    callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    result =model.fit(X_train,
                         Y_train,
                         validation_data=[X_test, Y_test],
                         epochs=iEpochs,
                         batch_size=batchSize,
                         callbacks=callbacks,
                         verbose=1)

    print '\nscore and acc on test set:'
    score_test, acc_test = model.evaluate(X_test, Y_test, batch_size=batchSize, verbose=1)
    print '\npredict class probabillities:'
    prob_test = model.predict(X_test, batchSize, verbose=1)

    # save model
    json_string = model.to_json()
    open(model_json +'.txt', 'w').write(json_string)

    model.save_weights(weight_name, overwrite=True)


    # matlab
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']


    print '\nSaving results: ' + model_name
    sio.savemat(model_name, {'model_settings': model_json,
                             'model': model_all,
                             'weights': weight_name,
                             'acc_history': acc,
                             'loss_history': loss,
                             'val_acc_history': val_acc,
                             'val_loss_history': val_loss,
                             'loss_test': score_test,
                             'acc_test': acc_test,
                             'prob_test': prob_test})


def fPredict3D(X,y,  sModelPath, sOutPath, batchSize=64):
    """Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y
    Input:
        X: Samples to predict on. The shape of X should fit to the input shape of the model
        y: Labels for the Samples. Number of Samples should be equal to the number of samples in X
        sModelPath: (String) full path to a trained keras model. It should be *_json.txt file. there has to be a corresponding *_weights.h5 file in the same directory!
        sOutPath: (String) full path for the Output. It is a *.mat file with the computed loss and accuracy stored. 
                    The Output file has the Path 'sOutPath'+ the filename of sModelPath without the '_json.txt' added the suffix '_pred.mat' 
        batchSize: Batchsize, number of samples that are processed at once"""
    sModelPath=sModelPath.replace("_json.txt", "")
    weight_name = sModelPath + '_weights.h5'
    model_json = sModelPath + '_json.txt'
    model_all = sModelPath + '_model.h5'

    # load weights and model (new way)
    model_json= open(model_json, 'r')
    model_string=model_json.read()
    model_json.close()
    model = model_from_json(model_string)

    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.load_weights(weight_name)


    score_test, acc_test = model.evaluate(X, y, batch_size=batchSize)
    print 'loss'+str(score_test)+ '   acc:'+ str(acc_test)
    prob_pre = model.predict(X, batch_size=batchSize, verbose=1)
    print prob_pre[0:14,:]
    _,sModelFileSave  = os.path.split(sModelPath)

    modelSave = sOutPath +sModelFileSave+ '_pred.mat'
    print 'saving Model:{}'.format(modelSave)
    sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test})

############for_GRID############for_GRID############for_GRID############for_GRID############for_GRID############for_GRID
def fGridTrain3D(sOutPath, patchSize,sInPaths=None,sInPaths_valid=None,X_train=None, Y_train=None, X_test=None, Y_test=None, architecture='Layers3', CV_Patient=0, model='motion_head'):#rigid for loops for simplicity

    sImportString = model + '_3D_architectures_l3'
    cnnModel = __import__(sImportString, globals(), locals(),
                          [],
                          -1)
    learning_rate = 0.001
    cnn, sModelName= cnnModel.fCreateModel(patchSize, learningRate=learning_rate, optimizer='Adam',
                                architecture=architecture
                                )
    print "Modelname:" + sModelName
    fTrain3D(sOutPath, cnn, sModelName, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,CV_Patient=CV_Patient,
         batchSize=64, iEpochs=300)



