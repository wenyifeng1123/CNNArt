# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:57:10 2016

@author: Thomas Kuestner
"""

"""Import"""

import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
import scipy.io as sio              # I/O
import os.path                      # operating system

import argparse

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
import keras
import fTrain_fPredict_l3 as trainer_3D


print "KerasVersion is %s"  % (keras.__version__)
print 'running cnn_main module'



"""functions"""    
def fLoadData(conten, lPred, l3D):
    useX_test = True
    # prepared in matlab
    print 'Loading data'
    Varnames = ['X_train', 'X_test', 'y_train', 'y_test']
    for sVarname in Varnames:
        if sVarname in conten:
            exec(sVarname + '=conten[sVarname]')
        else:
            exec(sVarname + '= None')
    if not lPred:
        print 'and shuffle the training set'
        pIdx = np.random.permutation(np.arange(len(X_train)))
        X_train = X_train[pIdx]
        y_train = y_train[pIdx]
        y_test= np.asarray([y_test[:,0], np.abs(np.asarray(y_test[:,0],dtype=np.float32)-1)]).T
    y_train= np.asarray([y_train[:,0], np.abs(np.asarray(y_train[:,0],dtype=np.float32)-1)]).T
    if lPred:
        print 'for prediction'
        if X_test ==None:
            return X_train, y_train
        else:
            return X_test, y_test
    else:
        return X_train, y_train, X_test, y_test

def fConcatInputs(DataList, arguments):
    dData=dict()
    if 'patchSize' in arguments: dData['patchSize']=DataList[0]['patchSize']
    if 'Patient' in arguments: dData['Patient']=DataList[0]['Patient']
    for arg in ['X_train','Y_train', 'y_train', 'X_test', 'Y_test', 'y_test']:
        if arg not in arguments:
            continue
        dData[arg]=DataList[0][arg]
        if len(DataList)>1:
            for iPartial in range(1,len(DataList)):
                dData[arg]=np.concatenate((dData[arg], DataList[iPartial][arg]), axis=0)
    return dData

def fAugmentDataWithRotation(dData):
    print '-------------augment samples with all 4 rotations, labels getting copied--------'
    if not 'X_test' in dData:
        print 'argument X_test not in dData!'
    outDict=dData.copy()
    for arg in zip(['X_train','X_test'],['y_train','y_test']):
        for iRot in range(1, 4):
            rotated=np.rot90(dData[arg[0]], k=iRot, axes=(2,3))
            outDict[arg[0]]=np.concatenate((outDict[arg[0]], rotated),axis=0)
            outDict[arg[1]]=np.concatenate((outDict[arg[1]], dData[arg[1]]), axis=0)

    return outDict

def fRemove_entries(entries, the_dict):
    for key in entries:
        if key in the_dict:
            del the_dict[key]

def fLoadMat(sInPath):
    """Data"""
    if os.path.isfile(sInPath):
        try:
            conten = sio.loadmat(sInPath)
        except:
            f = h5py.File(sInPath,'r')
            conten = {}
            conten['X_train'] = np.array(f['X_train'])
            conten['X_test'] = np.array(f['X_test'])
            if len(conten['X_train'].shape)==4:
                l3D=False
                print 'working with 2D-Data'
                conten['X_train']=np.transpose(conten['X_train'],(3, 2, 1, 0))
                conten['X_test']= np.transpose(conten['X_test'],(3, 2, 1, 0))
            else:
                l3D=True
                print 'working with 3D-Data'
                conten['X_train'] = np.transpose(conten['X_train'], (4, 3, 2, 1, 0))
                conten['X_test'] = np.transpose(conten['X_test'], (4, 3, 2, 1, 0))

            conten['y_train'] = np.transpose(np.array(f['y_train']))
            conten['y_test'] = np.transpose(np.array(f['y_test']))
            conten['patchSize'] = np.transpose(np.array(f['patchSize']))
            try:
                conten['Patient'] = np.transpose(np.array(f['iPat']))
                conten['Patient'] = int(conten['Patient'][0, 0])  # unpack the value
            except:
                conten['Patient'] = 0

    else:
        sys.exit('Input file is not existing')
    X_train, y_train, X_test, y_test = fLoadData(conten, False, l3D) # output order needed for hyperas
    
    fRemove_entries(('X_train', 'X_test', 'y_train', 'y_test'), conten )
    dData = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
    dOut = dData.copy()
    dOut.update(conten)
    return dOut # output dictionary (similar to conten, but with reshaped X_train, ...)


def fLoadMatPred(sInPath):
    if os.path.isfile(sInPath):

        useX_test = True

        f = h5py.File(sInPath, 'r')
        conten = {}
        if useX_test:
            print 'use X_test/y_test if they exist'
            try:
                conten['X_train'] = np.array(f['X_test'])
                if len(conten['X_train'].shape())==4:
                    l3D=False
                    print 'working with 2D-Data'
                    conten['X_train'] = np.transpose(conten['X_test'], (3, 2, 1, 0))
                else:
                    l3D = True
                    print 'working with 3D-Data'
                    conten['X_train']=np.transpose(conten['X_test'],(4, 3, 2, 1, 0))
                conten['y_train'] = np.transpose(np.array(f['y_test']))
            except:
                print '... they did not--->use X/y-train'
                conten['X_train'] = np.array(f['X_train'])
                if len(conten['X_train'].shape)==4:
                    l3D=False
                    print 'working with 2D-Data'
                    conten['X_train'] = np.transpose(conten['X_train'], (3, 2, 1, 0))
                else:
                    l3D = True
                    print 'working with 3D-Data'
                    conten['X_train']=np.transpose(conten['X_train'],(4, 3, 2, 1, 0))
                conten['y_train'] = np.transpose(np.array(f['y_train']))

        else:
            print 'not possible right now to use X_train/y_train if test data exist...'
            conten['X_train'] = np.transpose(np.array(f['X_train']), (4, 3, 2, 1, 0))
            conten['y_train'] = np.transpose(np.array(f['y_train']))
            #TODO implement
        conten['patchSize'] = np.transpose(np.array(f['patchSize']))
        try:
            conten['Patient'] = np.transpose(np.array(f['iPat']))
            conten['Patient'] = int(conten['Patient'][0, 0])  # unpack the value
        except:
            conten['Patient'] = 0

        X_train, y_train = fLoadData(conten, True,l3D)
    else:
        sys.exit('Input file is not existing')


    fRemove_entries(('X_train', 'y_train'), conten)
    dData = {'X_train': X_train, 'y_train': y_train}
    dOut = dData.copy()
    dOut.update(conten)
    return dOut  # output dictionary (similar to conten, but with reshaped X_train, ...)






# input parsing
parser = argparse.ArgumentParser(description='''CNN feature learning''', epilog='''(c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de''')
parser.add_argument("inPath",metavar='N',nargs='+',  type=str,
                    help="following input PatchData Paths, if trained with more Data, the Input *.mat file should contain: -Patchsize -X_train/Y_train/X_test/Y_test for training and at least one of the pairs X_train/Y_train or X_test/Ytest to predict on")#vorsicht!!! neue syntax
#parser.add_argument('-i','--inPath', nargs = 1, type = str, help='input path to *.mat of stored patches', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/in.mat')
parser.add_argument('-o','--outPath', nargs = 1, type = str, help='output path to the file used for storage (subfiles _model, _weights, ... are automatically generated)', default= '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Datatmp/out' )
parser.add_argument('-m','--model', nargs = 1, type = str, choices =['motion_head', 'motion_abd', 'motion_all', 'shim', 'noise', 'motion_pelvis_t1',
                    ], help='select the type of artefact you want to train on', default='motion_head')
parser.add_argument('-t','--train', dest='train', action='store_true', help='if set -> training | if not set -> prediction' )
parser.add_argument('-p','--paraOptim', dest='paraOptim', type = str, choices = ['grid','hyperas','none'], help='parameter optimization via grid search, hyper optimization or no optimization, grid is recommended', default = 'none')
parser.add_argument('-C','--CNN', dest='sModelPath', type=str,
                    help='choose a trained model for predicting, select the _json.txt file!',
                    default='/home/s1222/marvin_stuff/IQA/Codes_FeatureLearning/bestModels/abdomen_3030_lr_0.0001_bs_64.mat')
parser.add_argument('-a','--architecture', dest='architecture', type=str, help= 'your desired CNN-architecture (only 3D)',
                    choices=['Layers3', 'VNet',
                    'VNet_2', 'VNet_3', 'MNet'], default='Layers3')
args = parser.parse_args()
        
if os.path.isfile(args.outPath[0]):
    print('Warning! Output file is already existing and will be overwritten')


print 'data input path/paths is/are: ' +str(args.inPath)
print 'data output path is: ' +args.outPath[0]
print '===========model: {}==========='.format(args.model[0])
"""CNN Models"""
# dynamic loading of corresponding model

cnnModel = __import__(args.model[0], globals(), locals(), ['createModel', 'fTrain', 'fPredict'], -1) # dynamic module loading with specified functions and with relative implict importing (level=-1) -> only in Python2
#architecture loading ... other file for better owerview
# train (w/ or w/o optimization) and predicting
if args.train: # training
    if args.paraOptim == 'hyperas': # hyperas parameter optimization
        print 'Hyperas Optimization currently not supported, comments in the code are throwing errors!'
        pass
        # best_run, best_model = optim.minimize(model=cnnModel.fHyperasTrain,
        #                                       data=fLoadDataForOptim,
        #                                       algo=tpe.suggest,
        #                                       max_evals=5,
        #                                       trials=Trials())
        # X_train, y_train, X_test, y_test, patchSize = fLoadDataForOptim(args.inPath[0])
        # score_test, acc_test = best_model.evaluate(X_test, y_test)
        # prob_test = best_model.predict(X_test, best_run['batch_size'], 0)
        #
        # _, sPath = os.path.splitdrive(sOutPath)
        # sPath,sFilename = os.path.split(sPath)
        # sFilename, sExt = os.path.splitext(sFilename)
        # model_name = sPath + '/' + sFilename + str(patchSize[0,0]) + str(patchSize[0,1]) +'_best'
        # weight_name = model_name + '_weights.h5'
        # model_json = model_name + '_json'
        # model_all = model_name + '_model.h5'
        # json_string = best_model.to_json()
        # open(model_json, 'w').write(json_string)
        # #wei = best_model.get_weights()
        # best_model.save_weights(weight_name)
        # #best_model.save(model_all)
        #
        # result = best_run['result']
        # #acc = result.history['acc']
        # loss = result.history['loss']
        # val_acc = result.history['val_acc']
        # val_loss = result.history['val_loss']
        # sio.savemat(model_name,{'model_settings':model_json,
        #                             'model':model_all,
        #                             'weights':weight_name,
        #                             'acc':-best_run['loss'],
        #                             'loss': loss,
        #                             'val_acc':val_acc,
        #                             'val_loss':val_loss,
        #                             'score_test':score_test,
        #                             'acc_test':acc_test,
        #                             'prob_test':prob_test})

    elif args.paraOptim == 'grid': # grid search
        print "grid search is easier to use, when you just add for loops around the training code"
        pass
		# cnnModel.fGridTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath,
         #                    dData['patchSize'], CV_Patient=dData['Patient'])


    else: # no optimization
        dData = []
        for sPath in args.inPath:
            dData.append(fLoadMat(sPath))
            if not (dData[-1]['X_train'].shape[2] == dData[-1]['patchSize'][0, 0] and
                            dData[-1]['X_train'].shape[3] == dData[-1]['patchSize'][0, 1]):
                print 'shape of X_train does not match patchSize!!!!!!!!'

        arguments = ['X_train', 'y_train', 'X_test', 'y_test', 'patchSize', 'Patient']
        dData = fConcatInputs(dData, arguments)
        #dData = fAugmentDataWithRotation(dData)
        sOutPath = args.outPath[0]
        if len(dData['patchSize'][0])==2:
            l3D=False
            cnnModel.fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath,
                            dData['patchSize'], iEpochs=300, CV_Patient=dData['Patient'])
        else:
            l3D=True
            trainer_3D.fGridTrain3D(sOutPath, dData['patchSize'], X_train = dData['X_train'], Y_train = dData['y_train'],
                                    X_test = dData['X_test'], Y_test = dData['y_test'],architecture=args.architecture,
                                    CV_Patient=dData['Patient'], model=args.model[0])





else: # predicting
    dData = fLoadMatPred(args.inPath[0])
    if len(dData['patchSize'][0])==2:#2D
        cnnModel.fPredict(dData['X_train'], dData['y_train'], args.sModelPath, args.outPath[0], batchSize=64)
    else:#3D
        trainer_3D.fPredict3D(dData['X_train'], dData['y_train'], args.sModelPath, args.outPath[0], batchSize=64)



