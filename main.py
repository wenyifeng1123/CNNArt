# demo file
import os
import glob
import yaml
import numpy as np
import scipy.io as sio
import h5py
from DatabaseInfo import DatabaseInfo
import utils.DataPreprocessing as datapre
import utils.Training_Test_Split as ttsplit
import cnn_main

# parse parameters
# training or prediction
lTrain = True
lSave = False # save intermediate test, training sets
with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# initiate info objects
# default database: MRPhysics with ['newProtocol','dicom_sorted']
dbinfo = DatabaseInfo(cfg['MRdatabase'],cfg['subdirs'])


# load/create input data
patchSize = cfg['patchSize']
if cfg['sSplitting'] == 'normal':
    sFSname = 'normal'
elif cfg['sSplitting'] == 'crossvalidation_data':
    sFSname = 'crossVal_data'
elif cfg['sSplitting'] == 'crossvalidation_patient':
    sFSname = 'crossVal'

sOutsubdir = cfg['subdirs'][2] #sOutsubdir: 'testout'
#sOutPath CNN/Headcross/4040/testout建立新的文件夹testout
sOutPath = cfg['selectedDatabase']['pathout'] + os.sep + ''.join(map(str,patchSize)).replace(" ", "") + os.sep + sOutsubdir # + str(ind_split) + '_' + str(patchSize[0]) + str(patchSize[1]) + '.h5'
#.../4040/crossVal4040.h5
sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str,patchSize)).replace(" ", "") + '.h5'
# check if file is already existing -> skip patching

if glob.glob(sOutPath + os.sep + sFSname + ''.join(map(str,patchSize)).replace(" ", "") + '*_input.mat'): # deprecated
    sDatafile = sOutPath + os.sep + sFSname + ''.join(map(str,patchSize)).replace(" ", "") + '_input.mat'
    try:
        conten = sio.loadmat(sDatafile)
    except:
        f = h5py.File(sDatafile, 'r')
        conten = {}
        conten['X_train'] = np.transpose(np.array(f['X_train']), (3, 2, 0, 1))
        conten['X_test'] = np.transpose(np.array(f['X_test']), (3, 2, 0, 1))
        conten['y_train'] = np.transpose(np.array(f['y_train']))
        conten['y_test'] = np.transpose(np.array(f['y_test']))
        conten['patchSize'] = np.transpose(np.array(f['patchSize']))

    X_train = conten['X_train']
    X_test = conten['X_test']
    y_train = conten['y_train']
    y_test = conten['y_test']

elif glob.glob(sDatafile):
    with h5py.File(sDatafile, 'r') as hf:
        X_train = hf['X_train'][:]
        X_test = hf['X_test'][:]
        y_train = hf['y_train'][:]
        y_test = hf['y_test'][:]
        patchSize = hf['patchSize'][:]

else: # perform patching
    dAllPatches = np.zeros((patchSize[0], patchSize[1], 0))
    dAllLabels = np.zeros(0)
    dAllPats = np.zeros((0, 1))
    #selectedDatabase值为*id001，所以直接选motion_head的dataref:t1...0002和dataart:t1...0003
    lDatasets = cfg['selectedDatabase']['dataref'] + cfg['selectedDatabase']['dataart']
    iLabels = cfg['selectedDatabase']['labelref'] + cfg['selectedDatabase']['labelart'] #[0,1]
    for ipat, pat in enumerate(dbinfo.lPats):
        for iseq, seq in enumerate(lDatasets):
            # patches and labels of reference/artifact
            #import utils.DataPreprocessing as datapre
            tmpPatches, tmpLabels  = datapre.fPreprocessData(os.path.join(dbinfo.sPathIn, pat, dbinfo.sSubDirs[1], seq), cfg['patchSize'], cfg['patchOverlap'], 1 )
            dAllPatches = np.concatenate((dAllPatches, tmpPatches), axis=2)
            dAllLabels = np.concatenate((dAllLabels, iLabels[iseq]*tmpLabels), axis=0)
            dAllPats = np.concatenate((dAllPats,ipat*np.ones((tmpLabels.shape[0],1), dtype=np.int)), axis=0)

    # perform splitting
    X_train, y_train, X_test, y_test = ttsplit.fSplitDataset(dAllPatches, dAllLabels, dAllPats, cfg['sSplitting'], patchSize, cfg['patchOverlap'], cfg['dSplitval'], '')

    # save to file (deprecated)
    # sio.savemat(sOutPath + os.sep + sFSname + str(patchSize[0]) + str(patchSize[1]) + '_input.mat', {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'patchSize': cfg['patchSize']})
    with h5py.File(sDatafile, 'w') as hf:
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('X_test', data=X_test)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('y_test', data=y_test)
        hf.create_dataset('patchSize', data=patchSize)
        hf.create_dataset('patchOverlap', data=cfg['patchOverlap'])

# perform training
for iFold in range(0,len(X_train)-1):
    cnn_main.fRunCNN({'X_train': X_train[iFold], 'y_train': y_train[iFold], 'X_test': X_test[iFold], 'y_test': y_test[iFold], 'patchSize': patchSize}, cfg['network'], lTrain, cfg['sOpti'], sOutPath, cfg['batchSize'], cfg['lr'], cfg['epochs'])
