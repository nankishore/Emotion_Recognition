# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 01:48:57 2016
@author: nandakishore
"""
import sys
import pickle
import numpy as np
from sklearn.svm import SVC
import os
import cv2
import random
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn import preprocessing
from sklearn.externals import joblib

#Setup Project root directory. Folders required by the program will be automatiocally be created
projectRoot = "Emotion_Recognition/"

#Caffe root directory path
caffeRoot = "caffe-master/"

#Directory paths to save face patches, model and output results
faceDataset = projectRoot + "facepatches/"
imgFeatures = projectRoot + "cache-features/"
googleNetDir = caffeRoot + 'models/bvlc_googlenet/'
prototextFile = googleNetDir + 'deploy.prototxt'
googleNetModel = googleNetDir + 'bvlc_googlenet.caffemodel'
svmModel = projectRoot + "models-sigmoid-cv/"
outputResults = svmModel + "outputs/"

#Label assignment used throughtout the project
labels = {'0': "Happy", '1': "Sad", '2': "Normal", "3": "Other", "4": "False Face"}

#Check for the existence of a directory and create one if necessary
def checkDirectoryExistence(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    return

#Define system variable paths. Give the path for libcuda*.so files here
def setSystemVariablePath():
    caffePython = caffeRoot + 'python'
    cudaLibPath = '/home/ubuntu/usr/local/cuda-7.5/lib64'
    sys.path.append(caffePython)
    sys.path.append(cudaLibPath)

#Randomly sample 2000 images from the training set and 100 samples from test image
def splitDataset(featPath):
    training = []
    test = []
    for eachDirClass in os.listdir(featPath):
        numSamples = os.listdir(featPath
                                + eachDirClass)
        temp = random.sample(numSamples,
            200)
        training.append(temp)
        testSamples = set(numSamples) - set(temp)
        testSamplesList = list(testSamples)
        test.append(random.sample(testSamplesList,
            20))
    return {'training': list(np.ravel(training)), 'test': [item for sublist in test for item in sublist]}

#Extract features from a set of images. Input images should be of dimension 224*224 only. Please resize the image before
#feeding it to this function.
def extractFeatures(
        netModel, featureExtractionLayer,
        dataPath):
    t0 = time()
    counter  = 0
    for eachDirClass in os.listdir(dataPath):
        newDir = imgFeatures
                 + eachDirClass
                 + '/'
        checkDirectoryExistence(newDir)
        for eachSubFile in os.listdir(dataPath + eachDirClass):
            counter += 1
            img = cv2.imread(dataPath
                             + eachDirClass
                             + '/'
                             + eachSubFile)
            netModel.predict([img])
            featureVectors = netModel.blobs[featureExtractionLayer].data
            featFile = open(newDir
                            + eachSubFile[0:-4]
                            + '.b',
                            'wb')
            pickle.dump(featureVectors.flatten(),
                featFile)
            featFile.close()
            print "Currently processed: ", counter
            print np.shape(featureVectors.flatten())
    print "Feature extraction complete .."
    print("done in %0.3fs" % (time() - t0))
    return

#Extract class labels for the images created in the data split stage.
def extractClassLabels(train,
        test):
    trainClass = []
    testClass = []
    for trainFiles in train:
        className = trainFiles.split('-')
        trainLabel = labels.keys()[labels.values().index(className[0])]
        trainClass.append(int(trainLabel))
    for testFiles in np.ravel(test):
        className = testFiles.split('-')
        testLabel = labels.keys()[labels.values().index(className[0])]
        testClass.append(int(testLabel))
    print "Class labels created.."
    return {'trainClass': trainClass, 'testClass': testClass}

#function to read feature vectors
def getFeatureFiles(imgFeatPath):
    featureFiles = []
    classDir = os.listdir(imgFeatPath)
    for eachClass in classDir:
        classSamples = os.listdir(imgFeatPath
                                  + eachClass)
        featureFiles.append(classSamples)
    return [item for sublist in featureFiles for item in sublist]

#Create image feature matrix, used to feed the feature vector matrix to SVM.
def createImgFeatureMatrix(featDir,
        featFilesList):
    featMatrix = []
    for eachFeat in featFilesList:
        dirClass = eachFeat.split('-')[0]
        f = open(featDir
                 + dirClass
                 + '/'
                 + eachFeat,
                 'rb')
        features = pickle.load(f)
        featMatrix.append(features)
    print "Created feature matrix.."
    return np.array(featMatrix)

#Train SVM using Sigmoid kernel. Perform grid search over a set of parameters for C and Gamma with 5 fold cross validation.
def trainSVM(feat, classes):
    print "Initiated SVM Training.."
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {
                  'C': list(np.logspace(-10, 10, base=2)),
                  'gamma': list(np.logspace(-6, -1, 10)),
                  'tol': [0.0001]
                 }
    model = GridSearchCV(
        SVC(kernel='sigmoid', probability=True, verbose=True), param_grid, cv = 5,
        n_jobs = -1, scoring='accuracy')
    model.fit(feat, classes)
    print("done in %0.3fs" % (time() - t0))
    print "Grid Search Complete.. SVM is now fit with best parameters..!"
    return model

def main():
    setSystemVariablePath()
    checkDirectoryExistence(imgFeatures)
    checkDirectoryExistence(svmModel)
    checkDirectoryExistence(outputResults)
    import caffe
    #Select layer name from which features have to be extracted.
    featureExtractionLayer = 'pool5/7x7_s1'
    caffe.set_mode_gpu()
    #Initialize the GoogLeNet model.
    myNetwork = caffe.Classifier(
        prototextFile, googleNetModel, channel_swap=(2,1,0),
        raw_scale=255, image_dims=(224, 224))
    extractFeatures(
        myNetwork, featureExtractionLayer,
        faceDataset)
    splitData = splitDataset(imgFeatures)
    classLabels = extractClassLabels(splitData['training'],
        splitData['test'])
    trainFeatMatrix = createImgFeatureMatrix(imgFeatures,
        splitData['training'])
    #Normalize the feature vectors for zero mean and zero variance
    normalizer = preprocessing.Normalizer().fit(trainFeatMatrix)
    trainFeatMatrix = normalizer.transform(trainFeatMatrix)
    print "Normalization Complete.."
    print "Training feature matrix dimensions: ", np.shape(trainFeatMatrix)
    print "Class Labels Shape: ", np.shape(classLabels['trainClass'])
    mySVMModel = trainSVM(trainFeatMatrix,
        classLabels['trainClass'])
    #Save SVM model files.
    joblib.dump(mySVMModel,
                svmModel
                + 'mySVMModel.pkl')
    print "Model has been saved in '<project-root>/models/' directory"
    print "\nComplete Score output from grid: "
    mySVMModel.grid_scores_
    print "\nBest Model Parameters: "
    mySVMModel.best_score_
    mySVMModel.best_params_
    print("\nBest estimator found by grid search:")
    print(mySVMModel.best_estimator_)
    with open(outputResults +'gridScores.b', 'wb') as f:
        pickle.dump(mySVMModel.grid_scores_, f)
    with open(outputResults +'normalizerBackup.b', 'wb') as f1:
        pickle.dump(normalizer, f1)
    joblib.dump(normalizer,
                outputResults
                + 'NormalizerValues.pkl')
    testFeatMatrix = createImgFeatureMatrix(imgFeatures,
        splitData['test'])
    testFeatMatrix = normalizer.transform(testFeatMatrix)
    testResults = mySVMModel.predict(testFeatMatrix)
    print "Test feature matrix Shape: ", np.shape(testFeatMatrix)
    print "Class Labels Shape: ", np.shape(classLabels['testClass'])
    testReport = classification_report(classLabels['testClass'],
        testResults)
    testConfuMatrix = confusion_matrix(
        classLabels['testClass'], testResults,
        labels=range(3))
    with open(outputResults +'report.b', 'wb') as f2:
        pickle.dump((testReport, testConfuMatrix, outputResults),
            f2)

if __name__== "__main__":
    main()
