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
from sklearn.externals import joblib

#Check for the existence of a directory and create one if necessary
projectRoot = "Emotion_Recognition/"
caffeRoot = "caffe-master/"
faceDataset = projectRoot + "facepatches/"
imgFeatures = projectRoot + "cache-features/"
googleNetDir = caffeRoot + 'models/bvlc_googlenet/'
prototextFile = googleNetDir + 'deploy.prototxt'
googleNetModel = googleNetDir + 'bvlc_googlenet.caffemodel'
svmModel = projectRoot + "models-sigmoid-cv/"
outputResults = svmModel + "outputs/"
#Predicted outputs will be written to this folder.
predictionResults = projectRoot + 'predictions/'
labels = {'0': "Happy", '1': "Sad", '2': "Normal", "3": "Other", "4": "False Face"}
face_cascade = cv2.CascadeClassifier("Emotion_Recognition/haarcascade_frontalface_default.xml")
color_paletes = [
    (74,184,247),(247,184,74),(74,247,184),(224,247,184),
    (224,255,255),(153,153,255),(255,255,153)]
fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontScale = 0.8
thickness = 1

def checkDirectoryExistence(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    return

#Please set the path to your cuda installation directory
def setSystemVariablePath():
    caffePython = caffeRoot + 'python'
    cudaLibPath = '/home/ubuntu/usr/local/cuda-7.5/lib64'
    sys.path.append(caffePython)
    sys.path.append(cudaLibPath)

#Extract features from the pretrained model
def extractFeatures(netModel, featureExtractionLayer, imgPatch):
    netModel.predict([imgPatch])
    featureVectors = netModel.blobs[featureExtractionLayer].data
    print "Feature vector dimensions: "np.shape(featureVectors.flatten())
    return featureVectors.flatten()

#Load a saved model
def loadModel(modelPath, modelName):
    model = joblib.load(modelPath + modelName, 'r')
    return model

#A single pipeline to perform facial emotion recognition on a given image.
#Read image from command line
#Perform face detection
#classify the detected face to one of the three classes.
#write the output image
def main():
    inputImage = sys.argv[1]
    setSystemVariablePath()
    checkDirectoryExistence(imgFeatures)
    checkDirectoryExistence(svmModel)
    checkDirectoryExistence(predictionResults)
    import caffe
    featureExtractionLayer = 'pool5/7x7_s1'
    caffe.set_mode_gpu()
    myNetwork = caffe.Classifier(
        prototextFile, googleNetModel, channel_swap=(2,1,0),
        raw_scale=255, image_dims=(224, 224))
    imgCopy = cv2.imread(inputImage)
    gray = cv2.cvtColor(imgCopy,
        cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor = 1.05,
        minNeighbors = 5, minSize=(30, 30))
    clf = loadModel(svmModel,
        "mySVMModel.pkl")
    normalizer = joblib.load(outputResults
                             + 'NormalizerValues.pkl',
                             'r')

    assert len(faces) > 0, "No face has been detected in the given image. Please input another image."
    for num_faces_detected in range(0, len(faces)):
        img = cv2.imread(inputImage)
        print "Your input image shape is: ", np.shape(img)
        print "Number of faces detected in the image: "faces[num_faces_detected]
        bbox_x, bbox_y, bbox_width, bbox_height = faces[num_faces_detected]
        facePatch = img[bbox_x:bbox_x+bbox_width, bbox_y:bbox_y+bbox_height]
        try:
            feat = extractFeatures(
                myNetwork, featureExtractionLayer,
                cv2.resize(facePatch, (224,224)))
            features = normalizer.transform(feat)
            result = clf.predict(features)
            resultLabel = labels[str(result[0])]
            color = random.choice(color_paletes)
            text_size, baseline = cv2.getTextSize(
                resultLabel, fontFace,
                fontScale, thickness)
            text_width, text_height = text_size
            cv2.rectangle(
                imgCopy, (bbox_x-1,bbox_y-(text_height*2)), (bbox_x+(text_width), bbox_y),
                color, -1)
            cv2.rectangle(
                imgCopy, (bbox_x,bbox_y), (bbox_x+bbox_width, bbox_y+bbox_height),
                color, 2)
            cv2.putText(
                imgCopy, resultLabel, (bbox_x-1,bbox_y-(text_height/2)),
                fontFace,fontScale,(0,0,0),
                thickness)
        except cv2.error as e:
            continue
    cv2.imwrite(predictionResults
                + inputImage,
                imgCopy)

if __name__== "__main__":
    main()
