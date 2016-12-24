# Emotion Recognition using Convolutional Neural Network

Repository details:

All the scripts in this repository require the installation of OpenCV, Caffe and Cuda. Please set the respective paths in the script according to your installation.

1. trainModelGridSearch.py - train the SVM classifier layer on the features obtained from GoogLeNet. Dataset used in training the model is available upon request.

2. "models-sigmoid-cv" directory consists of a trained Emotion Classifier mdoel and the Normalizer transform value used on the training set.

3. predictEmotion.py - perform Emotion Classification Task on any given input image.

4. execution command - python predictEmotion.py "path to your image"

5. predictEmotion.py will search for faces in the input image using OpenCV Haar cascade classifier and feeds the face patch into the CNN. The predicted emotion of the face will be written on the output image with bounding box annotation. Output image path - 'predictions/' directory which will be created on runtime.
