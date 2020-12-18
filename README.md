# Image Classifier
Udacity NanoDegree - Machine Learning

## Description
The goal of this project is to develop an Image Classifier. The project has two parts. The first part of the project is to code in Jupyter Notebook to implement an image classifer with PyTorch. The second part of the project is to convert the code into a command line application that others can use. 

## Installation
Install Anaconda libraries of Python. The code should run with no issues using Python versions 3.*.
For the most part of the code, local device 'CPU' was used for processing. GPU is not necessary to run the code. 

## File Descriptions
The first part of the project contains three files:
1. Image Classifier Project.ipynb - Jupyter Notebook to implement an image classifer with PyTorch
2. Image Classifier Project.HTML - HTML copy of Jupyter Notebook
3. savedcheckpoint.pth - the checkpoint file saved from running the Jupyter Notebook. 

Data was provided and available through Udacity's workspace

The second part of the project contains 2 command line applications:
1. Train.py
Usage from command line: python train.py data_directory
This will train a new network on a dataset and save the model as a checkpoint.

2. Predict.py
Usage:python predict.py /path/to/image checkpoint
This will use the trained network from train.py to predict the class for an input image.

## Licensing
Must give credit to Udacity for the data.
