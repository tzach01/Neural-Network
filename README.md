# Open Source Neural Network Generator Program

# Goal
Create a free tool that can help a user quickly generate and train data models with any dataset they have without having to code themselves.

# Description
Creating a simple tool that can be used to predict user determined values by generating and running a deep neural network using Keras with a TensorFlow Backend. As this is my first attempt at learning a programming language, I am open to hearing better ways of doing things and new things I should be thinking about.

The intent is for a user to be able to upload any file (currently only a .csv file) with a table of organized data and set the parameters of the deep neural network. The program will be able to complete one hot encoding, avoid multi-colinearity, normalize data, and transform it in a way to be accepted by the Keras and Tensorflow libraries. 

The same cleaning will be applied to an analysis file also uploaded by the user in addition to any missing columns being added, extra columns being deleted, and columns automatic sorting of columns to match the training file. Users will be able to set the number of hidden layers, the number of nodes in the initial layer and all of the hidden layers, and the epochs that should be run.

The user will also define a file location for the program to output the final results in a .csv file that merges the results with the original analysis data for easy comparison.

In the future, I would like to include other functions such as decision tree classifier, linear regression, recurrent neural networks, and  convolution neural networks. I would also like to figure out how to use TKinter to display results or show which features have the highest correlation. I have not learning TensorFlow 2.0, but would be interested in determining how to incorporate changes made to that library.

# Dependencies include:
    tkinter
    numpy
    pandas
    scikit-learn
    keras
      TensorFlow v1 - Note currently only compatible with Python version 3.6 and lower
    
# Files included:
  Neural Network Project with GUI.py - Python code that includes the code for the GUI as well as the generator for a neural network
  
  sales_data_training.csv - Demo data for video game sales used for training model
  
  sales_data_test.csv - Demo data for video game same used for analysis and prediction
