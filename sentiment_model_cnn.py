"""
Model definition for CNN sentiment training


"""

import os
import tensorflow as tf


def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """

    cnn_model = None

    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """

    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))

"""
Model definition for CNN sentiment training

"""

import os
import tensorflow as tf
import numpy as np


#imports for creating the layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPool1D


#imports for extracting data from s3
import boto3

def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling
    """
    # This step is to load the glove dictionary file into the memory  
    embeddings = {}
    s3_access = boto3.resource('s3')
    glove_file = s3_access.Object('aiops-assign6', config["embeddings_path"]).get()['Body'].read().decode("utf-8").split('\n')
    #print('loading done')
    
    embedding_matrix = np.zeros((config["embeddings_dictionary_size"], config["embeddings_vector_size"]))
    
    
    ind = 0
    for line in glove_file:
        ind=ind+1
        split_line = line.split()
        #print(split_line[0])
        #due to issue with last line
        if ind>1193515:
            break
        dict_key = split_line[0]
        embeddings[dict_key] = np.array(split_line[1:], dtype='float32')
        #<unknown> has been manually removed
        embedding_matrix[ind]=embeddings[dict_key]
    
    
    print('Load '+str(len(embeddings))+' word vectors.')

     

    # creating the cnn-model
    model = Sequential()
    #embedding layer with given input and output features
    model.add(Embedding(input_length = config["padding_size"], input_dim = config["embeddings_dictionary_size"]\
                        ,output_dim = 25, weights=[embedding_matrix], trainable=True, name='embedding'))
    
    #Convolution1D layer with given features
    model.add(Conv1D(100, kernel_size = 2, strides = 1, padding = 'valid', activation = 'relu'))

    #GLobalMaxPool1D layer
    model.add(GlobalMaxPool1D())

    #Dense layer1
    model.add(Dense(100, activation = 'relu'))

    #Dense Layer2
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    cnn_model = model

    
    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """

    model.save(output)

    print("Model successfully saved at: {}".format(output))
