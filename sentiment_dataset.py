"""
Module holding dataset methods

Author pharnoux

"""

import os
import json
import math
import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset

import boto3


s3_access = boto3.resource('s3')

#as seen 
def train_input_fn(training_dir, file_fromS3, config):
    return _input_fn(training_dir, file_fromS3, config, "train")

def validation_input_fn(training_dir, file_fromS3, config):
    return _input_fn(training_dir, file_fromS3, config, "validation")

def eval_input_fn(training_dir, file_fromS3, config):
    return _input_fn(training_dir, file_fromS3, config, "eval")

def serving_input_fn(_, config):
   
    tensor = tf.placeholder(dtype=tf.float32, shape=[1, config["embeddings_vector_size"]])
    inputs = {config["input_tensor_name"]: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def _load_json_file(s3_name, config):

    features = []
    labels = []

    #with open(json_path, "r") as file:
    #directly extracting from S3
    #https://stackoverflow.com/questions/31976273/open-s3-object-as-a-string-with-boto3
    obj = s3_access.Object('aiops-assign6',s3_name )
    file = obj.get()['Body'].read().decode().split('\n')
    
    for line in file:

        entry = json.loads(line)
        
        
        if len(entry["features"]) != config["padding_size"]:
            raise ValueError("The size of the features of the entry with twitterid {} was not expected".format(\
                        entry["twitterid"]))
        labels.append(entry["sentiment"] / 4)
        features.append(entry["features"])

    return features, labels

def _input_fn(directory, file_name, config, mode):

    print("Fetching {} data...".format(mode))

    
    all_features = []
    all_labels = []

    
    features, labels = _load_json_file(file_name, config)
    all_features += features
    all_labels += labels

    num_data_points = len(all_features)
    print(num_data_points)
    num_batches = math.ceil(len(all_features) / config["batch_size"])

    dataset = Dataset.from_tensor_slices((all_features, all_labels))

    if mode == "train":

        dataset = Dataset.from_tensor_slices((all_features, all_labels))
        dataset = dataset.batch(config["batch_size"], drop_remainder=True).shuffle(10000, seed=12345).repeat(
            config["num_epoch"])
        num_batches = math.ceil(len(all_features) / config["batch_size"])

    if mode in ("validation", "eval"):

        dataset = dataset.batch(config["batch_size"], drop_remainder=True).repeat(config["num_epoch"])
        num_batches = math.ceil(len(all_features) / config["batch_size"])

    iterator = dataset.make_one_shot_iterator()
    dataset_features, dataset_labels = iterator.get_next()

    return [{config["input_tensor_name"]: dataset_features}, dataset_labels,
            {"num_data_point": num_data_points, "num_batches": num_batches}]
