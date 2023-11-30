#!/usr/bin/env python3

import os
import io

import numpy as np
import itertools
import pickle
import time
import matplotlib.pyplot as plt

from datetime import datetime

import argparse
import csv

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.activations import *

from tensorflow.keras import backend as K

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score


from tensorflow.keras.models import load_model


from classifiers import LSTM

from main import load_data

batch = 1024

cvscores =[]

def evaluate_model(model, testX, testy, batch_size):
    
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    
    return accuracy

if __name__ == '__main__':
    # Define paths to your saved model and testing dataset
    model_path = 'results/LSTM/run_ARUBA_Preprocessed_25_padded_batch_1024_epoch_400_early_stop_mask_0.0_2023_08_10_03_24_39/LSTM_ARUBA_Preprocessed_25_padded_batch_1024_epoch_400_early_stop_mask_0.0_BEST_0.h5'
    input_file = 'ARUBA_Preprocessed_25_padded'
    path = 'Subsets'
    
    X_TRAIN, Y_TRAIN, X_VALIDATION, Y_VALIDATION, X_TEST, Y_TEST, listActivities = load_data(input_file, path)


    x_test = X_TEST.reshape(X_TEST.shape[0],X_TEST.shape[1],1)
    y_test = to_categorical(Y_TEST)

    saved_model = tf.keras.models.load_model(model_path)
    score = evaluate_model(saved_model, x_test, y_test, batch)

    # store score
    cvscores.append(score)
    
    print('Accuracy: %.3f' % (score * 100.0))

    ##########_GENERATE_##########

    # Make prediction using the model
    Y_hat = saved_model.predict(x_test)
    Y_pred = np.argmax(Y_hat, axis=1)
    Y_pred = Y_pred.reshape(Y_pred.shape[0], 1)
    Y_pred = Y_pred.astype('int32')
    Y_test = Y_TEST.astype('int32')

    report = classification_report(Y_test, Y_pred, target_names=listActivities)
    print(report)
