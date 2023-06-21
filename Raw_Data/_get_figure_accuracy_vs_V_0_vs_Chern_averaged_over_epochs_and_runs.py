#!/usr/bin/env python3
"""
Created on Tue Apr 23 12:53:35 2019
@author: marcin
"""
# import theano
import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dropout

# Neural Network in Keras
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

encoder = LabelBinarizer()
import random

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from scipy import stats

#%%

##%% Accuracy vs epochs : averaged over all runs for all V_0
# from scipy import stats
NN_architecture_vec = np.array([0, 1, 2])
padding_str_vec = ["same"]
Number_of_epochs = 100

#% Accuracy vs V_0 : averaged over all epochs and all runs

preselection = 1
Batch_size = 64
if preselection == 0:
    title_preselection_string = "with preselection"
    selection_string = "_preselection.0"

if preselection == 1:
    title_preselection_string = "without preselection"
    selection_string = "_preselection.1"

C_bulk_vec = np.array([0, 1, 2, 3])
V_0_vec = np.array(
    [
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
    ]
)
# V_0_vec = np.array([0,0.2,0.4,0.6,0.8])


N_runs_vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
N_runs = N_runs_vec.shape[0]
N_epochs_max = 100
accuracy_tensor = np.zeros(
    (N_runs, N_epochs_max, V_0_vec.shape[0], C_bulk_vec.shape[0])
)
learning_rate = 0.001

string_V_max = "_V_max.1"


for NN_architecture in NN_architecture_vec:

    if NN_architecture == 0:
        architecture_string = "_dense"

    if NN_architecture == 1:
        architecture_string = "_CNN_1"

    if NN_architecture == 2:
        architecture_string = "_CNN_2"

    data_all_runs = np.zeros((V_0_vec.shape[0], N_runs))
    data_accuracy_vs_epochs = np.zeros((100, N_runs))

    filename_accuracy_train = (
        "final_results_accuracy_vs_epochs_train_NN." + str(NN_architecture) + ".dat"
    )
    filename_accuracy_val = (
        "final_results_accuracy_vs_epochs_val_NN." + str(NN_architecture) + ".dat"
    )

    filename_loss_train = (
        "final_results_loss_vs_epochs_train_NN." + str(NN_architecture) + ".dat"
    )
    filename_loss_val = (
        "final_results_loss_vs_epochs_train_NN." + str(NN_architecture) + ".dat"
    )

    acc_train_mat = np.zeros((100, 10))
    acc_val_mat = np.zeros((100, 10))

    loss_train_mat = np.zeros((100, 10))
    loss_val_mat = np.zeros((100, 10))
    for run_i in N_runs_vec:

        parameters_NN = (
            string_V_max
            + selection_string
            + "_std_N_epochs."
            + str(Number_of_epochs)
            + "_NN."
            + str(NN_architecture)
            + "_Batch."
            + f"{Batch_size:03d}"
            + "_lr.0.0010_run."
            + f"{run_i:03d}"
        )

        filename_acc_train = (
            "accuracy_keras_vs_epochs_training_D" + parameters_NN + ".callback_history"
        )
        filename_acc_val = (
            "accuracy_keras_vs_epochs_validating_D"
            + parameters_NN
            + ".callback_history"
        )

        filename_loss_train = (
            "loss_keras_vs_epochs_training_D" + parameters_NN + ".callback_history"
        )
        filename_loss_val = (
            "loss_keras_vs_epochs_validating_D" + parameters_NN + ".callback_history"
        )

        data_acc_train = np.loadtxt(filename_acc_train)
        data_acc_val = np.loadtxt(filename_acc_val)

        data_loss_train = np.loadtxt(filename_loss_train)
        data_loss_val = np.loadtxt(filename_loss_val)

        acc_train_mat[:, run_i] = data_acc_train[:, 1]
        acc_val_mat[:, run_i] = data_acc_val[:, 1]

    #        loss_train_mat[:,run_i] = data_loss_train[:,1]
    #        loss_val_mat[:,run_i] = data_loss_val[:,1]
    #%%
    fileID_acc_train = open(filename_accuracy_train, "w")
    fileID_acc_val = open(filename_accuracy_val, "w")
    fileID_loss_train = open(filename_loss_train, "w")
    fileID_loss_val = open(filename_loss_val, "w")
    for i in range(0, 100):
        fileID_acc_train.write(
            str(i)
            + " "
            + str(np.mean(acc_train_mat[i, :]))
            + " "
            + str(np.std(acc_train_mat[i, :]))
            + "\n"
        )
        fileID_acc_val.write(
            str(i)
            + " "
            + str(np.mean(acc_val_mat[i, :]))
            + " "
            + str(np.std(acc_val_mat[i, :]))
            + "\n"
        )
    #        fileID_loss_train.write( str(i) + " " + str(np.mean(data_loss_train[i,:])) + " " + str(np.std(data_loss_train[i,:])) + "\n" )
    #        fileID_loss_val.write( str(i) + " " + str(np.mean(data_loss_val[i,:])) + " " + str(np.std(data_loss_val[i,:])) + "\n" )

    fileID_acc_train.close()
    fileID_acc_val.close()
    fileID_loss_train.close()
    fileID_loss_val.close()
