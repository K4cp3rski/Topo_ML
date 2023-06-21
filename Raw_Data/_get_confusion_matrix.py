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

Nx = 24
Ny = Nx
Nx_half = 12
Ny_half = 12

#%%


def fold(LDOS, Nx, Ny):

    A = LDOS.reshape(Nx, Ny)
    Nx_half = int(Nx / 2)
    Ny_half = int(Ny / 2)
    part_L_Up = A[0:Nx_half, 0:Ny_half]
    part_L_Down = A[Nx_half:Nx, 0:Ny_half]
    part_R_up = A[0:Nx_half, Ny_half:Ny]
    part_R_down = A[Nx_half:Nx, Ny_half:Ny]
    stack = np.zeros((Nx_half, Nx_half, 4))
    stack[:, :, 0] = part_L_Up
    stack[:, :, 1] = np.rot90(part_L_Down, 3)
    stack[:, :, 2] = np.rot90(part_R_up, 1)
    stack[:, :, 3] = np.rot90(part_R_down, 2)

    return stack  # .reshape(1,int(Nx/2)*int(Ny/2)*4)


A = np.array(
    [
        [3, 4, 5, 1, 2, 3],
        [2, 0, 0, 0, 0, 4],
        [1, 0, 0, 0, 0, 5],
        [5, 0, 0, 0, 0, 1],
        [4, 0, 0, 0, 0, 2],
        [3, 2, 1, 5, 4, 3],
    ]
)

A2 = fold(A, 6, 6)
print(A2)

A3 = A2.reshape(3, 3, 4)
print(A3)

#%% Import data

index_D_Nx = 0
index_D_N = 1
index_D_seed = 2
index_D_fraction_window = 3
index_D_mu = 4
index_D_alpha = 5
index_D_gap = 6
index_D_V = 7
index_D_C_bulk = 8
index_D_C_marker_clean = 9
index_D_C_marker_disorder = 10
index_D_no_states = 11
index_D_LDOS_start = 12


index_S_Nx = 0
index_S_seed = 1
index_S_fraction_window = 2
index_S_xi = 3
index_S_e0 = 4
index_S_kf = 5
index_S_gap = 6
index_S_V = 7
index_S_C_bulk = 8
index_S_C_marker_clean = 9
index_S_C_marker_disorder = 10
indes_S_no_states = 11
index_S_LDOS_start = 12

#%%

D_testing_data_raw = np.loadtxt("D_test_V_max.2_preselection.0.dat")

#%%

np.random.shuffle(D_testing_data_raw)


num_classes = 4

#%Extract LDOS (X) and labels (y)

D_X_testing_raw = D_testing_data_raw[:, index_S_LDOS_start:]
D_y_testing_raw_integers = np.abs(np.rint(D_testing_data_raw[:, index_S_C_bulk]))


D_y_validating_categorical = keras.utils.to_categorical(
    D_y_testing_raw_integers, num_classes
)


for i in range(D_X_testing_raw.shape[0]):
    mean = np.mean(D_X_testing_raw[i, :])
    maximum = np.max(D_X_testing_raw[i, :])
    std = np.std(D_X_testing_raw[i, :])
    if mean > 0:
        D_X_testing_raw[i, :] = D_X_testing_raw[i, :] / std  # maximum


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
#%%
V_0_vec = np.array([0, 0.4, 0.8, 1.2, 1.6, 1.9])

N_runs_vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
N_runs = N_runs_vec.shape[0]
N_epochs_max = 100
accuracy_tensor = np.zeros(
    (N_runs, N_epochs_max, V_0_vec.shape[0], C_bulk_vec.shape[0])
)
learning_rate = 0.001

string_V_max = "_V_max.1"

run_i = 9
epoch = 99

# fig, axs = plt.subplots(V_0_vec.shape[0], 3, sharex=True,sharey=True)

fig, axs = plt.subplots(3, V_0_vec.shape[0], sharex=True, sharey=True)
for NN_architecture in NN_architecture_vec:
    if NN_architecture == 0:
        architecture_str = (
            string_V_max
            + selection_string
            + "_std_N_epochs."
            + str(Number_of_epochs)
            + "_CNN_unfolded"
        )
    if NN_architecture == 1:
        architecture_str = (
            string_V_max
            + selection_string
            + "_std_N_epochs."
            + str(Number_of_epochs)
            + "_CNN_folded"
        )
    if NN_architecture == 2:
        architecture_str = (
            string_V_max
            + selection_string
            + "_std_N_epochs."
            + str(Number_of_epochs)
            + "_CNN_pool"
        )

    #    if(NN_architecture == 0):
    #        architecture_string = "_dense"
    #        title_string = "NN.0"
    #        parameters_NN = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_NN." + str(NN_architecture) + "_Batch." + "{:03d}".format(Batch_size) + "_lr.0.0010_run." + "{:03d}".format(run_i)
    #        appendix_str = ".keras"
    #
    #    if(NN_architecture == 1):
    #        architecture_string = "_CNN_1"
    #        title_string = "NN.1"
    #        parameters_NN = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_NN." + str(NN_architecture) + "_Batch." + "{:03d}".format(Batch_size) + "_lr.0.0010_run." + "{:03d}".format(run_i)
    #        appendix_str = ".keras"
    #
    #    if(NN_architecture == 2):
    #        architecture_string = "_CNN_2"
    #        title_string = "NN.2"
    #        parameters_NN = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_NN." + str(NN_architecture) + "_Batch." + "{:03d}".format(Batch_size) + "_lr.0.0010_run." + "{:03d}".format(run_i)
    #        appendix_str = ".keras"
    #
    #    accuracy_averaged_over_epochs = 0
    #
    #
    #    filepath = "model_keras"+parameters_NN + "_epochs." + "{:03d}".format(epoch+1) + appendix_str
    #    model = load_model(filepath)

    for V_i in range(0, V_0_vec.shape[0]):
        V = V_0_vec[V_i]

        idx = np.where(D_testing_data_raw[:, index_D_V] == V)[0]
        if NN_architecture == 1 or NN_architecture == 2:
            LDOS = D_X_testing_raw[idx].reshape(idx.shape[0], 24, 24, 1)
        else:
            LDOS = D_X_testing_raw[idx]
        C_given = D_y_testing_raw_integers[idx]

        confusion_matrix = np.zeros((4, 4))
        C_0_amount = 0
        C_1_amount = 0
        C_2_amount = 0
        C_3_amount = 0
        #        for run_i in range(0,10):
        #            for epoch in range(0,N_epochs_max):
        for run_i in range(0, 10):
            for epoch in range(20, 100):
                print(V_i, run_i, epoch)

                if NN_architecture == 0:
                    architecture_string = "_dense"
                    title_string = "NN.0"
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
                    appendix_str = ".keras"

                if NN_architecture == 1:
                    architecture_string = "_CNN_1"
                    title_string = "NN.1"
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
                    appendix_str = ".keras"

                if NN_architecture == 2:
                    architecture_string = "_CNN_2"
                    title_string = "NN.2"
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
                    appendix_str = ".keras"

                accuracy_averaged_over_epochs = 0

                filepath = (
                    "model_keras"
                    + parameters_NN
                    + "_epochs."
                    + "{:03d}".format(epoch + 1)
                    + appendix_str
                )
                model = load_model(filepath)

                C_predicted = model.predict_classes(LDOS)
                total = C_given.shape[0]

                for i in range(C_given.shape[0]):
                    if C_given[i] == 0:
                        C_0_amount += 1
                    if C_given[i] == 1:
                        C_1_amount += 1
                    if C_given[i] == 2:
                        C_2_amount += 1
                    if C_given[i] == 3:
                        C_3_amount += 1
                    confusion_matrix[int(C_given[i]), int(C_predicted[i])] += 1

        confusion_matrix[0, :] = confusion_matrix[0, :] / C_0_amount * 100
        confusion_matrix[1, :] = confusion_matrix[1, :] / C_1_amount * 100
        confusion_matrix[2, :] = confusion_matrix[2, :] / C_2_amount * 100
        confusion_matrix[3, :] = confusion_matrix[3, :] / C_3_amount * 100

        im = axs[NN_architecture, V_i].matshow(confusion_matrix)
        axs[NN_architecture, V_i].xaxis.set_ticks_position("bottom")
        #        axs[V_i,2].yaxis.set_label_position("right")
        #        axs[V_i,2].yaxis.tick_right()
        axs[0, V_i].set_title("$V_0$ = " + str(V))

        # set the spacing between subplots
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.32, hspace=0.1
        )
#
#        axs[0,0].set_title("Model A")
#        axs[0,1].set_title("Model B")
#        axs[0,2].set_title("Model C")

#        im = axs[V_i,NN_architecture].matshow(confusion_matrix)
#        axs[V_i,NN_architecture].xaxis.set_ticks_position('bottom')
##        axs[V_i,2].yaxis.set_label_position("right")
##        axs[V_i,2].yaxis.tick_right()
#        axs[V_i,2].annotate("$V_0$ = " + str(V),xy=(3.5, 2),fontsize=14)
#        # set the spacing between subplots
#        plt.subplots_adjust(left=0.1,
#                            bottom=0.1,
#                            right=0.5,
#                            top=0.9,
#                            wspace=0.0,
#                            hspace=0.1)
#
#        axs[0,0].set_title("Model A")
#        axs[0,1].set_title("Model B")
#        axs[0,2].set_title("Model C")

# plt.colorbar(im,ax=axs[0,5])
axs[0, 5].annotate(" Model A", xy=(3.5, 2), fontsize=12)
axs[1, 5].annotate(" Model B", xy=(3.5, 2), fontsize=12)
axs[2, 5].annotate(" Model C", xy=(3.5, 2), fontsize=12)
#%%

from keras.utils.vis_utils import plot_model, pydot

plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

import keras
import pydot
import pydotplus

#%%
import visualkeras
from keras.utils.vis_utils import model_to_dot, plot_model
from pydotplus import graphviz

keras.utils.vis_utils.pydot = pydot
import tensorflow as tf
from PIL import ImageFont

# font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!

filepath_A = "model_keras_V_max.1_preselection.1_std_N_epochs.100_NN.0_Batch.064_lr.0.0010_run.001_epochs.022.keras"
model_A = load_model(filepath_A)
filepath_B = "model_keras_V_max.1_preselection.1_std_N_epochs.100_NN.1_Batch.064_lr.0.0010_run.001_epochs.022.keras"
model_B = load_model(filepath_B)

filepath_C = "model_keras_V_max.1_preselection.1_std_N_epochs.100_NN.2_Batch.064_lr.0.0010_run.001_epochs.022.keras"
model_C = load_model(filepath_C)
# visualkeras.layered_view(model_A, legend=True).show()#, font=font)
# visualkeras.layered_view(model_B, legend=True).show()#, font=font)
# visualkeras.layered_view(model_C, legend=True).show()#, font=font)

# visualkeras.layered_view(model_A).show() # display using your system viewer
# visualkeras.layered_view(model_A, to_file='output.png') # write to disk
visualkeras.layered_view(
    model_A, to_file="fig_model_A_output.png", legend=True
)  # .show() # write and show
visualkeras.layered_view(
    model_B, to_file="fig_model_B_output.png", legend=True
)  # .show() # write and show
visualkeras.layered_view(
    model_C, to_file="fig_model_C_output.png", legend=True
)  # .show() # write and show
# from keras.utils.vis_utils import plot_model
# print(model_A.summary())

print(model_C.summary())

# print(model_C.summary())
##plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#
#
# dot_img_file = '/tmp/model_1.png'
# tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
#
###%%
##
##import pydot
##dot_img_file = 'fig_model_1.png'
##tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
#
##%%
# tf.keras.utils.model_to_dot(
#    model,
#    show_shapes=False,
##
#    show_layer_names=True,
##    rankdir="TB",
#    expand_nested=False,
#    dpi=96,
#    subgraph=False,
# )
