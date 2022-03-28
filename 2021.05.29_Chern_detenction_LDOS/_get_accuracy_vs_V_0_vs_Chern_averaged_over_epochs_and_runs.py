#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:53:35 2019
@author: marcin
"""
import time
import numpy as np
import matplotlib.pyplot as plt
#import theano
import os
import keras
from keras.layers import Dropout
# Neural Network in Keras
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from sklearn.utils import shuffle
encoder = LabelBinarizer()
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import MaxPooling2D  
import random
from scipy import stats

from keras.models import load_model

Nx = 24
Ny = Nx
Nx_half = 12
Ny_half = 12

#%%

def fold(LDOS,Nx,Ny):
  
    A = LDOS.reshape(Nx,Ny)
    Nx_half = int(Nx/2)
    Ny_half = int(Ny/2)
    part_L_Up = A[0:Nx_half,0:Ny_half]
    part_L_Down = A[Nx_half:Nx,0:Ny_half]
    part_R_up = A[0:Nx_half,Ny_half:Ny]
    part_R_down = A[Nx_half:Nx,Ny_half:Ny]    
    stack = np.zeros((Nx_half,Nx_half,4))
    stack[:,:,0] = part_L_Up
    stack[:,:,1] = np.rot90(part_L_Down,3)
    stack[:,:,2] = np.rot90(part_R_up,1)
    stack[:,:,3] = np.rot90(part_R_down,2)
 
    
    return stack#.reshape(1,int(Nx/2)*int(Ny/2)*4)

    
 
A = np.array([[3,4,5,1,2,3],
              [2,0,0,0,0,4],
              [1,0,0,0,0,5],
              [5,0,0,0,0,1],
              [4,0,0,0,0,2],
              [3,2,1,5,4,3]])
    
A2 = fold(A,6,6) 
print(A2)

A3 = A2.reshape(3,3,4)
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
index_S_V  = 7
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

D_X_testing_raw  = D_testing_data_raw[:,index_S_LDOS_start:]                 
D_y_testing_raw_integers  =   np.abs(np.rint(D_testing_data_raw[:,index_S_C_bulk]))


D_y_validating_categorical     =   keras.utils.to_categorical(D_y_testing_raw_integers, num_classes)



for i in range(D_X_testing_raw.shape[0]):
    mean = np.mean(D_X_testing_raw[i,:])
    maximum = np.max(D_X_testing_raw[i,:])
    std = np.std(D_X_testing_raw[i,:])
    if(mean>0):    
        D_X_testing_raw[i,:] = D_X_testing_raw[i,:]/std#maximum


 
#%% 

##%% Accuracy vs epochs : averaged over all runs for all V_0
# from scipy import stats
NN_architecture_vec = np.array([0,1,2])
padding_str_vec = [ "same"]
Number_of_epochs = 100

#% Accuracy vs V_0 : averaged over all epochs and all runs

preselection = 1
Batch_size = 64
if(preselection == 0):
    title_preselection_string = "with preselection"
    selection_string = "_preselection.0"
    
if(preselection == 1):
    title_preselection_string = "without preselection"
    selection_string = "_preselection.1"
 
C_bulk_vec = np.array([0,1,2,3])
V_0_vec = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8,0.9,1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
# V_0_vec = np.array([0,0.2,0.4,0.6,0.8])

 
N_runs_vec = np.array([0,1,2,3,4,5,6,7,8,9])
N_runs = N_runs_vec.shape[0]
N_epochs_max = 100
accuracy_tensor = np.zeros((N_runs,N_epochs_max,V_0_vec.shape[0],C_bulk_vec.shape[0]))
learning_rate = 0.001

string_V_max = "_V_max.1"
for NN_architecture in NN_architecture_vec:
    if(NN_architecture == 0):
        architecture_str = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_CNN_unfolded"
    if(NN_architecture == 1):
        architecture_str = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_CNN_folded"             
    if(NN_architecture == 2):
        architecture_str = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_CNN_pool"
        


    for run_i in N_runs_vec:    
         
        if(NN_architecture == 0):
            architecture_string = "_dense"
            title_string = "NN.0"
            parameters_NN = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_NN." + str(NN_architecture) + "_Batch." + "{:03d}".format(Batch_size) + "_lr.0.0010_run." + "{:03d}".format(run_i)               
            appendix_str = ".keras"

        if(NN_architecture == 1):
            architecture_string = "_CNN_1"
            title_string = "NN.1"
            parameters_NN = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_NN." + str(NN_architecture) + "_Batch." + "{:03d}".format(Batch_size) + "_lr.0.0010_run." + "{:03d}".format(run_i)               
            appendix_str = ".keras"
            
        if(NN_architecture == 2):
            architecture_string = "_CNN_2"
            title_string = "NN.2"
            parameters_NN = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_NN." + str(NN_architecture) + "_Batch." + "{:03d}".format(Batch_size) + "_lr.0.0010_run." + "{:03d}".format(run_i)               
            appendix_str = ".keras"            
            
        accuracy_averaged_over_epochs = 0
        for epoch in range(0,N_epochs_max):
            
            filepath = "model_keras"+parameters_NN + "_epochs." + "{:03d}".format(epoch+1) + appendix_str
            model = load_model(filepath)
           
            for V_0_i in range(0,V_0_vec.shape[0]):
                for C_bulk in C_bulk_vec:
                    V_0 = V_0_vec[V_0_i]
                    string = " | run = " + str(run_i) + " | Epoch = " + str(epoch+1) +  " | V_0 = " + str(V_0) + " and C_bulk = " + str(C_bulk)   
                    idx_V_0 = np.where( np.logical_and(D_testing_data_raw[:,index_S_V] == V_0, D_y_testing_raw_integers[:] ==  C_bulk  ))[0]
                
                    if(NN_architecture == 1 or NN_architecture == 2):  
                        LDOS = D_X_testing_raw[idx_V_0,:].reshape(idx_V_0.shape[0],24,24,1)   
                    else:
                        LDOS = D_X_testing_raw[idx_V_0,:]
                    C_given     = D_y_testing_raw_integers[idx_V_0]
                    C_predicted = model.predict_classes(LDOS)
                    total = idx_V_0.shape[0]
#                    
                    correct = (np.where(C_predicted == C_given)[0]).shape[0] 
                    accuracy = correct/(1.0*total)
                    accuracy_tensor[run_i,epoch,V_0_i,C_bulk] = accuracy
                    string_accuracy = " | #samples = " + "{:04d}".format(total) + "|  #correct = " + "{:04d}".format(correct) + " | accuracy = " + "{:2.4f}".format(accuracy) 
                    print(title_string + string + " " + string_accuracy   )                    
                print("\n")
                
    print("Finished predictions ...")
    string_accuracy_tensor = architecture_string + "_500k_std_N_epochs." + str(Number_of_epochs) + "_Batch." + "{:03d}".format(Batch_size) + "_learning_rate." + "{:2.4f}".format(learning_rate)
    np.save("accuracy_tensor" + string_accuracy_tensor + ".mat" ,accuracy_tensor)
            
    N_epoch_saturated = 20

    for run_i in N_runs_vec:
        for C_bulk in C_bulk_vec:
            fileID = open("results_keras_testing_data_V_max_2_preselection.0_accuracy_vs_V_0_C_bulk." + str(C_bulk) + selection_string + architecture_string + "_run." + "{:03d}".format(run_i) + ".data_results","w")
            for V_0_i in range(0,V_0_vec.shape[0]):
                V = V_0_vec[V_0_i]
                string_file_ID = str(V) + " " + str(np.mean(accuracy_tensor[run_i,N_epoch_saturated:,V_0_i,C_bulk])) + " " + str(np.std(accuracy_tensor[run_i,N_epoch_saturated:,V_0_i,C_bulk])) + "\n"
                fileID.write(string_file_ID)
            fileID.close()

        fileID = open("results_keras_testing_data_V_max_2_preselection_0._accuracy_vs_V_0_C_bulk_averaged" + selection_string + architecture_string +  "_run." + "{:03d}".format(run_i)  + ".data_results","w")
        for V_0_i in range(0,V_0_vec.shape[0]):
            V = V_0_vec[V_0_i]
            string_file_ID = str(V) + " " + str(np.mean(accuracy_tensor[run_i,N_epoch_saturated:,V_0_i,:])) + " " + str(np.std(accuracy_tensor[run_i,N_epoch_saturated:,V_0_i,:])) + "\n"
            fileID.write(string_file_ID)
        fileID.close()
 