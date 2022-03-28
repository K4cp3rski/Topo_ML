#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:53:35 2019
@author: marcin
"""
import tensorflow
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
Nx = 24
Ny = Nx


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

string_V_max = "_V_max.1"

for preselection in np.array([1,0]):

    if(preselection == 0):
        D_training_data_raw = np.loadtxt("D_train" + string_V_max  + "_preselection.0.dat")
        D_validating_data_raw = np.loadtxt("D_validate" + string_V_max + "_preselection.0.dat")
        D_testing_data_raw = np.loadtxt("D_test" + string_V_max + "_preselection.0.dat")
        selection_string = "_preselection.0"
        
    if(preselection == 1):
        D_training_data_raw = np.loadtxt("D_train" + string_V_max + "_preselection.1.dat")
        D_validating_data_raw = np.loadtxt("D_validate" + string_V_max + "_preselection.1.dat")
        D_testing_data_raw = np.loadtxt("D_test" + string_V_max + "_preselection.1.dat")
        selection_string = "_preselection.1"
    
    

    
    np.random.shuffle(D_training_data_raw)
    np.random.shuffle(D_validating_data_raw)
    np.random.shuffle(D_testing_data_raw)
     
  
    
    num_classes = 4
    
    #%Extract LDOS (X) and labels (y)
    D_X_training_raw    = D_training_data_raw[:,index_D_LDOS_start:]
    D_X_validating_raw  = D_validating_data_raw[:,index_D_LDOS_start:]
    D_X_testing_raw  = D_testing_data_raw[:,index_S_LDOS_start:]
                      
    D_y_training_raw_integers    =   np.abs(np.rint(D_training_data_raw[:,index_D_C_bulk]))
    D_y_validating_raw_integers  =   np.abs(np.rint(D_validating_data_raw[:,index_D_C_bulk]))
    D_y_testing_raw_integers  =   np.abs(np.rint(D_testing_data_raw[:,index_S_C_bulk]))
    
     
    D_y_training_categorical       =   keras.utils.np_utils.to_categorical(D_y_training_raw_integers, num_classes)
    D_y_validating_categorical     =   keras.utils.np_utils.to_categorical(D_y_validating_raw_integers, num_classes)
    D_y_testing_categorical     =   keras.utils.np_utils.to_categorical(D_y_testing_raw_integers, num_classes)
    
    #LDOS normalization
    for i in range(D_X_training_raw.shape[0]):
        mean = np.mean(D_X_training_raw[i,:])
        maximum = np.max(D_X_training_raw[i,:])
        std = np.std(D_X_training_raw[i,:])
        if(mean>0):
            D_X_training_raw[i,:] = D_X_training_raw[i,:]/std#maximum
    
    for i in range(D_X_testing_raw.shape[0]):
        mean = np.mean(D_X_testing_raw[i,:])
        maximum = np.max(D_X_testing_raw[i,:])
        std = np.std(D_X_testing_raw[i,:])
        if(mean>0):    
            D_X_testing_raw[i,:] = D_X_testing_raw[i,:]/std#maximum
    
    for i in range(D_X_validating_raw.shape[0]):
        mean = np.mean(D_X_validating_raw[i,:])
        maximum = np.max(D_X_validating_raw[i,:])
        std = np.std(D_X_validating_raw[i,:])
        if(mean>0):    
            D_X_validating_raw[i,:] = D_X_validating_raw[i,:]/std#maximum
     
 
    NN_architecture_vec = np.array([0,1,2])
    padding_str_vec = ["same"]
    Number_of_epochs = 100
    number_of_runs = 10

    for Batch_size in np.array([64]):    
        for run_i  in range(0,number_of_runs):
            for NN_architecture in NN_architecture_vec:
                
                if(NN_architecture == 0):
                    parameters_NN = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_NN." + str(NN_architecture) + "_Batch." + "{:03d}".format(Batch_size) + "_lr.0.0010_run." + "{:03d}".format(run_i)  
                    model = Sequential()
                    model.add(Dense(256,input_dim=24*24,activation="relu"))
                    model.add(Dense(128,activation="relu"))
                    model.add(Dropout(0.5))
                    model.add(Dense(4, activation="softmax"))      
    
                    model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
                    start_time = time.time()
                    filepath = "model_keras"+parameters_NN + "_epochs.{epoch:03d}.keras"
                    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')                                   
                    history = model.fit(D_X_training_raw, D_y_training_categorical, shuffle="True" , batch_size=Batch_size, epochs=Number_of_epochs, verbose=1, validation_data=(D_X_validating_raw, D_y_validating_categorical),callbacks=[checkpoint])
                    stop_time = time.time()     
                    print("NN duration fit: ", "{:2.2f}".format(stop_time - start_time),"[s]" )                    

                                 
                
                if(NN_architecture == 1):
                    parameters_NN = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_NN." + str(NN_architecture) + "_Batch." + "{:03d}".format(Batch_size) + "_lr.0.0010_run." + "{:03d}".format(run_i)  
                    model = Sequential()
                    model.add(Conv2D(64, kernel_size = (3, 3), strides = (1,1), activation='relu', input_shape=(Nx,Ny,1),padding='same'))
                    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
                    model.add(Flatten())                                          
                    model.add(Dropout(0.5))
                    model.add(Dense(4, activation="softmax")) 

                    with open("model_keras_summary" + parameters_NN + "_report.txt","w") as fh:
                        model.summary(print_fn=lambda x: fh.write(x + '\n'))       
                    
                    
                    model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
                    start_time = time.time()
                    filepath = "model_keras"+parameters_NN + "_epochs.{epoch:03d}.keras"
                    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')                                   
                    history = model.fit(D_X_training_raw.reshape(D_X_training_raw.shape[0],Nx,Ny,1), D_y_training_categorical, shuffle="True" , batch_size=Batch_size, epochs=Number_of_epochs, verbose=1, validation_data=(D_X_validating_raw.reshape(D_X_validating_raw.shape[0],Nx,Ny,1), D_y_validating_categorical),callbacks=[checkpoint])
                    stop_time = time.time()     
                    print("NN duration fit: ", "{:2.2f}".format(stop_time - start_time),"[s]" )                    
                        
            
                if(NN_architecture == 2):
                    parameters_NN = string_V_max + selection_string + "_std_N_epochs." + str(Number_of_epochs) +  "_NN." + str(NN_architecture) + "_Batch." + "{:03d}".format(Batch_size) + "_lr.0.0010_run." + "{:03d}".format(run_i)  
                    model = Sequential()
                    model.add(Conv2D(64, kernel_size = (3, 3), strides = (1,1), activation='relu', input_shape=(Nx,Ny,1),padding='same'))
                    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
                    model.add(Conv2D(32, kernel_size = (3, 3), strides = (1,1), activation='relu', padding='same'))
                    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
                    model.add(Flatten())                                          
                    model.add(Dropout(0.5))
                    model.add(Dense(4, activation="softmax"))  
                
        
                    with open("model_keras_summary" + parameters_NN + "_report.txt","w") as fh:
                        model.summary(print_fn=lambda x: fh.write(x + '\n'))       
                    
                    
                    model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
                    start_time = time.time()
                    filepath = "model_keras"+parameters_NN + "_epochs.{epoch:03d}.keras"
                    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')                                   
                    history = model.fit(D_X_training_raw.reshape(D_X_training_raw.shape[0],Nx,Ny,1), D_y_training_categorical, shuffle="True" , batch_size=Batch_size, epochs=Number_of_epochs, verbose=1, validation_data=(D_X_validating_raw.reshape(D_X_validating_raw.shape[0],Nx,Ny,1), D_y_validating_categorical),callbacks=[checkpoint])
                    stop_time = time.time()     
                    print("CNN duration fit: ", "{:2.2f}".format(stop_time - start_time),"[s]" )

        
    
                np.savetxt("accuracy_keras_vs_epochs_training_D" + parameters_NN + ".callback_history",np.column_stack((np.arange(1,Number_of_epochs+1),np.array(history.history['accuracy']))))
                np.savetxt("accuracy_keras_vs_epochs_validating_D" + parameters_NN + ".callback_history",np.column_stack((np.arange(1,Number_of_epochs+1),np.array(history.history['val_accuracy']))))
            
                np.savetxt("loss_keras_vs_epochs_training_D" + parameters_NN + ".callback_history",np.column_stack((np.arange(1,Number_of_epochs+1),np.array(history.history['loss']))))
                np.savetxt("loss_keras_vs_epochs_validating_D" + parameters_NN + ".callback_history",np.column_stack((np.arange(1,Number_of_epochs+1),np.array(history.history['val_loss']))))
