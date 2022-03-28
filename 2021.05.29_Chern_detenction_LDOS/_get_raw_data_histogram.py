#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:01:46 2021
@author: marcin
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def train_validate_test_split_data_raw(data_raw, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(data_raw.shape[0])
    m = data_raw.shape[0]
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = data_raw[perm[:train_end],:]
    validate = data_raw[perm[train_end:validate_end],:]
    test = data_raw[perm[validate_end:],:]
    return train, validate, test





Nx = 24
Ny = 24


index_Nx = 0;
index_N = 1;
index_seed = 2;
index_energy_window = 3;
index_mu = 4;
index_alpha = 5;
index_gap_clean = 6;
index_v_sample = 7;
index_C_bulk = 8;
index_C_marker_clean = 9;
index_C_marker_disorder = 10;
index_number_of_states = 11;
index_LDOS_start = 12;


filename = 'training_data_unique_seed_LDOS_normalized_to_1_raw_D_flat.0_Nx.24_Delta.1.00_balanced_set.data_raw'

filename = 'D_train_V_max.1_preselection.1.dat'

data_raw = np.loadtxt(filename)  



#%%
data_dictionary = []
for i in range(0,data_raw.shape[0]):
    data_row = data_raw[i,:]
    dictionary = {'Nx': data_row[index_Nx],
                  'N': data_row[index_N],
                  'seed': data_row[index_seed],
                  'energy_window': data_row[index_energy_window],
                  'mu': data_row[index_mu],
                  'alpha': data_row[index_alpha],
                  'gap_clean': data_row[index_gap_clean],
                  'v_sample': data_row[index_v_sample],
                  'Chern_bulk': np.abs(data_row[index_C_bulk]),
                  'Chern_marker_clean': np.abs(data_row[index_C_marker_clean]),
                  'Chern_marker_disorder': np.abs(data_row[index_C_marker_disorder]),
                  'number_of_states': data_row[index_number_of_states],
                  'LDOS': data_row[index_LDOS_start:]
                }
    data_dictionary.append(dictionary)
dataFrame = pd.DataFrame(data_dictionary)



#%% #extract test data : V <= 1
dataFrame_tmp = dataFrame

bins = 100
condition_0 = np.logical_and(np.round(dataFrame_tmp['Chern_bulk'])==0 , dataFrame_tmp['v_sample'] < 10)  
condition_1 = np.logical_and(np.round(np.round(dataFrame_tmp['Chern_bulk'])==1) , dataFrame_tmp['v_sample'] < 10)  
condition_2 = np.logical_and(np.round(np.round(dataFrame_tmp['Chern_bulk'])==2) , dataFrame_tmp['v_sample'] < 10)  
condition_3 = np.logical_and(np.round(np.round(dataFrame_tmp['Chern_bulk'])==3) , dataFrame_tmp['v_sample'] < 10)  
idx_C_bulk_0 = np.where(condition_0)[0]
idx_C_bulk_1 = np.where(condition_1)[0]
idx_C_bulk_2 = np.where(condition_2)[0]
idx_C_bulk_3 = np.where(condition_3)[0]

plt.hist(dataFrame_tmp['Chern_marker_disorder'][idx_C_bulk_0],bins=bins,alpha=0.4)
plt.hist(dataFrame_tmp['Chern_marker_disorder'][idx_C_bulk_1],bins=bins,alpha=0.6)
plt.hist(dataFrame_tmp['Chern_marker_disorder'][idx_C_bulk_2],bins=bins,alpha=0.8)
plt.hist(dataFrame_tmp['Chern_marker_disorder'][idx_C_bulk_3],bins=bins,alpha=0.2)
plt.ylim(0,200)
plt.xlabel("Chern marker |C|")
plt.ylabel("count")
 
 