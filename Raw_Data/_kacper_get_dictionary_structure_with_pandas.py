#!/usr/bin/env python3
"""
Created on Sun May  2 16:01:46 2021
@author: marcin
@editor: kcybinski

Edited on Mon 28.03.2022
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Changed train dataset size to 0.8, validation is 0.2 of this dataset
def train_validate_test_split_data(
    data_raw, train_percent=0.8, validate_percent=0.2
):
    train, test = train_test_split(data_raw, train_size=train_percent)
    train, validate = train_test_split(train, test_size=validate_percent)
    return train, validate, test


Nx = 24
Ny = 24


index_Nx = 0
index_N = 1
index_seed = 2
index_energy_window = 3
index_mu = 4
index_alpha = 5
index_gap_clean = 6
index_v_sample = 7
index_C_bulk = 8
index_C_marker_clean = 9
index_C_marker_disorder = 10
index_number_of_states = 11
index_LDOS_start = 12


filename = "training_data_unique_seed_LDOS_normalized_to_1_raw_D_flat.0_Nx.24_Delta.1.00_balanced_set.data_raw"
data_raw = np.loadtxt(filename)

idx_v_cutoff = np.where(data_raw[:, index_v_sample] < 1)[0]
data_raw_V_max_1 = data_raw[idx_v_cutoff, :]


idx_v_cutoff = np.where(data_raw[:, index_v_sample] < 2)[0]
data_raw_V_max_2 = data_raw[idx_v_cutoff, :]


#%%
(
    data_train_preselection_0,
    data_validate_preselection_0,
    data_test_preselection_0,
) = train_validate_test_split_data(
    data_raw_V_max_1, train_percent=0.8, validate_percent=0.1
)

np.savetxt("D_train_V_max.1_preselection.0.dat", data_train_preselection_0)
np.savetxt("D_validate_V_max.1_preselection.0.dat", data_validate_preselection_0)
np.savetxt("D_test_V_max.1_preselection.0.dat", data_test_preselection_0)

idx = np.where(
    np.abs(
        np.abs(data_raw_V_max_1[:, index_C_bulk])
        - np.abs(data_raw_V_max_1[:, index_C_marker_disorder])
    )
    < 0.5
)[0]
data_raw_V_max_1_preselected = data_raw_V_max_1[idx, :]


(
    data_train_preselection_1,
    data_validate_preselection_1,
    data_test_preselection_1,
) = train_validate_test_split_data(
    data_raw_V_max_1_preselected, train_percent=0.8, validate_percent=0.1
)

np.savetxt("D_train_V_max.1_preselection.1.dat", data_train_preselection_1)
np.savetxt("D_validate_V_max.1_preselection.1.dat", data_validate_preselection_1)
np.savetxt("D_test_V_max.1_preselection.1.dat", data_test_preselection_1)

#%%
(
    data_train_preselection_0,
    data_validate_preselection_0,
    data_test_preselection_0,
) = train_validate_test_split_data(
    data_raw_V_max_2, train_percent=0.8, validate_percent=0.1
)

np.savetxt("D_train_V_max.2_preselection.0.dat", data_train_preselection_0)
np.savetxt("D_validate_V_max.2_preselection.0.dat", data_validate_preselection_0)
np.savetxt("D_test_V_max.2_preselection.0.dat", data_test_preselection_0)

idx = np.where(
    np.abs(
        np.abs(data_raw_V_max_2[:, index_C_bulk])
        - np.abs(data_raw_V_max_2[:, index_C_marker_disorder])
    )
    < 0.5
)[0]
data_raw_V_max_2_preselected = data_raw_V_max_2[idx, :]


(
    data_train_preselection_2,
    data_validate_preselection_2,
    data_test_preselection_2,
) = train_validate_test_split_data(
    data_raw_V_max_2_preselected, train_percent=0.8, validate_percent=0.1
)

np.savetxt("D_train_V_max.2_preselection.1.dat", data_train_preselection_1)
np.savetxt("D_validate_V_max.2_preselection.1.dat", data_validate_preselection_1)
np.savetxt("D_test_V_max.2_preselection.1.dat", data_test_preselection_1)


#%%
data_dictionary = []
for i in range(0, data_raw.shape[0]):
    data_row = data_raw[i, :]
    dictionary = {
        "Nx": data_row[index_Nx],
        "N": data_row[index_N],
        "seed": data_row[index_seed],
        "energy_window": data_row[index_energy_window],
        "mu": data_row[index_mu],
        "alpha": data_row[index_alpha],
        "gap_clean": data_row[index_gap_clean],
        "v_sample": data_row[index_v_sample],
        "Chern_bulk": np.abs(data_row[index_C_bulk]),
        "Chern_marker_clean": np.abs(data_row[index_C_marker_clean]),
        "Chern_marker_disorder": np.abs(data_row[index_C_marker_disorder]),
        "number_of_states": data_row[index_number_of_states],
        "LDOS": data_row[index_LDOS_start:],
    }
    data_dictionary.append(dictionary)
dataFrame = pd.DataFrame(data_dictionary)



# dataFrame_train = dataFrame.sample(frac = 0.6)
# dataFrame_test = dataFrame.drop(dataFrame_train.index).sample(frac = 0.4)
# dataFrame_validate = dataFrame.drop(dataFrame_train.index).drop(dataFrame_test.index)


# dataFrame_train.reset_index(inplace=True)
# dataFrame_test.reset_index(inplace=True)
# dataFrame_validate.reset_index(inplace=True)

dataFrame_train, dataFrame_validate, dataFrame_test = train_validate_test_split_data(
    dataFrame, train_percent=0.6, validate_percent=0.2
)
dataFrame_train.reset_index(inplace=True)
dataFrame_test.reset_index(inplace=True)
dataFrame_validate.reset_index(inplace=True)

#%% #extract training data : V <= 1
dataFrame_tmp = dataFrame_train

bins = 100
condition_0 = np.logical_and(
    np.round(dataFrame_tmp["Chern_bulk"]) == 0, dataFrame_tmp["v_sample"] < 1
)
condition_1 = np.logical_and(
    np.round(dataFrame_tmp["Chern_bulk"]) == 1, dataFrame_tmp["v_sample"] < 1
)
condition_2 = np.logical_and(
    np.round(dataFrame_tmp["Chern_bulk"]) == 2, dataFrame_tmp["v_sample"] < 1
)
condition_3 = np.logical_and(
    np.round(dataFrame_tmp["Chern_bulk"]) == 3, dataFrame_tmp["v_sample"] < 1
)
idx_C_bulk_0 = np.where(condition_0)[0]
idx_C_bulk_1 = np.where(condition_1)[0]
idx_C_bulk_2 = np.where(condition_2)[0]
idx_C_bulk_3 = np.where(condition_3)[0]

plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_0], bins=bins, alpha=0.4)
plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_1], bins=bins, alpha=0.6)
plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_2], bins=bins, alpha=0.8)
plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_3], bins=bins, alpha=0.2)
plt.ylim(0, 2000)

#%%
C_0 = dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_0].to_numpy()
aaaaaaaaa = np.array([dataFrame_tmp["LDOS"][idx_C_bulk_0].to_numpy()])

#%%

C_1 = np.array(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_1])
LDOS_1 = dataFrame_tmp["LDOS"][idx_C_bulk_1]


C_2 = np.array(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_2])
LDOS_2 = dataFrame_tmp["LDOS"][idx_C_bulk_2]


C_3 = np.array(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_3])
LDOS_3 = dataFrame_tmp["LDOS"][idx_C_bulk_3]
#%%


#%%


#%% #extract validating data : V <= 1
dataFrame_tmp = dataFrame_validate

bins = 100
condition_0 = np.logical_and(
    np.round(dataFrame_tmp["Chern_bulk"]) == 0, dataFrame_tmp["v_sample"] < 1
)
condition_1 = np.logical_and(
    np.round(np.round(dataFrame_tmp["Chern_bulk"]) == 1), dataFrame_tmp["v_sample"] < 1
)
condition_2 = np.logical_and(
    np.round(np.round(dataFrame_tmp["Chern_bulk"]) == 2), dataFrame_tmp["v_sample"] < 1
)
condition_3 = np.logical_and(
    np.round(np.round(dataFrame_tmp["Chern_bulk"]) == 3), dataFrame_tmp["v_sample"] < 1
)
idx_C_bulk_0 = np.where(condition_0)[0]
idx_C_bulk_1 = np.where(condition_1)[0]
idx_C_bulk_2 = np.where(condition_2)[0]
idx_C_bulk_3 = np.where(condition_3)[0]

plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_0], bins=bins, alpha=0.4)
plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_1], bins=bins, alpha=0.6)
plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_2], bins=bins, alpha=0.8)
plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_3], bins=bins, alpha=0.2)
plt.ylim(0, 200)


#%% #extract test data : V <= 1
dataFrame_tmp = dataFrame_test

bins = 100
condition_0 = np.logical_and(
    np.round(dataFrame_tmp["Chern_bulk"]) == 0, dataFrame_tmp["v_sample"] < 10
)
condition_1 = np.logical_and(
    np.round(np.round(dataFrame_tmp["Chern_bulk"]) == 1), dataFrame_tmp["v_sample"] < 10
)
condition_2 = np.logical_and(
    np.round(np.round(dataFrame_tmp["Chern_bulk"]) == 2), dataFrame_tmp["v_sample"] < 10
)
condition_3 = np.logical_and(
    np.round(np.round(dataFrame_tmp["Chern_bulk"]) == 3), dataFrame_tmp["v_sample"] < 10
)
idx_C_bulk_0 = np.where(condition_0)[0]
idx_C_bulk_1 = np.where(condition_1)[0]
idx_C_bulk_2 = np.where(condition_2)[0]
idx_C_bulk_3 = np.where(condition_3)[0]

plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_0], bins=bins, alpha=0.4)
plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_1], bins=bins, alpha=0.6)
plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_2], bins=bins, alpha=0.8)
plt.hist(dataFrame_tmp["Chern_marker_disorder"][idx_C_bulk_3], bins=bins, alpha=0.2)
plt.ylim(0, 200)


#%%
dataFrame.to_pickle("pandas_data_set_Shiba_data.pkl")

#%%
output = pd.read_pickle("pandas_data_set_Shiba_data.pkl")
# compression_opts = dict(method='zip',  archive_name='out.csv')
# pd.to_csv('out.zip', index=False, compression=compression_opts)

#%%
plt.imshow(dataFrame["LDOS"][idx_C_bulk_1[0]].reshape(24, 24))
