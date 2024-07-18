"""
data loader
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def load_county_data(states: list, years: list, drop_features: list, year: int):
    # states: the FIPS of all states to be included

    filename_yield = "data/Xy_county_" + str(year) + ".csv"
    
    
    df_all = pd.read_csv(filename_yield)
    
    # Filter certain years:
    #print(df_all.shape)
    dist_threshold = np.mean(df_all['dist'])
    df_all = df_all[df_all['dist'] < dist_threshold]
    #print(df_all.shape)

    drop_list = ['FIPS', 'year', 'yield', 'STATE_FIPS', 'GCVI_count', 'dist']
    drop_list = drop_list + drop_features

    df_all = df_all[df_all['year'].isin(years)]
    Xy = df_all[df_all['STATE_FIPS'].isin(states)]

    X = Xy.drop(drop_list, axis=1)

    y = Xy['yield']
    y = y.to_numpy()
    y = np.expand_dims(y, axis=1)

    dist = Xy['dist'].to_numpy()
    dist = np.expand_dims(dist, axis=1)

    dist_copy = np.copy(dist)

    return X, y, dist_copy


def load_scym_data(state, year, drop_features = [], num_sample = 0, random_seed = 100):
    # state: the abbr of the state, e.g. 'IL'
    # num_sample: the number of samples to use from the dataframe

    filename_harmonics_scym = "data/Field_data_" + str(year) + ".csv"
    
    df_scym = pd.read_csv(filename_harmonics_scym)

    drop_list = ['pointID', 'state', 'year', 'GCVI_count', 'dist']
    drop_list = drop_list + drop_features

    if num_sample == 0:
        df_scym_sample = df_scym
    else:
        num_sample = min(num_sample, df_scym.shape[0])
        df_scym_sample = df_scym.sample(n = num_sample, random_state = random_seed)

    dist = df_scym_sample['dist'].to_numpy()
    dist = np.expand_dims(dist, axis=1)

    df_scym_sample = df_scym_sample.drop(drop_list, axis=1)
    
    df_scym_sample = df_scym_sample[df_scym_sample['yield_tha'] < 21.52]
    df_scym_sample = df_scym_sample[df_scym_sample['yield_tha'] > 0]

    X_scym = df_scym_sample.drop(['yield_scym', 'yield_tha'], axis=1)
    y_scym = df_scym_sample['yield_tha']
    y_county_yield = df_scym_sample['yield_tha']

    y_scym = y_scym.to_numpy()
    y_scym = np.expand_dims(y_scym, axis=1)

    y_county_yield = y_county_yield.to_numpy()
    y_county_yield = np.expand_dims(y_county_yield, axis=1)


    return X_scym, y_scym, y_county_yield, dist


def data_normalization(X_src, y_src, X_tar, y_tar, normalization = True, type = 'Standard'):

    if type == 'Standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    if normalization:
        X_src_n = scaler.fit_transform(X_src)
        X_tar_n = scaler.transform(X_tar)

    else:
        X_src_n = X_src.to_numpy()
        X_tar_n = X_tar.to_numpy()

    return X_src_n, y_src, X_tar_n, y_tar


def torch_dataloader(X, y, batch_size, shuffle = True):

    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    train_dataset_src = TensorDataset(X_train, y_train)
    train_loader_src = DataLoader(train_dataset_src, batch_size=batch_size, shuffle=shuffle)

    return train_loader_src

