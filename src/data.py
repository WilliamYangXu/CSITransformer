#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# for loading data and pre-processing data
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

img_height = 16
img_width = 32
img_channels = 2

class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __len__(self):
        return self.matdata.shape[0]

    def __getitem__(self, index):
        return self.matdata[index] 

def load_data(
        file_path,
        shuffle = False,
        train_test_ratio=0.8,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        drop_last=True):

    print("loading data...")
    mat = h5py.File(file_path + '/Hdata.mat', 'r')
    data = np.transpose(mat['H_train'])
    data = data.astype('float32')
    data = np.reshape(data, [len(data), img_channels, img_height, img_width])

    if shuffle:
        data_copy = np.copy(data)
        data_transpose = data_copy.transpose()
        np.random.shuffle(data_transpose)
        data_shuffle = data_transpose.transpose()

    partition = int(data.shape[0] * train_test_ratio)
    x_train, x_test = data[:partition], data[partition:]
    x_train_shuffle, x_test_shuffle = data_shuffle[:partition], data_shuffle[partition:]

    # dataLoader for training
    train_dataset = DatasetFolder(x_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory, drop_last=drop_last)
    # dataLoader for validating
    test_dataset = DatasetFolder(x_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    if shuffle:
        train_shuffle_dataset = DatasetFolder(x_train_shuffle)
        train_shuffle_loader = torch.utils.data.DataLoader(train_shuffle_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        test_shuffle_dataset = DatasetFolder(x_test_shuffle)
        test_shuffle_loader = torch.utils.data.DataLoader(test_shuffle_dataset, batch_size=batch_size,
                                                          shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


        return train_loader, test_loader, train_dataset, test_dataset,                train_shuffle_loader, test_shuffle_loader, train_shuffle_dataset, test_shuffle_dataset

    return train_loader, test_loader, train_dataset, test_dataset

