import os
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List
import threading

import h5py
import numpy as np
import tenseal as ts
import torch
import torch.nn as nn
from icecream import ic
from tenseal.enc_context import Context
from tenseal.tensors.ckkstensor import CKKSTensor
from tenseal.tensors.ckksvector import CKKSVector
from tenseal.tensors.plaintensor import PlainTensor
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset


print(f'torch version: {torch.__version__}')
print(f'tenseal version: {ts.__version__}')

project_path = Path.cwd().parent.parent
print(f'project_path: {project_path}')

# some global variables
host = 'localhost'
port = 10080
max_recv = 4096

dry_run = True # break after 2 batches for 2 epoch, set batch size to be 2
if dry_run:
    batch_size = 2
    epoch = 2
else:
    batch_size = 32
    epoch = 400
total_batch = 13245/batch_size


class ECG(Dataset):
    """The class used by the client to load the dataset

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, data_dir: str, train_name: str, test_name: str, train=True):
        if train:
            with h5py.File(project_path/data_dir/train_name, 'r') as hdf:
                self.x = hdf['x_train'][:]
                self.y = hdf['y_train'][:]
        else:
            with h5py.File(project_path/data_dir/test_name, 'r') as hdf:
                self.x = hdf['x_test'][:]
                self.y = hdf['y_test'][:]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])


class Client: 
    """
    The class that represents the client in the protocol.    
    """    
    def __init__(self) -> None:
        # paths to files and directories
        self.data_dir = 'data'  # used to be 'mitdb'
        self.train_name = 'train_ecg.hdf5'
        self.test_name = 'test_ecg.hdf5'

    def load_ecg_dataset(self) -> None:
        train_dataset = ECG(self.data_dir, self.train_name, self.test_name, train=True)
        test_dataset = ECG(self.data_dir, self.train_name, self.test_name, train=False)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
    def init_socket(self, host, port):
        """[summary]

        Args:
            host ([str]): [description]
            port ([int]): [description]
        """
        client_socket = socket.socket()
        client_socket.connect((host, port))  # connect to a remote [server] address,

    def send_msg(sock, msg):
        # prefix each message with a 4-byte length in network byte order
        msg = struct.pack('>I', len(msg)) + msg
        sock.sendall(msg)


if __name__ == '__main__':
    client = Client()
    client.init_socket(host, port)
    











