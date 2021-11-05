import os
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List, Union, Tuple
import math

from sockets import send_msg, recv_msg

import h5py
import numpy as np
import tenseal as ts
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from icecream import ic

ic.configureOutput(includeContext=True)
from tenseal.enc_context import Context
from tenseal.tensors.ckkstensor import CKKSTensor
from tenseal.tensors.ckksvector import CKKSVector
from tenseal.tensors.plaintensor import PlainTensor
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset


class Client:
    """
    The class that represents the client in the protocol.    
    """    
    def __init__(self) -> None:
        # paths to files and directories
        self.socket = None
        self.train_loader = None
        self.test_loader = None
        self.context = None
        self.ecg_model = None
        self.device = None

    def init_socket(self, host, port) -> None:
        """Connect to the server's socket 

        Args:
            host ([str]): [description]
            port ([int]): [description]
        """
        self.socket = socket.socket()
        self.socket.connect((host, port))  # connect to a remote [server] address,
        print(self.socket)
    
    def load_ecg_dataset(self, 
                         train_name: str, 
                         test_name: str,
                         batch_size: int) -> None:
        """[summary]

        Args:
            train_name (str): [description]
            test_name (str): [description]
            batch_size (int): [description]
        """
        train_dataset = ECG(train_name, test_name, train=True)
        test_dataset = ECG(train_name, test_name, train=False)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)


class ECG(Dataset):
    """The class used by the client to load the dataset

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, train_name: str, test_name: str, train=True):
        if train:
            with h5py.File(train_name, 'r') as hdf:
                self.x = hdf['x_train'][:]
                self.y = hdf['y_train'][:]
        else:
            with h5py.File(test_name, 'r') as hdf:
                self.x = hdf['x_test'][:]
                self.y = hdf['y_test'][:]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), \
               torch.tensor(self.y[idx])


def main():
    client = Client()
    client.init_socket(host='localhost', port=10080)
    hyperparams = pickle.loads(recv_msg(sock=client.socket))
    client.load_ecg_dataset(train_name="data/train_ecg.hdf5",
                            test_name="data/test_ecg.hdf5",
                            batch_size=hyperparams["batch_size"])
    

if __name__ == "__main__":
    main()