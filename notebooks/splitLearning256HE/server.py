import os
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List, Union, Tuple

from sockets import send_msg, recv_msg

import h5py
import numpy as np
import tenseal as ts
import torch
import torch.nn as nn
from torch import Tensor
from icecream import ic
ic.configureOutput(includeContext=True)
from tenseal.enc_context import Context
from tenseal.tensors.ckkstensor import CKKSTensor
from tenseal.tensors.ckksvector import CKKSVector
from tenseal.tensors.plaintensor import PlainTensor
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset
print(f'tenseal version: {ts.__version__}')
print(f'torch version: {torch.__version__}')


class Server:
    def __init__(self) -> None:
        # paths to files and directories
        self.socket = None
        self.train_loader = None
        self.test_loader = None
        self.context = None
        self.ecg_model = None
        self.device = None

    def init_socket(self, host, port):
        """[summary]

        Args:
            host ([str]): [description]
            port ([int]): [description]
        """
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))  # associates the socket with its local address
        self.socket.listen()
        print('Listening on', (host, port))
        self.socket, addr = self.socket.accept()  # wait for the client to connect
        print(f'Connection: {self.socket} \nAddress: {addr}')


def main(hyperparams):
    server = Server()
    server.init_socket(host='localhost', port=10080)
    send_msg(sock=server.socket, msg=pickle.dumps(hyperparams))
    

if __name__ == "__main__":
    hyperparams = {
        'dry_run': False,
        'batch_size': 4,
        'total_batch': 3312, # 13245/batch_size
        'epoch': 10,
        'lr': 0.001,
        'seed': 0
    }
    main(hyperparams)