import os
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List

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


# some global variables
host = 'localhost'
port = 10080
max_recv = 4096


class Server:
    """
    The class that represents the server in the protocol
    """
    def __init__(self) -> None:
        pass

    def init_socket(self, host, port):
        """[summary]

        Args:
            host ([str]): [description]
            port ([int]): [description]
        """
        server_socket = socket.socket()
        server_socket.bind((host, port))  # associates the socket with its local address
        server_socket.listen()
        print('Listening on', (host, port))
        conn, addr = server_socket.accept()  # wait for the client to connect
        print(f'Connection: {conn} \nAddress: {addr}')


if __name__ == '__main__':
    server = Server()
    server.init_socket(host, port)