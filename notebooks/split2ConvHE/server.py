import os
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
import tenseal as ts
import torch
from torch.cuda import _sleep
import torch.nn as nn
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


def recvall(sock, n) -> Union[None, bytes]:
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


class EcgServer(nn.Module):
    def __init__(self, init_weight_path: Union[str, Path]):
        """Constructor

        Args:
            init_weight (str): the string path to the initial weight
        """
        super(EcgServer, self).__init__()
        checkpoint = torch.load(init_weight_path)
        self.linear3_weight: torch.Tensor = checkpoint["linear3.weight"]  # [128, 512]
        self.linear3_bias: torch.Tensor = checkpoint["linear3.bias"]  # [128]
        self.linear4_weight: torch.Tensor = checkpoint["linear4.weight"]  # [5, 128]
        self.linear4_bias: torch.Tensor = checkpoint["linear4.bias"]  # [5]

    def enc_linear(self, 
                    enc_x: CKKSTensor, 
                    W: torch.Tensor, 
                    b: torch.Tensor) -> CKKSTensor:
        """
        The linear layer on homomorphic encrypted data
        Based on https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        """
        Wt = torch.transpose(W, 0, 1)
        return enc_x.mm(Wt) + b

    @staticmethod
    def approx_leaky_relu(enc_x: CKKSTensor):
        # 2.30556314780491e-19*x**5 - 0.000250098672095587*x**4 - 
        # 2.83384427035571e-17*x**3 + 0.0654264479654812*x**2 + 
        # 0.505000000000001*x + 0.854102848838318
        return enc_x.polyval([0.854, 0.505, 0.0654])

    def forward(self, enc_x: CKKSTensor) -> CKKSTensor:
        x = self.enc_linear(enc_x, self.linear3_weight, self.linear3_bias)  # [batch_size, 128]
        x = EcgServer.approx_leaky_relu(x)
        x = self.enc_linear(x, self.linear4_weight, self.linear4_bias)  # [batch_size, 5]
        return x

    def backward(self):
        """ 
        Calculates the gradients
        """
        raise NotImplementedError

    def update_params(self):
        """
        Update the parameters based on the gradients calculated in backward()
        """
        raise NotImplementedError


class Server:
    """
    The class that represents the server in the protocol
    """
    def __init__(self) -> None:
        self.socket = None
        self.device = None
        self.ecg_model = None
        self.client_ctx = None

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

    def recv_msg(self):
        # read message length and unpack it into an integer
        raw_msglen = recvall(self.socket, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # read the message data
        data: bytes = recvall(self.socket, msglen)
        return data

    def recv_ctx(self):
        client_ctx_bytes: bytes = self.recv_msg()
        self.client_ctx: Context = Context.load(client_ctx_bytes)

    def send_msg(self, msg) -> None:
        # prefix each message with a 4-byte length in network byte order
        msg = struct.pack('>I', len(msg)) + msg
        self.socket.sendall(msg)

    def build_model(self, init_weight_path: Union[str, Path]) -> None:
        """Build the neural network model for the server
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'server device: {torch.cuda.get_device_name(0)}')
        self.ecg_model = EcgServer(init_weight_path)
        # self.ecg_model.to(self.device)

    def set_random_seed(self, seed: int) -> None:
        """Setting the random seed for the training process

        Args:
            seed (int): [description]
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def train(self, epoch: int, total_batch: int):
        train_losses = list()
        train_accs = list()
        test_losses = list()
        test_accs = list()
        best_test_acc = 0  # best test accuracy

        for e in range(epoch):
            print(f"Epoch {e+1} -----")
            
            train_loss = 0.0
            correct, total = 0, 0
            for i in range(total_batch):
                # --- Forward Pass ---
                # receive the encrypted activation maps from the client
                he_a_bytes: bytes = self.recv_msg()
                print("\U0001F601 Received he_a from the client")
                he_a: CKKSTensor = CKKSTensor.load(context=self.client_ctx,
                                                   data=he_a_bytes)
                # the server puts the encrypted activations 
                # through 2 linear layers and send the outputs to the client
                he_a2: CKKSTensor = self.ecg_model(he_a)
                print(f"Server's outputs (he_a2): {he_a2}")
                print("\U0001F602 Sending he_a2 to the client")
                self.send_msg(msg=he_a2.serialize())
                
                # --- Backward pass ---
                debug = 1


def main():
    server = Server()
    server.init_socket(host, port)
    # receiving the hyperparameters from the clients
    hyperparams = server.recv_msg()
    print("Receiving hyperparams from the client")
    hyperparams = pickle.loads(hyperparams)
    ic(hyperparams)
    server.build_model('init_weight.pth')
    server.set_random_seed(seed=hyperparams['seed'])
    # before training, receive the TenSeal context (without the secret key) from the client
    server.recv_ctx()
    print("Received the tenseal context from the client")
    server.train(epoch=hyperparams['epoch'], total_batch=hyperparams['total_batch'])


if __name__ == '__main__':
    # some global variables
    host = 'localhost'
    port = 10080
    max_recv = 4096

    main()
    