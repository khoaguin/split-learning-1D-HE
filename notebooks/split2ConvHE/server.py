import os
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List, Union, Tuple

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


def recvall(sock, n) -> Union[None, bytes]:
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


class EcgServer:
    def __init__(self, init_weight_path: Union[str, Path]):
        """The constructor

        Args:
            init_weight (str): the string path to the initial weight
        """        
        checkpoint = torch.load(init_weight_path)
        self.params = dict(
            W3 = checkpoint["linear3.weight"],  # [128, 512]
            b3 = checkpoint["linear3.bias"],  # [128]
            W4 = checkpoint["linear4.weight"],  # [5, 128]
            b4 = checkpoint["linear4.bias"]  # [5]
        )
        self.grads = dict(
            dJdW3 = None,
            dJdb3 = None,
            dJdW4 = None,
            dJdb4 = None
        )
        self.cache = dict()

    def enc_linear(self, 
                    enc_x: CKKSTensor, 
                    W: Union[Tensor, CKKSTensor], 
                    b: Tensor):
        """
        The linear layer on homomorphic encrypted data
        Based on https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        """
        if type(W) == CKKSTensor:
            Wt = W.transpose()
        else:
            Wt = W.T
        y = enc_x.mm(Wt) + b
        dydW = enc_x
        dydx = W
        return y, dydW, dydx

    @staticmethod
    def approx_leaky_relu(enc_x: CKKSTensor) -> Tuple[CKKSTensor, CKKSTensor]:
        # 2.30556314780491e-19*x**5 - 0.000250098672095587*x**4 - 
        # 2.83384427035571e-17*x**3 + 0.0654264479654812*x**2 + 
        # 0.505000000000001*x + 0.854102848838318
        c1, c2, c3 = 0.854, 0.505, 0.0654
        y = enc_x.polyval([c1, c2, c3])
        dydx = c2 + 2*c3*enc_x
        return y, dydx

    def forward(self, he_a: CKKSTensor) -> CKKSTensor:
        he_a0, da0dW3, da0da = self.enc_linear(he_a, 
                                        self.params["W3"], 
                                        self.params["b3"])
        he_a1, da1da0 = EcgServer.approx_leaky_relu(he_a0)
        he_a2, da2dW4, da2da1  = self.enc_linear(he_a1, 
                                                 self.params["W4"], 
                                                 self.params["b4"]) 
        
        self.cache["da2dW4"] = da2dW4
        self.cache["da2da1"] = da2da1
        self.cache["da1da0"] = da1da0
        self.cache["da0dW3"] = da0dW3
        self.cache["da0da"] = da0da

        return he_a2

    def backward(self, dJda2) -> Tuple[CKKSTensor, CKKSTensor]:
        """Calculates the gradients of the loss function J w.r.t 
            the weights and biases

        Args:
            dJda2 (Tensor): the gradient of the loss function w.r.t
                            the output of the last linear layer
        Returns:
            dJda (CKKSTensor): the gradient of the loss function w.r.t
                            the encrypted activation map received from the client
            dJda0 (CKKSTensor): sends this to the client so he can decrypt
                            and calculate 
        """
        self.grads["dJdb4"] = dJda2.sum(0)  # sum accross all samples in a batch
        assert self.grads["dJdb4"].shape == self.params["b4"].shape, \
            "the grad of the loss function w.r.t b4 and b4 must have the same shape"
        
        da2dW4_T: CKKSTensor = self.cache["da2dW4"].transpose()
        self.grads["dJdW4"] = (da2dW4_T.mm(dJda2)).transpose()
        assert self.grads["dJdW4"].shape == list(self.params["W4"].shape), \
            "the grad of the loss function w.r.t W4 and W4 must have the same shape"

        dJda1: Tensor = torch.matmul(dJda2, self.cache["da2da1"])
        dJda0: CKKSTensor = dJda1 * self.cache["da1da0"]  # element-wise multiplication
        self.grads["dJdb3"] = dJda0.sum(0)  # sum accross all samples in a batch
        assert self.grads["dJdb3"].shape == list(self.params["b3"].shape), \
            "the grad of the loss function w.r.t b3 and b3 must have the same shape"

        # this causes out of memory error due to mul of 2 encrypted matrices
        # work around: send dJda0 to the client so he computes dJdW3 for the server
        # self.grads["dJdW3"] = (dJda0.transpose()).mm(self.cache["da0dW3"])

        dJda: CKKSTensor = dJda0.mm(self.cache["da0da"])
        
        # send dJda0 to the client so he can find dJdW3 for the server
        return dJda, dJda0

    def update_params(self, lr: float):
        """
        Update the parameters based on the gradients calculated in backward()
        """
        self.params["W3"] = self.params["W3"] - lr*self.grads["dJdW3"]
        self.params["b3"] = self.params["b3"] - lr*self.grads["dJdb3"]
        self.params["W4"] = self.params["W4"] - lr*self.grads["dJdW4"]
        self.params["b4"] = self.params["b4"] - lr*self.grads["dJdb4"]


    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad] = None
        self.cache = dict()


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

    def train(self, epoch: int, total_batch: int, lr: float):
        train_losses = list()
        train_accs = list()
        test_losses = list()
        test_accs = list()
        best_test_acc = 0  # best test accuracy

        for e in range(epoch):
            print(f"---- Epoch {e+1} ----")
            
            train_loss = 0.0
            correct, total = 0, 0
            for i in range(total_batch):
                print("Forward pass ---")
                self.ecg_model.clear_grad_and_cache()
                # receive the encrypted activation maps from the client
                he_a_bytes: bytes = self.recv_msg()
                print("\U0001F601 Received he_a from the client")
                he_a: CKKSTensor = CKKSTensor.load(context=self.client_ctx,
                                                   data=he_a_bytes)
                # the server puts the encrypted activations 
                # through 2 linear layers and send the outputs to the client
                he_a2: CKKSTensor = self.ecg_model.forward(he_a)
                print("\U0001F601 Sending he_a2 to the client")
                self.send_msg(msg=he_a2.serialize())
                
                print("--- Backward pass --- ")
                # get the the gradients of the loss w.r.t a2 from the client
                dJda2_bytes: bytes = self.recv_msg()
                print("\U0001F601 Received dJda2 from the client")
                dJda2: Tensor = pickle.loads(dJda2_bytes)
                # calculate the gradients of the loss w.r.t the weights and biases
                dJda, dJda0 = self.ecg_model.backward(dJda2)
                # sending the gradients back to the client
                print("\U0001F601 Sending dJda and dJda0 to the client")
                grads_bytes = {
                    "dJda": dJda.serialize(),
                    "dJda0": dJda0.serialize()
                }
                self.send_msg(msg=pickle.dumps(grads_bytes))
                print("\U0001F601 Received dJdW3 from the client")
                self.ecg_model.grads["dJdW3"] = pickle.loads(self.recv_msg())
                assert self.ecg_model.grads["dJdW3"].shape == self.ecg_model.params["W3"].shape, \
                    "dJdW3 and W3 do not have the same shape"
                # update the parameters based on the gradients
                self.ecg_model.update_params(lr=lr)

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
    server.train(epoch=hyperparams['epoch'], 
                 total_batch=hyperparams['total_batch'],
                 lr=hyperparams['lr'])


if __name__ == '__main__':
    # some global variables
    host = 'localhost'
    port = 10080
    max_recv = 4096

    main()
    