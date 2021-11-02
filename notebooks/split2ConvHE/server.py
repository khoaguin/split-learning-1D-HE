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


class ECGServer512:
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
            dJdW3 = torch.zeros(self.params["W3"].shape),
            dJdb3 = torch.zeros(self.params["b3"].shape),
            dJdW4 = torch.zeros(self.params["W4"].shape),
            dJdb4 = torch.zeros(self.params["b4"].shape)
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
        he_a0, enc_a, W3 = self.enc_linear(he_a, 
                                           self.params["W3"], 
                                           self.params["b3"])
        he_a1, da1da0 = EcgServer.approx_leaky_relu(he_a0)
        he_a2, enc_a1, W4  = self.enc_linear(he_a1, 
                                             self.params["W4"], 
                                             self.params["b4"]) 
        
        self.cache["da2dW4"] = enc_a1
        self.cache["da2da1"] = W4
        self.cache["da1da0"] = da1da0
        self.cache["da0dW3"] = enc_a
        self.cache["da0da"] = W3

        return he_a2

    def backward(self, dJda2) \
        -> Tuple[CKKSTensor, CKKSTensor, CKKSTensor, CKKSTensor]:
        """Calculates the gradients of the loss function J w.r.t 
            the weights and biases

        Args:
            dJda2 (Tensor): the gradient of the loss function w.r.t
                            the output of the last linear layer
        Returns:
            dJda (CKKSTensor): the gradient of the loss function w.r.t
                            the encrypted activation map received from the client
            dJda0 (CKKSTensor): sends this to the client so he can decrypt
                            and calculate dJdW3 and dJdb3 for the server
            dJdW4 (CKKSTensor): send this to the client so he decrypts it
                                and then send it back to the server
        """
        self.grads["dJdb4"] = dJda2.sum(0)  # sum accross all samples in a batch
        assert self.grads["dJdb4"].shape == self.params["b4"].shape, \
            "the grad of the loss function w.r.t b4 and b4 must have the same shape"
        
        da2dW4_T: CKKSTensor = self.cache["da2dW4"].transpose()
        dJdW4: CKKSTensor = (da2dW4_T.mm(dJda2)).transpose()
        assert dJdW4.shape == list(self.params["W4"].shape), \
            "the grad of the loss function w.r.t W4 and W4 must have the same shape"

        dJda1: Tensor = torch.matmul(dJda2, self.cache["da2da1"])
        dJda0: CKKSTensor = dJda1 * self.cache["da1da0"]  # element-wise multiplication
        # dJdb3: CKKSTensor = dJda0.sum(0)  # sum accross all samples in a batch
        # assert dJdb3.shape == list(self.params["b3"].shape), \
        #     "the grad of the loss function w.r.t b3 and b3 must have the same shape"

        # this causes out of memory error due to mul of 2 encrypted matrices
        # work around: send dJda0 to the client so he computes dJdW3 for the server
        # self.grads["dJdW3"] = (dJda0.transpose()).mm(self.cache["da0dW3"])

        dJda: CKKSTensor = dJda0.mm(self.cache["da0da"])
        
        return dJda, dJda0, dJdW4

    def check_update_grads(self, grads_plaintext) -> None:
        """Check and update the gradients in plaintext received from the client

        Args:
            grads_plaintext ([dict]): the dictionary that contains the 
                                      gradients needed in plaintext
        """
        assert grads_plaintext["dJdW3"].shape == self.grads["dJdW3"].shape, \
            f"dJdW3 received from the client is in wrong shape"
        assert grads_plaintext["dJdb3"].shape == self.grads["dJdb3"].shape, \
            f"dJdb3 received from the client is in wrong shape"
        assert grads_plaintext["dJdW4"].shape == self.grads["dJdW4"].shape, \
            f"dJdW4 received from the client is in wrong shape"

        self.grads["dJdW3"] = grads_plaintext["dJdW3"]
        self.grads["dJdb3"] = grads_plaintext["dJdb3"]
        self.grads["dJdW4"] = grads_plaintext["dJdW4"]

    def update_params(self, lr: float):
        """
        Update the parameters based on the gradients calculated in backward()
        """
        self.params["W3"] = self.params["W3"] - lr*self.grads["dJdW3"]
        self.params["b3"] = self.params["b3"] - lr*self.grads["dJdb3"]
        self.params["W4"] = self.params["W4"] - lr*self.grads["dJdW4"]
        self.params["b4"] = self.params["b4"] - lr*self.grads["dJdb4"]

    def clear_grad_and_cache(self):
        """Clear the cache dictionary and make all grads zeros for the 
           next forward pass on a new batch
        """
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


class ECGServer256:
    def __init__(self, init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.params = dict(
            W = checkpoint["linear.weight"],  # [5, 256],
            b = checkpoint["linear.bias"]  # [5]
        )
        self.grads = dict(
            dJdW = torch.zeros(self.params["W"].shape),
            dJdb = torch.zeros(self.params["b"].shape),
        )
        self.cache = dict()

    def enc_linear(self, 
                   enc_x: CKKSTensor, 
                   W: Tensor, 
                   b: Tensor):
        """
        The linear layer on homomorphic encrypted data
        Based on https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        """
        Wt = W.T
        y = enc_x.mm(Wt) + b
        dydW = enc_x
        dydx = W
        return y, dydW, dydx

    def forward(self, he_a: CKKSTensor) -> CKKSTensor:
        # a2 = a*W' + b 
        he_a2, _, W = self.enc_linear(he_a, 
                                      self.params["W"],
                                      self.params["b"])
        self.cache["da2da"] = W
        return he_a2

    def backward(self, dJda2: Tensor, context: Context) -> CKKSTensor:
        self.grads["dJdb"] = dJda2.sum(0)  # sum accross all samples in a batch
        assert self.grads["dJdb"].shape == self.params["b"].shape, \
            "the grad of the loss function w.r.t b and b must have the same shape"

        dJda: Tensor = torch.matmul(dJda2, self.cache["da2da"])
        dJda: CKKSTensor = ts.ckks_tensor(context, dJda.tolist())

        return dJda

    def clear_grad_and_cache(self):
        """Clear the cache dictionary and make all grads zeros for the 
           next forward pass on a new batch
        """
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

    def check_update_grads(self, dJdW) -> None:
        """Check and update the gradients in plaintext received from the client

        Args:
            grads_plaintext ([dict]): the dictionary that contains the 
                                      gradients needed in plaintext
        """
        assert dJdW.shape == self.grads["dJdW"].shape, \
            f"dJdW received from the client is in wrong shape"

        self.grads["dJdW"] = dJdW

    def update_params(self, lr: float):
        """
        Update the parameters based on the gradients calculated in backward()
        """
        self.params["W"] = self.params["W"] - lr*self.grads["dJdW"]
        self.params["b"] = self.params["b"] - lr*self.grads["dJdb"]


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

    def build_model(self, 
                    init_weight_path: Union[str, Path]) -> None:
        """Build the neural network model for the server
        """
        self.ecg_model = ECGServer256(init_weight_path)

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

    def train(self, 
              epoch: int, 
              total_batch: int, 
              lr: float,
              dry_run: bool):
        """Training used for ECG model with 2 linear layers

        Args:
            epoch (int): [description]
            total_batch (int): [description]
            lr (float): [description]
            dry_run (bool): [description]
        """
        for e in range(epoch):
            print(f"---- Epoch {e+1} ----")
            start = time.time()
            for i in range(total_batch):
                if dry_run: print("\U0001F601 Received he_a from the client")
                print("Forward pass ---")
                self.ecg_model.clear_grad_and_cache()
                # receive the encrypted activation maps from the client
                he_a_bytes: bytes = self.recv_msg()
                he_a: CKKSTensor = CKKSTensor.load(context=self.client_ctx,
                                                   data=he_a_bytes)
                # the server puts the encrypted activations 
                # through 2 linear layers and send the outputs to the client
                he_a2: CKKSTensor = self.ecg_model.forward(he_a)
                if dry_run: print("\U0001F601 Sending he_a2 to the client")
                self.send_msg(msg=he_a2.serialize())
                
                dJda2_bytes: bytes = self.recv_msg()
                if dry_run: print("\U0001F601 Received dJda2 from the client")
                print("Backward pass --- ")                
                dJda2: Tensor = pickle.loads(dJda2_bytes)
                # calculate the gradients of the loss w.r.t the weights and biases
                dJda, dJda0, dJdW4 = self.ecg_model.backward(dJda2)
                # sending the gradients back to the client
                grads_bytes = {
                    "dJda": dJda.serialize(),
                    "dJda0": dJda0.serialize(),
                    "dJdW4": dJdW4.serialize()
                }
                if dry_run: print("\U0001F601 Sending dJda, dJda0, dJdW4 to the client")
                self.send_msg(msg=pickle.dumps(grads_bytes))
                
                grads_plaintext = self.recv_msg()
                grads_plaintext = pickle.loads(grads_plaintext)
                if dry_run: print("\U0001F601 Received dJdW3, dJdb3, dJdW4 from the client")
                self.ecg_model.check_update_grads(grads_plaintext)
                self.ecg_model.update_params(lr=lr)
            
            end = time.time()
            print(f'time taken for one epoch: {end-start:.2f}s')
    
    def train2(self, 
               epoch: int, 
               total_batch: int, 
               lr: float,
               dry_run: bool):
        """Training used for ECG model with 2 linear layers

        Args:
            epoch (int): [description]
            total_batch (int): [description]
            lr (float): [description]
            dry_run (bool): [description]
        """
        for e in range(epoch):
            print(f"---- Epoch {e+1} ----")
            start = time.time()
            for i in range(total_batch):
                if dry_run: print("Forward pass ---")
                self.ecg_model.clear_grad_and_cache()
                # receive the encrypted activation maps from the client
                he_a_bytes: bytes = self.recv_msg()
                if dry_run: print("\U0001F601 Received he_a from the client")
                he_a: CKKSTensor = CKKSTensor.load(context=self.client_ctx,
                                                   data=he_a_bytes)
                # the server puts the encrypted activations through
                # the linear layer and send the outputs to the client
                he_a2: CKKSTensor = self.ecg_model.forward(he_a)
                if dry_run: print("\U0001F601 Sending he_a2 to the client")
                self.send_msg(msg=he_a2.serialize())
                
                if dry_run: print("Backward pass --- ")
                grads_bytes: bytes = self.recv_msg()
                if dry_run: print("\U0001F601 Received dJda2, dJdW from the client")
                grads = pickle.loads(grads_bytes)
                self.ecg_model.check_update_grads(grads["dJdW"])
                dJda: CKKSTensor = self.ecg_model.backward(grads["dJda2"], 
                                                           self.client_ctx)
                # sending the dJda to the client
                if dry_run: print("\U0001F601 Sending dJda to the client")
                self.send_msg(msg=dJda.serialize())
                self.ecg_model.update_params(lr=lr) # updating the parameters

            end = time.time()
            print(f'time taken for one epoch: {end-start:.2f}s')


def main():
    server = Server()
    server.init_socket(host, port)
    # receiving the hyperparameters from the clients
    hyperparams = server.recv_msg()
    print("\U0001F601 Received the hyperparams from the client")
    hyperparams = pickle.loads(hyperparams)
    ic(hyperparams)
    server.build_model('weights/init_weight_256.pth')
    server.set_random_seed(seed=hyperparams['seed'])
    # before training, receive the TenSeal context (without the secret key) from the client
    server.recv_ctx()
    print("\U0001F601 Received the tenseal context from the client")
    server.train2(epoch=hyperparams['epoch'], 
                 total_batch=hyperparams['total_batch'],
                 lr=hyperparams['lr'],
                 dry_run=hyperparams['dry_run'])


if __name__ == '__main__':
    # some global variables
    host = 'localhost'
    port = 10080
    max_recv = 4096

    main()
    