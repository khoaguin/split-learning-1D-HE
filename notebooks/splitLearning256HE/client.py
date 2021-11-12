import os
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List, Union, Tuple
import math

from sockets import send_msg, recv_msg

import pandas as pd
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


class EcgClient256(nn.Module):
    """The client's 1D CNN model

    Args:
        nn ([torch.Module]): [description]
    """
    def __init__(self, 
                 context: Context, 
                 init_weight_path: Union[str, Path]):
        super(EcgClient256, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, 
                               out_channels=16, 
                               kernel_size=7, 
                               padding=3,
                               stride=1)  # 128 x 16
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = nn.Conv1d(in_channels=16, 
                               out_channels=8, 
                               kernel_size=5, 
                               padding=2)  # 64 x 8
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2)  # 32 x 8 = 256
        
        self.load_init_weights(init_weight_path)
        self.context = context

    def load_init_weights(self, init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.conv1.weight.data = checkpoint["conv1.weight"]
        self.conv1.bias.data = checkpoint["conv1.bias"]
        self.conv2.weight.data = checkpoint["conv2.weight"]
        self.conv2.bias.data = checkpoint["conv2.bias"]

    def forward(self, x: Tensor) -> Tuple[Tensor, CKKSTensor]:
        x = self.conv1(x)  
        x = self.relu1(x)
        x = self.pool1(x)  
        x = self.conv2(x) 
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 256)  # [batch_size, 256]
        enc_x: CKKSTensor = ts.ckks_tensor(self.context, x.tolist())
        return x, enc_x


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

    def make_tenseal_context(self, 
                             poly_modulus_degree: int,
                             coeff_mod_bit_sizes: List[int], 
                             global_scale: int) -> Context:
        """Generate the TenSeal context to encrypt the activation maps

        Args:
            poly_modulus_degree (int): [description]
            coeff_mod_bit_sizes (List[int]): [description]
            global_scale (int): [description]

        Returns:
            Context: [description]
        """
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS, 
            poly_modulus_degree=poly_modulus_degree, 
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.global_scale = global_scale
        # self.context.generate_galois_keys()
    
    def send_context(self) -> None:
        """Send the context to the server
        """
        send_msg(sock=self.socket,
                 msg=self.context.serialize(save_secret_key=False))

    def build_model(self, init_weight_path: Union[str, Path]) -> None:
        """Build the neural network model for the client

        Raises:
            TypeError: if the tenseal context needed to encrypt the activation 
                        map is None, then raise an error
        """
        if self.context == None:
            raise TypeError("tenseal context is None")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Client device: {torch.cuda.get_device_name(0)}')
        self.ecg_model = EcgClient256(context=self.context, 
                                      init_weight_path=init_weight_path)
        self.ecg_model.to(self.device)

    def train(self, hyperparams: dict) -> None:
        seed = hyperparams["seed"]
        verbose = hyperparams["verbose"]
        lr = hyperparams["lr"]
        total_batch = hyperparams["total_batch"]
        epoch = hyperparams["epoch"]
        # set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        train_losses = list()
        train_accs = list()
        # test_losses = list()
        # test_accs = list()
        # best_test_acc = 0  # best test accuracy
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(self.ecg_model.parameters(), lr=lr)
        for e in range(epoch):
            print(f"---- Epoch {e+1} ----")
            start = time.time()
            e_train_loss, e_correct, e_samples = self.training_loop(verbose, total_batch, 
                                                                    loss_func, optimizer)
            end = time.time()
            train_losses.append(e_train_loss / total_batch)
            train_accs.append(e_correct / e_samples)
            train_status = f"training loss: {train_losses[-1]:.4f}, "\
                           f"training acc: {train_accs[-1]*100:.2f}, "\
                           f"training time: {end-start:.2f}s"
            print(train_status)
            send_msg(sock=self.socket, msg=pickle.dumps(train_status))
                
        return train_losses, train_accs

    def training_loop(self, verbose, total_batch, loss_func, optimizer):
        epoch_train_loss = 0.0
        epoch_correct = 0 
        epoch_total_samples = 0
        for i, batch in enumerate(self.train_loader):
            optimizer.zero_grad()
            if verbose: print("Forward Pass ---")
            x, y = batch  # get the input data and ground-truth output in the batch
            x, y = x.to(self.device), y.to(self.device)  # put to cuda or cpu
            a, he_a = self.ecg_model(x)  # [batch_size, 256]
            if verbose: print("\U0001F601 Sending he_a to the server")
            send_msg(sock=self.socket, msg=he_a.serialize())
            he_a2: CKKSTensor = CKKSTensor.load(context=self.context,
                                                data=recv_msg(sock=self.socket))
            if verbose: print("\U0001F601 Received he_a2 from the server")
            a2: List = he_a2.decrypt().tolist() # the client decrypts he_a2
            a2: Tensor = torch.tensor(a2, requires_grad=True).to(self.device)
            a2.retain_grad()
            y_hat: Tensor = F.softmax(a2, dim=1)  # apply softmax
            # the client calculates the training loss (J) and accuracy
            batch_loss: Tensor = loss_func(y_hat, y)
            epoch_train_loss += batch_loss.item()
            epoch_correct += torch.sum(y_hat.argmax(dim=1) == y).item()
            epoch_total_samples += len(y)
            if verbose: print(f'batch {i+1} loss: {batch_loss:.4f}')
            
            if verbose: print("Backward Pass ---")
            batch_loss.backward()
            dJda2: Tensor = a2.grad.clone().detach()
            dJdW = torch.matmul(dJda2.T, a)              
            server_grads = {
                "dJda2": dJda2.to('cpu'),
                "dJdW": dJdW.detach().to('cpu')
            }
            if verbose: print("\U0001F601 Sending dJda2, dJdW to the server")
            send_msg(sock=self.socket, msg=pickle.dumps(server_grads))
            dJda: CKKSTensor = CKKSTensor.load(context=self.context,
                                               data=recv_msg(sock=self.socket))
            if verbose: print("\U0001F601 Received dJda from the server")
            dJda = torch.Tensor(dJda.decrypt().tolist()).to(self.device)
            a.backward(dJda)  # calculating the gradients w.r.t the conv layers
            optimizer.step()  # updating the parameters

            if i == total_batch-1:
                break

        return epoch_train_loss, epoch_correct, epoch_total_samples


def main():
    client = Client()
    client.init_socket(host='localhost', port=10080)
    hyperparams = pickle.loads(recv_msg(sock=client.socket))
    if hyperparams["verbose"]:
        print("\U0001F601 Received the hyperparameters from the Server")
        print(hyperparams)
    client.load_ecg_dataset(train_name="data/train_ecg.hdf5",
                            test_name="data/test_ecg.hdf5",
                            batch_size=hyperparams["batch_size"])
    # client.make_tenseal_context(4096, 
    #                             [40, 20, 20], 
    #                             pow(2, 21))
    client.make_tenseal_context(2048, 
                                [18, 18, 18], 
                                pow(2, 16))
    if hyperparams["verbose"]:
        print("\U0001F601 Sending the context to the server (without the private key)")
    client.send_context()
    client.build_model('weights/init_weight_256.pth')
    train_losses, train_accs = client.train(hyperparams)
    df = pd.DataFrame({  # save model training process into csv file
            'train_losses': train_losses,
            'train_accs': train_accs,
        })

    if hyperparams["save_model"]:
        torch.save(client.ecg_model.state_dict(), 
                   'weights/trained_client_256_2.pth')
        df.to_csv('outputs/loss_and_acc_2.csv')


if __name__ == "__main__":
    main()