import os
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List, Union, Tuple

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

project_path = Path(__file__).absolute().parents[1]


class PTBXL(Dataset):
    """
    The class used by the client to 
    load the PTBXL dataset

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, train=True):
        if train:
            with h5py.File(project_path/'data/train_ptbxl.hdf5', 'r') as hdf:
                self.x = hdf['X_train'][:]
                self.y = hdf['y_train'][:]
        else:
            with h5py.File(project_path/'data/test_ptbxl.hdf5', 'r') as hdf:
                self.x = hdf['X_test'][:]
                self.y = hdf['y_test'][:]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])


class ECGClient(nn.Module):
    def __init__(self,
                 context: Context,
                 init_weight_path: Union[str, Path]):
        super(ECGClient, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, 
                               out_channels=16, 
                               kernel_size=7, 
                               padding=3,
                               stride=1)  # 16 x 1000
        self.lrelu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 16 x 500
        self.conv2 = nn.Conv1d(in_channels=16, 
                               out_channels=8, 
                               kernel_size=5, 
                               padding=2)  # 8 x 500
        self.lrelu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2)  # 8 x 250

        self.load_init_weights(init_weight_path)
        self.context = context

    def load_init_weights(self, init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.conv1.weight.data = checkpoint["conv1.weight"]
        self.conv1.bias.data = checkpoint["conv1.bias"]
        self.conv2.weight.data = checkpoint["conv2.weight"]
        self.conv2.bias.data = checkpoint["conv2.bias"]

    def forward(self, 
                x: Tensor,
                batch_enc: bool) -> Tuple[Tensor, CKKSTensor]:
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.pool2(x)
        x = x.view(-1, 8 * 250)        
        enc_x: CKKSTensor = ts.ckks_tensor(self.context, x.tolist(), batch=batch_enc)

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
    
    def load_ptbxl_dataset(self, 
                           batch_size: int) -> None:
        """[summary]

        Args:
            train_name (str): [description]
            test_name (str): [description]
            batch_size (int): [description]
        """
        train_dataset = PTBXL(train=True)
        test_dataset = PTBXL(train=False)
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
        _ = send_msg(sock=self.socket,
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
        self.ecg_model = ECGClient(context=self.context, 
                                   init_weight_path=init_weight_path)
        self.ecg_model.to(self.device)

    def train(self, hyperparams: dict) -> None:
        # get the hyperparameters
        seed = hyperparams["seed"]
        verbose = hyperparams["verbose"]
        lr = hyperparams["lr"]
        total_batch = hyperparams["total_batch"]
        epoch = hyperparams["epoch"]
        batch_encrypted = hyperparams['batch_encrypted']
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
            e_train_loss, e_correct, e_samples = \
                self.training_loop(verbose, loss_func, optimizer, batch_encrypted)
            end = time.time()
            train_losses.append(e_train_loss / total_batch)
            train_accs.append(e_correct / e_samples)
            train_status = f"training loss: {train_losses[-1]:.4f}, "\
                           f"training acc: {train_accs[-1]*100:.2f}, "\
                           f"training time: {end-start:.2f}s"
            print(train_status)
            _ = send_msg(sock=self.socket, msg=pickle.dumps(train_status))
                
        return train_losses, train_accs

    def training_loop(self, verbose, loss_func, optimizer, batch_encrypted):
        epoch_train_loss = 0.0
        epoch_correct = 0 
        epoch_total_samples = 0
        for i, batch in enumerate(self.train_loader):
            start = time.time()
            optimizer.zero_grad()
            if verbose: print("Forward Pass ---")
            x, y = batch  # get the input data and ground-truth output in the batch
            x, y = x.to(self.device), y.to(self.device)  # put to cuda or cpu
            a, he_a = self.ecg_model.forward(x, batch_encrypted)
            if verbose: print("\U0001F601 Sending he_a to the server")
            _ = send_msg(sock=self.socket, msg=he_a.serialize())
            he_a2_bytes, _ = recv_msg(sock=self.socket)
            he_a2: CKKSTensor = CKKSTensor.load(context=self.context,
                                                data=he_a2_bytes)
            if verbose: print(f"\U0001F601 Received he_a2 from the server with shape {he_a2.shape}")
            a2: List = he_a2.decrypt().tolist() # the client decrypts he_a2
            a2: Tensor = torch.tensor(a2, requires_grad=True)
            a2 = a2.squeeze(dim=1).to(self.device)  # [batch_size, 5]
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
            _ = send_msg(sock=self.socket, msg=pickle.dumps(server_grads))
            dJda_bytes, _ = recv_msg(sock=self.socket)
            dJda = pickle.loads(dJda_bytes)
            if verbose: print(f"\U0001F601 Received dJda from the server with shape {dJda.shape}")
            # dJda = torch.Tensor(dJda.decrypt().tolist()).to(self.device)
            dJda = dJda.to(self.device)
            a.backward(dJda)  # calculating the gradients w.r.t the conv layers
            optimizer.step()  # updating the parameters
            end = time.time()
            if verbose: print(f"training time for 1 batch: {end-start:.2f}s")

        return epoch_train_loss, epoch_correct, epoch_total_samples


def main():
    # establish the connection, receive the hyperparams
    client = Client()
    client.init_socket(host='localhost', port=10080)
    hyperparams, _ = recv_msg(sock=client.socket)
    hyperparams = pickle.loads(hyperparams)
    if hyperparams["verbose"]:
        print("\U0001F601 Received the hyperparameters from the Server")
        print(hyperparams)
    # make the tenseal context and send it (without the private key) to the server
    client.make_tenseal_context(4096,   # 4096b
                                [40, 20, 20],
                                pow(2, 21))
    if hyperparams["verbose"]:
        print("\U0001F601 Sending the context to the server (without the private key)")
    client.send_context()
    # load the dataset
    client.load_ptbxl_dataset(batch_size=hyperparams["batch_size"])
    # build and train the model
    client.build_model(project_path/'u_shaped_split_he/weights/init_weight_ptbxl.pth')
    train_losses, train_accs = client.train(hyperparams)
    # save the training results and model
    if hyperparams["save_model"]:
        df = pd.DataFrame({  # save model training process into csv file
            'train_losses': train_losses,
            'train_accs': train_accs,
        })
        df.to_csv(project_path/'u_shaped_split_he/outputs/loss_and_acc_ptbxl_4096b.csv')
        torch.save(client.ecg_model.state_dict(), 
                   project_path/'u_shaped_plit_he/weights/trained_client_ptbxl_4096b.pth')


if __name__ == "__main__":
    print(f'project dir: {project_path}')
    main()