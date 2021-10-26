import os
import pickle
import socket
import struct
import time
from pathlib import Path
from typing import List, Union
import math

import h5py
import numpy as np
import tenseal as ts
import torch
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

print(f'torch version: {torch.__version__}')
print(f'tenseal version: {ts.__version__}')

project_path = Path.cwd().parent.parent
print(f'project_path: {project_path}')


def recvall(sock, n) -> Union[None, bytes]:
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


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


class EcgClient(nn.Module):
    """The client's 1D CNN model

    Args:
        nn ([torch.Module]): [description]
    """
    def __init__(self, context: Context, init_weight_path: Union[str, Path]):
        super(EcgClient, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)  # 128 x 16
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2)  # 32 x 16
        
        self.load_init_weights(init_weight_path)
        self.context = context
    
    def forward(self, x: Tensor) -> CKKSTensor:
        x = self.conv1(x)  # [batch_size, 16, 128]
        x = self.relu1(x)
        x = self.pool1(x)  # [batch_size, 16, 64]
        x = self.conv2(x)  # [batch_size, 16, 64]
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 16)  # [batch_size, 16, 32]
        enc_x: CKKSTensor = ts.ckks_tensor(self.context, x.tolist())  # [batch_size, 512]
        return enc_x
    
    def load_init_weights(self, init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.conv1.weight.data = checkpoint["conv1.weight"]
        self.conv1.bias.data = checkpoint["conv1.bias"]
        self.conv2.weight.data = checkpoint["conv2.weight"]
        self.conv2.bias.data = checkpoint["conv2.bias"]

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

    def load_ecg_dataset(self, data_dir: str, 
                            train_name: str, test_name: str,
                            batch_size: int) -> None:
        """[summary]

        Args:
            data_dir (str): [description]
            train_name (str): [description]
            test_name (str): [description]
            batch_size (int): [description]
        """
        train_dataset = ECG(data_dir, train_name, test_name, train=True)
        test_dataset = ECG(data_dir, train_name, test_name, train=False)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
    def init_socket(self, host, port) -> None:
        """[summary]

        Args:
            host ([str]): [description]
            port ([int]): [description]
        """
        self.socket = socket.socket()
        self.socket.connect((host, port))  # connect to a remote [server] address,

    def send_msg(self, msg) -> None:
        # prefix each message with a 4-byte length in network byte order
        msg = struct.pack('>I', len(msg)) + msg
        self.socket.sendall(msg)

    def recv_msg(self):
        # read message length and unpack it into an integer
        raw_msglen = recvall(self.socket, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # read the message data
        data: bytes = recvall(self.socket, msglen)
        return data

    def make_tenseal_context(self, poly_modulus_degree: int,
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
        self.send_msg(self.context.serialize(save_secret_key=False))

    def build_model(self, init_weight_path: Union[str, Path]) -> None:
        """Build the neural network model for the client

        Raises:
            TypeError: if the tenseal context needed to encrypt the activation 
                        map is None, then raise an error
        """
        if self.context == None:
            raise TypeError("tenseal context is None")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'client device: {torch.cuda.get_device_name(0)}')
        self.ecg_model = EcgClient(context=self.context, 
                                    init_weight_path=init_weight_path)
        self.ecg_model.to(self.device)

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
        if self.ecg_model == None:
            raise TypeError("client's model has not been constructed yet")
        train_losses = list()
        train_accs = list()
        test_losses = list()
        test_accs = list()
        best_test_acc = 0  # best test accuracy
        loss_func = nn.CrossEntropyLoss()

        for e in range(epoch):
            print(f"Epoch {e+1} -----")
            epoch_train_loss = 0.0
            epoch_correct, epoch_total_samples = 0, 0

            # training loop
            for i, batch in enumerate(self.train_loader):
                # --- Forward Pass ---
                start = time.time()
                x, y = batch  # get the input data and ground-truth output in the batch
                x, y = x.to(self.device), y.to(self.device)  # put to cuda or cpu
                he_a: CKKSTensor = self.ecg_model(x)  # [batch_size, 512]
                # converting the HE encrypted activation maps into byte stream
                he_a_bytes: bytes = he_a.serialize() 
                # the client sends the byte streams to the server
                print("\U0001F601 Sending he_a to the server")
                self.send_msg(msg=he_a_bytes)
                # the client receives the encrypted activations after 2 linear layers 
                # from the server
                he_a2_bytes: bytes = self.recv_msg()
                print("\U0001F601 Received he_a2 from the server")
                he_a2: CKKSTensor = CKKSTensor.load(context=self.context,
                                                    data=he_a2_bytes)
                # the client decrypts he_a2 and apply softmax to get the predictions
                a2: List = he_a2.decrypt().tolist() # the client decrypts he_a2
                a2: Tensor = torch.tensor(a2, requires_grad=True).to(self.device)
                a2.retain_grad()
                y_hat: Tensor = F.softmax(a2, dim=1)  # apply softmax
                print(f'y_hat: {y_hat}')
                # the client calculates the training loss (J) and accuracy
                batch_loss: Tensor = loss_func(y_hat, y)
                epoch_train_loss += batch_loss.item()
                epoch_correct += torch.sum(y_hat.argmax(dim=1) == y).item()
                epoch_total_samples += len(y)

                end = time.time()
                print(f'time taken for the one forward pass: {end-start}')

                # --- Backward pass ---
                # calculates the gradients of the loss w.r.t y_hat and a2
                batch_loss.backward()
                dJda2: Tensor = a2.grad.clone().detach().to("cpu")
                # send the gradients to the server
                dJda2_bytes: bytes = pickle.dumps(dJda2)
                print("\U0001F601 Sending dJda2 to the server")
                self.send_msg(msg=dJda2_bytes)

                if i == total_batch-1: break
            
            # save the average training losses and accuracies over all batches for each epoch
            train_losses.append(epoch_train_loss / total_batch)
            train_accs.append(epoch_correct / epoch_total_samples)

    def test():
        pass


def main(hyperparams):
    client = Client()
    client.init_socket(host, port)
    # sending hyperparameters to the server
    ic(hyperparams)
    print("Sending hyperparams to the server")
    client.send_msg(msg=pickle.dumps(hyperparams))
    client.load_ecg_dataset('data', 'train_ecg.hdf5', 'test_ecg.hdf5', 
                            hyperparams['batch_size'])
    client.make_tenseal_context(8192, 
                                [21, 21, 21, 21, 21, 21, 21], 
                                pow(2, 21))
    print("Sending the context to the server (without the private key)")
    client.send_context()
    client.build_model('init_weight.pth')
    client.set_random_seed(hyperparams['seed'])
    client.train(epoch=hyperparams['epoch'], 
                 total_batch=hyperparams['total_batch'])

    
if __name__ == '__main__':
    # some global variables
    host = 'localhost'
    port = 10080
    max_recv = 4096
    
    dry_run = True # break after 2 batches for 2 epoch, set batch size to be 2
    if dry_run:
        batch_size = 2
        epoch = 2
        total_batch = 2
    else:
        batch_size = 32
        epoch = 400
        total_batch = math.ceil(13245/batch_size)
    lr = 0.001
    seed = 0

    hyperparams = {
        'batch_size': batch_size,
        'total_batch': total_batch,
        'epoch': epoch,
        'lr': lr,
        'seed': seed
    }

    main(hyperparams)
