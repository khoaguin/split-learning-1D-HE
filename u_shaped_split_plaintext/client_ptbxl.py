from pathlib import Path
import struct
import socket
import pickle
import time

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from sockets import send_msg, recv_msg

project_path = Path(__file__).absolute().parents[1]
print(f'project dir: {project_path}')


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
    def __init__(self):
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
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.pool2(x)
        x = x.view(-1, 8 * 250)

        return x


def train(ecg_client, train_loader, total_batch, device, soc):
    epoch = 10
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = Adam(ecg_client.parameters(), lr=lr)
    train_losses = list()
    train_accs = list()
    total_comm = 0

    for e in range(epoch):
        start = time.time()
        print("Epoch {} - ".format(e+1), end='')

        train_loss = 0.0
        correct, total = 0, 0
        for _, batch in enumerate(train_loader):
            # --- forward pass ---
            optimizer.zero_grad()  # initialize all gradients to zero
            x, y = batch  # load the data
            x, y = x.to(device), y.to(device)
            a = ecg_client(x)  # a: activation maps
            msg = a.clone().detach().requires_grad_(True)
            send_size1 = send_msg(soc, pickle.dumps(msg))  # send the activation maps to the server
            a2, recv_size1 = recv_msg(soc)
            a2 = pickle.loads(a2)  # receives a2 from the server
            a2.retain_grad()
            y_hat = F.softmax(a2, dim=1)
            J = criterion(y_hat, y)  # the client calculate the loss
            # --- backward pass ---
            J.backward()
            dJda2 = a2.grad.clone().detach()
            send_size2 = send_msg(soc, pickle.dumps(dJda2))  # send dJda2 to the server
            dJda, recv_size2 = recv_msg(soc)
            dJda = pickle.loads(dJda)   # receives dJ/da from the server
            # calculate the grad of the loss w.r.t the 
            # weights of the client's model
            a.backward(dJda)  
            optimizer.step()  # update the parameters

            train_loss += J.item()
            correct += torch.sum(y_hat.argmax(dim=1) == y).item()
            total += len(y)
            total_comm += send_size1+send_size2+recv_size1+recv_size2

        end = time.time()
        train_losses.append(train_loss / total_batch)
        train_accs.append(correct / total)
        train_status = f"training loss: {train_losses[-1]:.4f}, "\
                       f"training acc: {train_accs[-1]*100:.2f}, "\
                       f"training time: {end-start:.2f}s"
        print(train_status, end='\n')
        send_msg(sock=soc, msg=pickle.dumps(train_status))
    
    print(f"total communication: {total_comm:.2f} (Mb)")
    return train_losses, train_accs
        

def main():
    # connecting to the server
    host = 'localhost'
    port = 10080
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.connect((host, port))
    print(f'connected to {sock}')
    # making the dataset
    batch_size = 4
    train_dataset = PTBXL(train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    total_batch = 4817  # 19267 / 4
    # prepare for training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.get_device_name(0)
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ecg_client = ECGClient()
    checkpoint = torch.load("weights/init_weight_ptbxl.pth")
    ecg_client.conv1.weight.data = checkpoint["conv1.weight"]
    ecg_client.conv1.bias.data = checkpoint["conv1.bias"]
    ecg_client.conv2.weight.data = checkpoint["conv2.weight"]
    ecg_client.conv2.bias.data = checkpoint["conv2.bias"]
    ecg_client.to(device)
    # training
    train_losses, train_accs = train(ecg_client, train_loader,  
                                     total_batch, device, sock)

    # save model training process into csv file
    df = pd.DataFrame({  
        'train_losses': train_losses,
        'train_accs': train_accs,
    })
    # df.to_csv(project_path/'outputs/loss_and_acc_split_plaintext.csv')
    # torch.save(ecg_client.state_dict(), 
    #            project_path/'weights/trained_client_split_plaintext.pth')


if __name__ == "__main__":
    main()