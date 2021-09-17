from typing import Union
from pathlib import Path

import syft as sy
import torch
import torch.nn as nn


class EcgClient(sy.Module):
    '''
    The model on the client side contains the convolutional layers
    '''
    def __init__(self, torch_ref):
        super(EcgClient, self).__init__(torch_ref=torch_ref)
        self.conv1 = self.torch_ref.nn.Conv1d(1, 16, 7, padding=3)  # 128 x 16
        self.relu1 = self.torch_ref.nn.LeakyReLU()
        self.pool1 = self.torch_ref.nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = self.torch_ref.nn.Conv1d(16, 16, 5, padding=2)  # 64 x 16
        self.relu2 = self.torch_ref.nn.LeakyReLU()
        self.pool2 = self.torch_ref.nn.MaxPool1d(2)  # 32 x 16
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 16)
        return x


class EcgServer(sy.Module):
    '''
    The model on the server side contains fully connected layers
    '''
    def __init__(self, torch_ref):
        super(EcgServer, self).__init__(torch_ref=torch_ref)
        self.linear3 = nn.Linear(32 * 16, 128)
        self.relu3 = nn.LeakyReLU() 
        self.linear4 = nn.Linear(128, 5)
        self.softmax4 = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.softmax4(x)
        return x


def loading_models(init_weights_path: Union[str, Path]):
    ''''
    Loading the models and set some initial weights according to 
    https://github.com/SharifAbuadbba/split-learning-1D
    '''
    ecg_client = EcgClient(torch_ref=torch)
    checkpoint = torch.load(init_weights_path)
    ecg_client.conv1.weight.data = checkpoint["conv1.weight"]
    ecg_client.conv1.bias.data = checkpoint["conv1.bias"]
    ecg_client.conv2.weight.data = checkpoint["conv2.weight"]
    ecg_client.conv2.bias.data = checkpoint["conv2.bias"]

    ecg_server = EcgServer(torch_ref=torch)
    ecg_server.linear3.weight.data = checkpoint["linear3.weight"]
    ecg_server.linear3.bias.data = checkpoint["linear3.bias"]
    ecg_server.linear4.weight.data = checkpoint["linear4.weight"]
    ecg_server.linear4.bias.data = checkpoint["linear4.bias"]
    
    return ecg_client, ecg_server