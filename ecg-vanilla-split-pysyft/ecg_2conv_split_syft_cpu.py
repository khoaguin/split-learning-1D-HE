from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from icecream import ic
ic.configureOutput(includeContext=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD

import syft as sy
from syft.core.node.vm.vm import VirtualMachine
from syft.core.node.vm.client import VirtualMachineClient
from syft.ast.module import Module
from syft.core.remote_dataloader import RemoteDataset
from syft.core.remote_dataloader import RemoteDataLoader

print(f'torch version: {torch.__version__}')
print(f'syft version: {sy.__version__}')


# paths to files and directories
project_path = Path.cwd().parent
print(f'project_path: {project_path}')
data_dir = 'mitdb'
train_name = 'train_ecg.hdf5'
test_name = 'test_ecg.hdf5'
all_name = 'all_ecg.hdf5'
model_dir = 'model'
model_name = 'conv2'
model_ext = '.pth'
csv_dir = 'csv'
csv_ext = '.csv'
csv_name = 'conv2'
csv_accs_name = 'accs_conv2'
init_weights = 'init_weight.pth'


# hyper params
args = {
    "batch_size": 32,
    "total_batch": 414,  # 32*414=13248. We have 13245 data samples
    "test_batch_size": 32,
    "epochs": 400,
    "lr": 0.001,
    "seed": 0,
    "cuda": False,
    "log_interval": 10,
    "save_model": True,
}


class ECG(Dataset):
    '''
    The class used to load the ECG dataset
    '''
    def __init__(self, mode='train'):
        if mode == 'train':
            with h5py.File(project_path/data_dir/train_name, 'r') as hdf:
                self.x = torch.tensor(hdf['x_train'][:], dtype=torch.float)
                self.y = torch.tensor(hdf['y_train'][:])                
        elif mode == 'test':
            with h5py.File(project_path/data_dir/test_name, 'r') as hdf:
                self.x = torch.tensor(hdf['x_test'][:], dtype=torch.float)
                self.y = torch.tensor(hdf['y_test'][:])
        else:
            raise ValueError('Argument of mode should be train or test')
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_remote_loader(path: str, batch_size: int, client: VirtualMachineClient, debug=False):
    '''
    Get the pointer of the remote data loader to the dataset on the client's side.
    '''
    rds = RemoteDataset(path=path, data_type="torch_tensor")
    rdl = RemoteDataLoader(remote_dataset=rds, batch_size=batch_size)
    rdl_ptr = rdl.send(client)
    # call create_dataset to create the real Dataset object on remote side
    rdl_ptr.load_dataset()
    # call create_dataloader to create the real DataLoader object on remote side
    rdl_ptr.create_dataloader()
    if debug:
        for i, batch in enumerate(rdl_ptr):
            x_ptr, y_ptr = batch[0], batch[1]
            assert isinstance(x_ptr.get_copy(), torch.Tensor)
            assert isinstance(y_ptr.get_copy(), torch.Tensor)
            if i<2:
                ic(x_ptr, y_ptr, len(x_ptr))
                ic(x_ptr.get_copy(), y_ptr.get_copy())

    return rdl_ptr


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


def loading_models():
    ''''
    Loading the models and set some initial weights according to 
    https://github.com/SharifAbuadbba/split-learning-1D
    '''
    ecg_client = EcgClient(torch_ref=torch)
    checkpoint = torch.load(init_weights)
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


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(client_model_ptr, server_model, remote_torch, 
            train_loader_ptr, test_loader_ptr, optim_client, optim_server):
    '''
    The split training process
    '''
    train_losses = list()
    train_accs = list()
    test_losses = list()
    test_accs = list()
    best_test_acc = 0  # best test accuracy


def main():
    # make the client and the server
    server: VirtualMachine = sy.VirtualMachine(name="server")
    client: VirtualMachineClient = server.get_root_client()
    remote_torch: Module = client.torch
    # the client load the dataset and save them into .pt files
    train_dataset = ECG(mode='train')
    test_dataset = ECG(mode='test')
    torch.save(train_dataset, "train_dataset.pt")
    torch.save(test_dataset, "test_dataset.pt")
    # the server get the pointer to the remote data loader
    train_rdl_ptr = get_remote_loader(path='train_dataset.pt', 
                                        batch_size=args["batch_size"], 
                                        client=client,
                                        debug=False)
    test_rdl_ptr = get_remote_loader(path='train_dataset.pt', 
                                        batch_size=args["batch_size"], 
                                        client=client,
                                        debug=False)
    # load the models
    ecg_client, ecg_server = loading_models()
    ecg_client_ptr = ecg_client.send(client)
    # prepare for the training process
    set_random_seed(seed=args["seed"])
    optim_client = remote_torch.optim.Adam(params=ecg_client_ptr.parameters(), lr=args["lr"])
    optim_server = torch.optim.Adam(params=ecg_server.parameters(), lr=args["lr"])
    criterion = nn.CrossEntropyLoss()
    train

main()
