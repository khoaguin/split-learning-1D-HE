from pathlib import Path
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

import syft as sy
from syft.core.node.vm.vm import VirtualMachine
from syft.core.node.vm.client import VirtualMachineClient

print(torch.__version__)
print(sy.__version__)

# paths to files and directories
project_path = Path.cwd().parent
print(project_path)
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


class ECG(Dataset):
    # The class used to load the ECG dataset
    def __init__(self, mode='train'):
        if mode == 'train':
            with h5py.File(project_path/data_dir/train_name, 'r') as hdf:
                self.x = hdf['x_train'][:]
                self.y = hdf['y_train'][:]
        elif mode == 'test':
            with h5py.File(project_path/data_dir/test_name, 'r') as hdf:
                self.x = hdf['x_test'][:]
                self.y = hdf['y_test'][:]
        else:
            raise ValueError('Argument of mode should be train or test')
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])



def main():
    server: VirtualMachine = sy.VirtualMachine(name="server")
    client: VirtualMachineClient = server.get_root_client() 


main()