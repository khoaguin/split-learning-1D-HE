import h5py
from icecream import ic

import torch
from torch.utils.data import Dataset, DataLoader

from syft.core.node.vm.client import VirtualMachineClient
from syft.core.remote_dataloader import RemoteDataset
from syft.core.remote_dataloader import RemoteDataLoader


class ECG(Dataset):
    '''
    The class used to load the ECG dataset
    '''
    def __init__(self, path, mode='train'):
        if mode == 'train':
            with h5py.File(path, 'r') as hdf:
                self.x = torch.tensor(hdf['x_train'][:], dtype=torch.float)
                self.y = torch.tensor(hdf['y_train'][:])                
        elif mode == 'test':
            with h5py.File(path, 'r') as hdf:
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
    Get the pointer to the remote data loader of the dataset on the client's side.
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