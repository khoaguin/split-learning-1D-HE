from pathlib import Path
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from tqdm import tqdm
import logging
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
from icecream import ic

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

# A logger for this file
log = logging.getLogger(__name__)

data_dir = 'data'
train_name = 'train_ecg.hdf5'
test_name = 'test_ecg.hdf5'
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
    "dry_run": True
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
    checkpoint = torch.load(project_path/init_weights)
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


def train(client, client_model_ptr, server_model, train_rdl_ptr,
            optim_client_ptr, optim_server, criterion,
            train_losses, train_accs, total_batch):
    '''
    The split training process for one epoch
    '''
    train_loss = 0.0
    correct, total = 0, 0
    for i, batch in enumerate(tqdm(train_rdl_ptr)):
        x_ptr, y_gt_ptr = batch[0], batch[1]
        optim_server.zero_grad()
        optim_client_ptr.zero_grad()
        # compute and get the activation signals from the client model
        activs_ptr = client_model_ptr(x_ptr)
        # the server still gets access to plain activation signals
        activs = activs_ptr.clone().get(request_block=True)
        # the server continues the forward pass on the activation maps
        y_hat = server_model(activs)
        # the server asks to access ground truths in plain text
        y_gt = y_gt_ptr.get_copy()
        # calculates cross-entropy loss
        loss = criterion(y_hat, y_gt)
        train_loss += loss.item()
        correct += torch.sum(y_hat.argmax(dim=1) == y_gt).item()
        # backward propagation (calculating gradients of the loss w.r.t the weights)
        loss.backward()
        # send the gradients to the client
        client_grad_ptr = activs.grad.clone().send(client)
        # update the gradients of the client's model
        activs_ptr.backward(client_grad_ptr)
        # update the weights based on the gradients
        optim_client_ptr.step()
        optim_server.step()
        total += len(y_gt)
        if args["dry_run"]:
            if i==10:
                break
    
    train_losses.append(train_loss / total_batch)
    train_accs.append(correct / total)
    log.info(f'train loss: {train_losses[-1]: .4f}, train accuracy: {train_accs[-1]*100: 2f}')


def test(client_model_ptr, server_model, test_rdl_ptr, criterion,
            test_losses, test_accs, best_test_acc, total_batch):
    # testing
    with torch.no_grad():
        test_loss = 0.0
        correct, total = 0, 0
        for i, batch in enumerate(tqdm(test_rdl_ptr)):
            x_ptr, y_gt_ptr = batch[0], batch[1]
            # forward pass
            activs_ptr = client_model_ptr(x_ptr)
            activs = activs_ptr.clone().get(request_block=True)
            y_hat = server_model(activs)
            # the server asks to access ground truths in plain text
            y_gt = y_gt_ptr.get_copy()
            # calculate test loss
            loss = criterion(y_hat, y_gt)
            test_loss += loss.item()
            correct += torch.sum(y_hat.argmax(dim=1) == y_gt).item()
            total += len(y_gt)
            if args["dry_run"]:
                if i==10: 
                    break
                    
        test_losses.append(test_loss / total_batch)
        test_accs.append(correct / total)
        log.info(f'test loss: {test_losses[-1]: .4f}, test accuracy: {test_accs[-1]*100: 2f}')
    
    if test_accs[-1] > best_test_acc:
        best_test_acc = test_accs[-1]
        # save the best model
        client_model_ptr.get(
            request_block=True,
            reason="test evaluation",
            timeout_secs=5).save("./best-model-client.pt")
        server_model.save('best-model-server.pt')


def train_and_test(epochs, total_batch, client, client_model_ptr, server_model, criterion, 
                        train_rdl_ptr, test_rdl_ptr, optim_client_ptr, optim_server):
    train_losses = list()
    train_accs = list()
    test_losses = list()
    test_accs = list()
    best_test_acc = 0  # best test accuracy
    if args["dry_run"]:
        epochs = 5
    for epoch in range(1, epochs + 1):
        log.info(f"Epoch {epoch} --- ")
        epoch_start = time.time()
        train(client, client_model_ptr, server_model, train_rdl_ptr,
                optim_client_ptr, optim_server, criterion,
                train_losses, train_accs, total_batch)
        test(client_model_ptr, server_model, test_rdl_ptr, criterion,
                test_losses, test_accs, best_test_acc, total_batch)
        epoch_end = time.time()
        log.info(f"Epoch time: {int(epoch_end - epoch_start)} seconds")
    return train_losses, train_accs, test_losses, test_accs, best_test_acc


def plot_losses_accs(train_losses, train_accs, test_losses, test_accs):
    df = pd.DataFrame({  # save model training process into csv file
        'loss': train_losses,
        'test_loss': test_losses,
        'acc': train_accs,
        'test_acc': test_accs
    })
    df.to_csv('loss_and_acc.csv')
    df = pd.read_csv('loss_and_acc.csv')
    test_accs = df['test_acc']
    train_accs = df['acc']
    test_losses = df['test_loss']
    train_losses = df['loss']

    log.info(f'best train accuracy: {train_accs.max()*100:.2f} at epoch {train_accs.idxmax()+1}')
    log.info(f'best test accuracy: {test_accs.max()*100:.4f} at epoch {test_accs.idxmax()+1}')
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    ax[0].plot(train_losses, color='red')
    ax[0].plot(test_losses, color='green')
    ax[0].set_xticks([0, 100, 200, 300, 400])
    ax[0].set_xlabel('Epoch', size=16)
    ax[0].set_ylabel('Loss', size=16)
    ax[0].set_ylim(0.9, 1.1)
    ax[0].set_yticks([0.9, 1.0, 1.1, 1.2])
    ax[0].grid(alpha=0.5)
    ax[0].tick_params(labelsize=16)
    ax[0].legend(['Train', 'Test'], loc='right', fontsize=16)

    ax[1].set_ylim(0.7, 1.0)
    ax[1].set_yticks([0.7, 0.8, 0.9, 1.0])
    ax[1].plot(train_accs, color='red')
    ax[1].plot(test_accs, color='green')
    yt = ax[1].get_yticks()
    ax[1].set_yticklabels(['{:,.0%}'.format(x) for x in yt])
    ax[1].set_xticks([0, 100, 200, 300, 400])
    ax[1].set_xlabel('Epoch', size=16)
    ax[1].set_ylabel('Accuracy', size=16, labelpad=-5)
    ax[1].grid(alpha=0.5)
    ax[1].tick_params(labelsize=16)
    ax[1].legend(['Train', 'Test'], loc='right', fontsize=16)

    fig.savefig('loss_acc_conv2_split.png', bbox_inches='tight')


@hydra.main(config_path='conf', config_name='config')
def main(cfg : DictConfig) -> None:
    # saving some crucial information into the log file
    log.info(f'torch version: {torch.__version__}')
    log.info(f'syft version: {sy.__version__}')
    log.info(f'project_path: {project_path}')
    output_dir = Path(os.getcwd())
    log.info(f'output_path: {output_dir}')
    log.info(OmegaConf.to_yaml(cfg))
    # make the client and the server
    server: VirtualMachine = sy.VirtualMachine(name='server')
    client: VirtualMachineClient = server.get_root_client()
    remote_torch: Module = client.torch
    # the client load the dataset and save them into .pt files
    train_dataset = ECG(mode='train')
    test_dataset = ECG(mode='test')
    torch.save(train_dataset, output_dir/'train_dataset.pt')
    torch.save(test_dataset, output_dir/'test_dataset.pt')
    # the server get the pointer to the remote data loader
    train_rdl_ptr = get_remote_loader(path=str(output_dir/'train_dataset.pt'), 
                                        batch_size=args['batch_size'], 
                                        client=client,
                                        debug=False)
    test_rdl_ptr = get_remote_loader(path=str(output_dir/'test_dataset.pt'), 
                                        batch_size=args['batch_size'], 
                                        client=client,
                                        debug=False)
    # load the models
    ecg_client, ecg_server = loading_models()
    ecg_client_ptr = ecg_client.send(client)
    # prepare for the training process
    set_random_seed(seed=args['seed'])
    optim_client_ptr = remote_torch.optim.Adam(params=ecg_client_ptr.parameters(), lr=args['lr'])
    optim_server = torch.optim.Adam(params=ecg_server.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()
    # training process
    train_losses, train_accs, test_losses, test_accs, best_test_acc = \
        train_and_test(args['epochs'], args['total_batch'], client, ecg_client_ptr, ecg_server, criterion, 
                        train_rdl_ptr, test_rdl_ptr, optim_client_ptr, optim_server)
    plot_losses_accs(train_losses, train_accs, test_losses, test_accs)


@hydra.main(config_path=project_path/'conf', config_name='config')
def main2(cfg : DictConfig) -> None:
    # paths to files and directories
    project_path = Path(get_original_cwd()).parent
    project_parent = project_path.parent
    log.info(f'project_path: {project_path}')
    log.info(f'project_parent_path: {project_path.parent}')


if __name__ == "__main__":
    main2()
