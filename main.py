from pathlib import Path
import logging
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
from icecream import ic
ic.configureOutput(includeContext=True)

import torch
import torch.nn as nn

import syft as sy
from syft.core.node.vm.vm import VirtualMachine
from syft.core.node.vm.client import VirtualMachineClient
from syft.ast.module import Module
from syft.core.remote_dataloader import RemoteDataset
from syft.core.remote_dataloader import RemoteDataLoader

from models.ecg_two_convs import EcgClient, EcgServer, loading_models
from utils.datasets import ECG, get_remote_loader
from utils.processes import set_random_seed, train_and_test
from utils.tools import save_results, plot_losses_accs


log = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='config')
def vanilla_split_two_convs(cfg : DictConfig) -> None:
    # logging some important information
    log.info(f'torch version: {torch.__version__}')
    log.info(f'syft version: {sy.__version__}')
    project_path = Path(get_original_cwd())
    log.info(f'project_path: {project_path}')
    output_path = Path(os.getcwd())
    log.info(f'output_path: {output_path}')
    log.info(OmegaConf.to_yaml(cfg))
    # make the client and the server
    server: VirtualMachine = sy.VirtualMachine(name='server')
    client: VirtualMachineClient = server.get_root_client()
    remote_torch: Module = client.torch
    # the client load the dataset and save them into .pt files
    train_dataset = ECG(path=project_path/'data/train_ecg.hdf5', mode='train')
    test_dataset = ECG(path=project_path/'data/test_ecg.hdf5', mode='test')
    torch.save(train_dataset, output_path/'train_dataset.pt')
    torch.save(test_dataset, output_path/'test_dataset.pt')
    # the server get the pointers to the remote data loader
    train_rdl_ptr = get_remote_loader(path=str(output_path/'train_dataset.pt'), 
                                        batch_size=cfg.batch_size, 
                                        client=client,
                                        debug=False)
    test_rdl_ptr = get_remote_loader(path=str(output_path/'test_dataset.pt'), 
                                        batch_size=cfg.batch_size, 
                                        client=client,
                                        debug=False)
    # load the models
    ecg_client, ecg_server = loading_models(init_weights_path=project_path/'data/init_weight.pth')
    ecg_client_ptr = ecg_client.send(client)
    # prepare for the training process
    set_random_seed(seed=cfg.seed)
    optim_client_ptr = remote_torch.optim.Adam(params=ecg_client_ptr.parameters(), lr=cfg.lr)
    optim_server = torch.optim.Adam(params=ecg_server.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    # training process
    train_losses, train_accs, test_losses, test_accs, best_test_acc = \
        train_and_test(cfg.epochs, cfg.total_batch, cfg.dry_run, cfg.save_model,
                            client, ecg_client_ptr, ecg_server, criterion, 
                            train_rdl_ptr, test_rdl_ptr, optim_client_ptr, optim_server)
    # save the results and plot them
    results_path = output_path / 'loss_and_acc.cvs'
    save_results(train_losses, train_accs, test_losses, test_accs, results_path)
    plot_losses_accs(cfg.dry_run, results_path)
    

@hydra.main(config_path='conf', config_name='config2')
def split_learning_he(cfg: DictConfig) -> None:
    project_path = Path(get_original_cwd())
    log.info(f'project_path: {project_path}')
    output_path = Path(os.getcwd())
    log.info(f'output_path: {output_path}')
    log.info(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    split_learning_he()