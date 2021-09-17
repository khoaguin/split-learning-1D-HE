import numpy as np
import torch
from icecream import ic
import logging
import time
from tqdm import tqdm
ic.configureOutput(includeContext=True)
log = logging.getLogger(__name__)


def set_random_seed(seed: int):
    # ic(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(client, client_model_ptr, server_model, train_rdl_ptr,
            optim_client_ptr, optim_server, criterion,
            train_losses, train_accs, total_batch, dry_run):
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
        if dry_run:
            if i==10:
                break
    
    train_losses.append(train_loss / total_batch)
    train_accs.append(correct / total)
    log.info(f'train loss: {train_losses[-1]: .4f}, train accuracy: {train_accs[-1]*100: 2f}')


def test(client_model_ptr, server_model, test_rdl_ptr, criterion,
            test_losses, test_accs, best_test_acc, total_batch, dry_run, save_model):
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
            if dry_run:
                if i==10: 
                    break
        
        test_losses.append(test_loss / total_batch)
        test_accs.append(correct / total)
        log.info(f'test loss: {test_losses[-1]: .4f}, test accuracy: {test_accs[-1]*100: 2f}')
    
    if test_accs[-1] > best_test_acc:
        best_test_acc = test_accs[-1]
        # save the best model
        if save_model:
            client_model_ptr.get(
                request_block=True,
                reason="test evaluation",
                timeout_secs=5).save("./best-model-client.pt")
            server_model.save('best-model-server.pt')


def train_and_test(epochs, total_batch, dry_run, save_model, 
                    client, client_model_ptr, server_model, criterion, 
                    train_rdl_ptr, test_rdl_ptr, optim_client_ptr, optim_server):
    train_losses = list()
    train_accs = list()
    test_losses = list()
    test_accs = list()
    best_test_acc = 0  # best test accuracy
    if dry_run:
        epochs = 5
    for epoch in range(1, epochs + 1):
        log.info(f"Epoch {epoch} --- ")
        epoch_start = time.time()
        train(client, client_model_ptr, server_model, train_rdl_ptr,
                optim_client_ptr, optim_server, criterion,
                train_losses, train_accs, total_batch, dry_run)
        test(client_model_ptr, server_model, test_rdl_ptr, criterion,
                test_losses, test_accs, best_test_acc, total_batch, dry_run, save_model)
        epoch_end = time.time()
        log.info(f"Epoch time: {int(epoch_end - epoch_start)} seconds")
    return train_losses, train_accs, test_losses, test_accs, best_test_acc
