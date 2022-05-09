from pathlib import Path
import socket
import pickle

import torch
import torch.nn as nn
from torch.optim import Adam

from sockets import send_msg, recv_msg


project_path = Path(__file__).absolute().parents[1]
print(f'project dir: {project_path}')


class EcgServer(nn.Module):
    def __init__(self):
        super(EcgServer, self).__init__()
        self.linear = nn.Linear(8 * 250, 5)

    def forward(self, x):
        x = self.linear(x)
        return x


def train(ecg_server, device, conn, total_batch):
    epoch = 10
    lr = 0.001
    optimizer = Adam(ecg_server.parameters(), lr=lr)
    for e in range(epoch):
        print("Epoch {} - ".format(e+1), end='')
        
        for _ in range(total_batch):
            # --- forward pass ---
            optimizer.zero_grad()  # initialize all gradients to zero
            a, _ = recv_msg(conn)
            a = pickle.loads(a)  # receive the activation maps from the client
            a = a.to(device)
            a.retain_grad()
            a2 = ecg_server(a)  # forward propagation
            msg = a2.clone().detach().requires_grad_(True)
            send_msg(conn, msg=pickle.dumps(msg))  # send a2 to the client
            # --- backward pass ---
            dJda2, _ = recv_msg(conn)
            dJda2 = pickle.loads(dJda2)  # receive dJ/da2 from the client
            # calculate the grads of the loss w.r.t 
            # the weights of the server model
            a2.backward(dJda2)
            dJda = a.grad.clone().detach()
            send_msg(conn, msg=pickle.dumps(dJda))
            optimizer.step()  # update the parameters
        
        train_status, _ = recv_msg(conn)
        train_status = pickle.loads(train_status)
        print(train_status)
        

def main():
    # connect to the client
    host = 'localhost'
    port = 10080
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)
    conn, addr = sock.accept()
    print('Conntected to', addr)
    # prepare for training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.get_device_name(0)
    ecg_server = EcgServer()
    checkpoint = torch.load("weights/init_weight_ptbxl.pth")
    ecg_server.linear.weight.data = checkpoint["linear.weight"]
    ecg_server.linear.bias.data = checkpoint["linear.bias"]
    ecg_server.to(device)
    # training
    total_batch = 4817
    train(ecg_server, device, conn, total_batch)
    # torch.save(ecg_server.state_dict(), 
    #            project_path/'weights/trained_server_split_plaintext.pth')


main()