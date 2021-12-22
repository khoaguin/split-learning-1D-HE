from pathlib import Path
import socket
import struct
import pickle

import torch
import torch.nn as nn
from torch.optim import Adam

project_path = Path(__file__).parents[1]
print(f'project dir: {project_path}')

def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


class EcgServer(nn.Module):
    def __init__(self):
        super(EcgServer, self).__init__()
        self.linear = nn.Linear(512, 5)

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
            a = pickle.loads(recv_msg(conn))  # receive the activation maps from the client
            a = a.to(device)
            a.retain_grad()
            a2 = ecg_server(a)  # forward propagation
            msg = a2.clone().detach().requires_grad_(True)
            send_msg(conn, msg=pickle.dumps(msg))  # send a2 to the client
            # --- backward pass ---
            dJda2 = pickle.loads(recv_msg(conn))  # receive dJ/da2 from the client
            # calculate the grads of the loss w.r.t 
            # the weights of the server model
            a2.backward(dJda2)
            dJda = a.grad.clone().detach()
            send_msg(conn, msg=pickle.dumps(dJda))
            optimizer.step()  # update the parameters
        
        train_status = pickle.loads(recv_msg(conn))
        print(train_status)
        

def main():
    # connect to the client
    host = 'localhost'
    port = 10080
    s = socket.socket()
    s.bind((host, port))
    s.listen(5)
    conn, addr = s.accept()
    print('Conntected to', addr)
    # prepare for training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.get_device_name(0)
    ecg_server = EcgServer()
    checkpoint = torch.load("./weights/init_weight.pth")
    ecg_server.linear.weight.data = checkpoint["linear.weight"]
    ecg_server.linear.bias.data = checkpoint["linear.bias"]
    ecg_server.to(device)
    # training
    total_batch = 3312
    train(ecg_server, device, conn, total_batch)
    torch.save(ecg_server.state_dict(), 
               './weights/trained_server.pth')


main()