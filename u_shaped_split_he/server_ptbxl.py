import pickle
import socket
from pathlib import Path
from typing import Union

from sockets import send_msg, recv_msg

import numpy as np
import tenseal as ts
import torch
from torch import Tensor
from icecream import ic
ic.configureOutput(includeContext=True)
from tenseal.enc_context import Context
from tenseal.tensors.ckkstensor import CKKSTensor
print(f'tenseal version: {ts.__version__}')
print(f'torch version: {torch.__version__}')

project_path = Path(__file__).absolute().parents[1]


class ECGServer:
    def __init__(self, 
                 init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.params = dict(
            W = checkpoint["linear.weight"],  # [5, 256],
            b = checkpoint["linear.bias"]  # [5]
        )
        self.grads = dict(
            dJdW = torch.zeros(self.params["W"].shape),
            dJdb = torch.zeros(self.params["b"].shape),
        )
        self.cache = dict()

    def enc_linear(self, 
                   enc_x: CKKSTensor, 
                   W: Tensor, 
                   b: Tensor,
                   batch_encrypted: bool):
        """
        The linear layer on homomorphic encrypted data
        Based on https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        """
        if batch_encrypted:
            enc_x.reshape_([1, enc_x.shape[0]])
        Wt = W.T
         # if batch_encrypted is true, y's shape will be [1, 5],
         # otherwise [batch_size, 5]
        y: CKKSTensor = enc_x.mm(Wt) + b
        dydW = enc_x
        dydx = W
        return y, dydW, dydx

    def forward(self, 
                he_a: CKKSTensor, 
                batch_encrypted: bool) -> CKKSTensor:
        # a2 = a*W' + b
        he_a2, _, W = self.enc_linear(he_a, 
                                      self.params["W"],
                                      self.params["b"],
                                      batch_encrypted)
        self.cache["da2da"] = W
        return he_a2

    def backward(self, 
                 dJda2: Tensor, 
                 context: Context):
        """Calculate the gradients of the loss function w.r.t the bias
           and the encrypted activation map a received from the client    

        Args:
            dJda2 (Tensor): the derivative of the loss function w.r.t the output
                            of the linear layer. shape: [batch_size, 5]
            context (Context): the tenseal context, used to encrypt the output

        Returns:
            dJda (CKKSTensor): the derivative of the loss function w.r.t the
                               activation map received from the client. 
                               This will be sent to the client so he can calculate
                               the gradients w.r.t the conv layers weights.
        """
        self.grads["dJdb"] = dJda2.sum(0)  # sum accross all samples in a batch
        assert self.grads["dJdb"].shape == self.params["b"].shape, \
            "the grad of the loss function w.r.t b and b must have the same shape"

        dJda: Tensor = torch.matmul(dJda2, self.cache["da2da"])
        # dJda: CKKSTensor = ts.ckks_tensor(context, dJda.tolist())

        return dJda

    def clear_grad_and_cache(self):
        """Clear the cache dictionary and make all grads zeros for the 
           next forward pass on a new batch
        """
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

    def check_update_grads(self, dJdW) -> None:
        """Check and update the gradients in plaintext received from the client

        Args:
            grads_plaintext ([dict]): the dictionary that contains the 
                                      gradients needed in plaintext
        """
        assert dJdW.shape == self.grads["dJdW"].shape, \
            f"dJdW received from the client is in wrong shape"

        self.grads["dJdW"] = dJdW

    def update_params(self, lr: float):
        """
        Update the parameters based on the gradients calculated in backward()
        """
        self.params["W"] = self.params["W"] - lr*self.grads["dJdW"]
        self.params["b"] = self.params["b"] - lr*self.grads["dJdb"]


class Server:
    def __init__(self) -> None:
        self.socket = None
        self.device = None
        self.ecg_model = None
        self.client_ctx = None
        self.connection = None

    def init_socket(self, host, port):
        """[summary]

        Args:
            host ([str]): [description]
            port ([int]): [description]
        """
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))  # associates the socket with its local address
        self.socket.listen()
        print('Listening on', (host, port))
        self.connection, addr = self.socket.accept()  # wait for the client to connect
        print(f'Connected. \nAddress: {addr}')

    def recv_ctx(self):
        client_ctx_bytes, _ = recv_msg(sock=self.connection)
        self.client_ctx: Context = Context.load(client_ctx_bytes)

    def build_model(self, 
                    init_weight_path: Union[str, Path]) -> None:
        """Build the neural network model for the server
        """
        self.ecg_model = ECGServer(init_weight_path)

    def train(self, hyperparams: dict):
        seed = hyperparams["seed"]
        verbose = hyperparams["verbose"]
        lr = hyperparams["lr"]
        total_batch = hyperparams["total_batch"]
        epoch = hyperparams["epoch"]
        batch_encrypted = hyperparams["batch_encrypted"]
        # set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        for e in range(epoch):
            print(f"---- Epoch {e+1} ----")
            self.training_loop(total_batch, verbose, lr, batch_encrypted, epoch)
            train_status, _ = recv_msg(self.connection)
            print(pickle.loads(train_status))

    def training_loop(self, total_batch, verbose, lr, batch_encrypted, epoch):
        avg_communication = 0
        for i in range(total_batch):
            # forward pass
            if verbose: print("Forward pass ---")
            self.ecg_model.clear_grad_and_cache()
            # receive the encrypted activation maps from the client
            he_a, recv_size1 = recv_msg(sock=self.connection)
            he_a: CKKSTensor = CKKSTensor.load(context=self.client_ctx,
                                               data=he_a)
            if verbose: print(f"\U0001F601 Received he_a from the client with shape {he_a.shape}")
            he_a2: CKKSTensor = self.ecg_model.forward(he_a, batch_encrypted)
            if verbose: print("\U0001F601 Sending he_a2 to the client")
            send_size1 = send_msg(sock=self.connection, msg=he_a2.serialize())
            
            if verbose: print("Backward pass --- ")
            grads, recv_size2 = recv_msg(sock=self.connection)
            grads = pickle.loads(grads)
            if verbose: print("\U0001F601 Received dJda2, dJdW from the client")
            self.ecg_model.check_update_grads(grads["dJdW"])
            dJda = self.ecg_model.backward(grads["dJda2"], 
                                           self.client_ctx)
            if verbose: print("\U0001F601 Sending dJda to the client")
            send_size2 = send_msg(sock=self.connection, msg=pickle.dumps(dJda))
            self.ecg_model.update_params(lr=lr) # updating the parameters

            # comminucation_size = recv_size1 + recv_size2 + send_size1 + send_size2
            # avg_communication += comminucation_size

            # if i == 10:
            #     print(f"avg communication for 1 forward and backward pass: "
            #           f"{avg_communication/10} (Mb)")
            #     print(f"approximated total communication: "
            #           f"{avg_communication/10 * total_batch * epoch * 1e-6} (Tb)")
            #     break


def main(hyperparams):
    # establish the connection, send the hyperparams
    server = Server()
    server.init_socket(host='localhost', port=10080)
    if hyperparams["verbose"]:
        print(f"Hyperparams: {hyperparams}")
        print("\U0001F601 Sending the hyperparameters to the Client")
    send_msg(sock=server.connection, msg=pickle.dumps(hyperparams))
    # receive the tenseal context from the client
    server.recv_ctx()
    if hyperparams["verbose"]:
        print("\U0001F601 Received the TenSeal context from the Client")
    # build and train the model
    server.build_model(project_path/'u_shaped_split_he/weights/init_weight_ptbxl.pth')
    server.train(hyperparams)
    # save the model to .pth file
    if hyperparams["save_model"]:
        torch.save(server.ecg_model.params, 
                   project_path/'u_shaped_split_he/weights/trained_server_ptbxl_2048.pth')


if __name__ == "__main__":
    print(f'project dir: {project_path}')
    hyperparams = {
        'verbose': False,
        'batch_size': 4,
        'total_batch': 4817,  # 19267 / 4
        'epoch': 10,
        'lr': 0.001,
        'seed': 0,
        'batch_encrypted': True,
        'save_model': True
    }
    main(hyperparams)