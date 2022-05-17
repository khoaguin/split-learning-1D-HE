# Split Learning HE
> TL;DR Split learning of 1D CNN models combined with homomorphic encryption on ECG datasets

### Requirements
torch==1.10.0+cu102  
tenseal==0.3.10

### Source Code
---

#### Repository Structure

* `data/`  
    * `train_ecg.hdf5` - the processed training split from the [MIT-DB](https://physionet.org/content/mitdb/1.0.0/) dataset
    * `test_ecg.hdf5` - the processed testing split from the [MIT-DB](https://physionet.org/content/mitdb/1.0.0/) dataset
    * `ptbxl_processing.ipynb` - code needed to process the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) dataset. Running the code will output `train_ptbxl.hdf5` and `test_ptbxl.hdf5`
* `local_plaintext/`
    * `train.ipynb` - code to train the 1D CNN locally on the MIT-DB dataset
    * `visual_invertibility.ipynb` - code to demonstrate the privacy leakage of the activation maps produced by the convolutional layers on the MIT-DB dataset
    * `train_ptbxl.ipynb` - code to train the 1D CNN locally on the PTB-XL dataset
    * `visual_invertibility_ptbxl.ipynb` - similar to `visual_invertibility.ipynb` but for the PTB-XL dataset  
* `local_plaintext_big` 
    * `train.ipynb` - code to train the 1D CNN model on the MIT-DB dataset but with bigger activation maps. 
    * `visual_invertibility.ipynb` - demonstrate the privacy leakage
* `u_shaped_split_he`
    * `client.py` and `server.py`: code for the client side and server side to train the split learning protocol using homomorphically encrypted activation maps on the MIT-DB dataset
    * `client_ptbxl.py` and `server_ptbxl.py`: similarly, but for the PTB-XL dataset
* `u_shaped_split_he_big`
    * `client.py` and `server.py`: code to train the split 1D CNN using HE with bigger activation maps size, only for the MIT-DB dataset
* `u_shaped_split_plaintext`
    * `client.py` and `server.py`: code for the client side and server side to train the split learning protocol on plaintext activation maps for the MIT-DB dataset
    * `client_ptbxl.py` and `server_ptbxl.py`: similarly, but for the PTB-XL dataset
* `u_shaped_split_plaintext_big`
    * `client.py` and `server.py`: code for the client and the server to train the split learning protocol on plaintext activation maps with bigger size, only for the MIT-DB dataset

#### Running the code

To run the code, simply `cd` into the directory and run the code for server side and client side. Note that you need to run the code for server side first. For example, if you want to run the u-shaped split learning using HE for the PTB-XL dataset, do the following:  
```
cd u_shaped_split_he
python server_ptbxl.py
```
Then, open a new tab and run 
```
python client_ptbxl.py
```
