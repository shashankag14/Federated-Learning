# Federated Learning - Your Data Stays With You !
A PyTorch implementation of Federated Learning from scratch partially based on the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). Ithas been implemented using [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

<img src="https://user-images.githubusercontent.com/74488693/147395776-4930a16b-ef23-44a7-9f58-6f65a2deb208.png" height="250" width="500">

Federated learning (FL) is an approach that downloads the current model and computes an updated model at the device itself (ala edge computing) using local data. These locally trained models are then sent from the devices back to the central server where they are aggregated, i.e. averaging weights, and then a single consolidated and improved global model is sent back to the devices.

# Getting started
To install the required libraries, run the following script :
> sh [requirements.sh](requirements.sh)

Run the following command to train using Federated Learning :
```
python3 run_federated.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                        [--epoch EPOCH] [--global_epoch GLOBAL_EPOCH]
                        [--local_epoch LOCAL_EPOCH] [--init_lr INIT_LR]
                        [--num_clients NUM_CLIENTS]
                        [--num_select_clients NUM_SELECT_CLIENTS]

```
Run the following command to train without Federated Learning (for reference):
```
python3 run_baseline.py [-h] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
                        [--epoch EPOCH] [--global_epoch GLOBAL_EPOCH]
                        [--local_epoch LOCAL_EPOCH] [--init_lr INIT_LR]
                        [--num_clients NUM_CLIENTS]
                        [--num_select_clients NUM_SELECT_CLIENTS]

```

# Hyperparams 

| Parameters | Description | Value used |
| --- | --- | --- |
| `--epoch` | Number of epochs for baseline training| 15 |
| `--batch_size` | Batch size | 100 | 
| `--global_epoch` | [ONLY FOR FED_LEARNING] Number of global epochs (updates to server) | 5 |
| `--local_epoch` | [ONLY FOR FED_LEARNING] Number of epochs for clients to train per global epoch | 5 |
| `--init_lr` | Initial learning rate | 5e-5 |
| `--num_clients` | Total number of clients | 8 |
| `--num_select_clients` | Number of randomly selected clients for local training | 4 |

# Results of Federated Learning
* Test Accuracy = 98.5%
* Test Loss = 0.048 




