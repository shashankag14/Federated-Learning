import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils
from utils import *

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),
                             ])

# Train and test datasets
train_data = datasets.MNIST(
    root = data_dir,
    train = True,
    transform = transform,
    download = True,
)
test_data = datasets.MNIST(
    root = data_dir,
    train = False,
    transform = transform
)

# Train and test dataloaders for batch wise training/testing
train_dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = args.batch_size, shuffle = True)

def prepare_client_data(train_data):
    split_len = [int(train_data.data.shape[0] / utils.args.num_clients) for _ in range(utils.args.num_clients)]
    client_data = torch.utils.data.random_split(train_data, split_len)
    client_dataloader = [torch.utils.data.DataLoader(each_client_data, batch_size = utils.args.batch_size, shuffle = True) for each_client_data in client_data]
    return client_dataloader

DATA_SANITY_CHECK = 0
if DATA_SANITY_CHECK:
    print(train_data, "Train Data size", train_data.data.size(), "Train target size", train_data.targets.size())
    print(test_data)
    print(len(train_dataloader), len(test_dataloader))
