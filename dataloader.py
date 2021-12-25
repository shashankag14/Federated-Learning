import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

DATA_SANITY_CHECK = 0
if DATA_SANITY_CHECK:
    print(train_data, "Train Data size", train_data.data.size(), "Train target size", train_data.targets.size())
    print(test_data)
    print(len(train_dataloader), len(test_dataloader))
