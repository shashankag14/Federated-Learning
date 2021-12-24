from torchvision import datasets
import torchvision.transforms as transforms

import utils

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),
                             ])

train_data = datasets.MNIST(
    root = utils.data,
    train = True,
    transform = transform,
    download = True,
)
test_data = datasets.MNIST(
    root = utils.data,
    train = False,
    transform = transform
)

DATA_SANITY_CHECK = 1
if DATA_SANITY_CHECK:
    print(train_data, "Train Data size", train_data.data.size(), "Train target size", train_data.targets.size())
    print(test_data)
