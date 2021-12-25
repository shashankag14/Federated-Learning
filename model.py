import torch
import torch.nn as nn

class Model(nn.Module) :
	def __init__(self, in_channel=1, num_classes=10): #default values for MNIST
		super(Model,self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channel, 16, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		# Due to Maxpooling : 28 / 2 -> 14 / 2 -> 7
		self.fc = nn.Linear(32 * 7 * 7, num_classes)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		# reshaping from a 4d tensor to 1d
		x = x.reshape(x.shape[0], -1) # keep minibatch intact @ pos = 0 and reshape other pos
		x = self.fc(x)
		return x



MODEL_SANITY_CHECK = 0
if MODEL_SANITY_CHECK :
	x = torch.randn([16, 1, 28, 28])
	model = Model(1, 10)
	output = model(x)
	print(output.shape)
