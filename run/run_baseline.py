import os
import torch.nn
from tqdm import tqdm
import time

import model
import dataloader
import utils
import plot
import test

print("RUNNING BASELINE (w/o Federated Learning)")

device = utils.device
print("Device being used : {}".format(device))

model = model.Model()
criterion = torch.nn.CrossEntropyLoss().to(device)
opt = torch.optim.Adam(model.parameters(), lr = utils.args.init_lr)

def train(num_epochs, model, dataloader):
	model.train()
	running_loss = 0
	epoch_losses = []

	for epoch in range(1, num_epochs+1):
		start_time = time.time()
		with tqdm(dataloader, unit="Batch") as tepoch:
			for iter, (img, label) in enumerate(dataloader) :
				tepoch.set_description(f"Epoch {epoch}")

				img, labels = img.to(device), label.to(device)

				pred = model(img)
				loss = criterion(pred, label)
				opt.zero_grad()
				loss.backward()
				opt.step()

				running_loss += loss.item()

				tepoch.update()
				tepoch.set_postfix(loss=loss.item())

		current_epoch_loss = running_loss/len(dataloader)
		epoch_losses.append(current_epoch_loss)

		end_time = time.time()
		epoch_mins, epoch_secs = utils.compute_time(start_time, end_time)

		f = open(os.path.join(utils.baseline_results, "baseline_train_loss.txt"), 'w')
		f.write(str(current_epoch_loss))
		f.close()

		print(f'Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {current_epoch_loss:.3f}')

	return epoch_losses


if __name__ == '__main__':
	epoch_losses = train(utils.args.epoch, model, dataloader.train_dataloader)
	plot.plot_loss(epoch_losses)

	# test_loss, test_acc = test(model, dataloader.test_dataloader)
	# print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:.2f}')
