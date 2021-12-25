import torch
import os
import matplotlib.pyplot as plt

import utils

def show_random_img(data):
	figure = plt.figure(figsize=(10, 8))
	cols, rows = 5, 5
	for i in range(1, cols * rows + 1):
		sample_idx = torch.randint(len(data), size=(1,)).item()
		img, label = data[sample_idx]
		figure.add_subplot(rows, cols, i)
		plt.title(label)
		plt.axis("off")
		plt.imshow(img.squeeze(), cmap="gray")
	plt.show()

def plot_loss(epoch_losses):
	plt.figure()
	plt.plot(range(len(epoch_losses)), epoch_losses)
	plt.xlabel('Epochs')
	plt.ylabel('Train loss')
	plt.legend()
	plt.savefig(os.path.join(utils.baseline_results, "baseline_plot.png"))
	print("Trian loss plot has been saved in {}".format(os.path.join(utils.baseline_results, "baseline_plot.png")))
