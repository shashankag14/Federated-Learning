import torch.nn

import utils

criterion = torch.nn.CrossEntropyLoss().to(utils.device)

def test(model, dataloader):
	model.eval()
	running_loss, test_acc, correct_preds = 0., 0., 0

	for iter, (img, label) in enumerate(dataloader):
		img, label = img.to(utils.device), label.to(utils.device)

		pred = model(img)
		loss = criterion(pred, label)
		running_loss += loss.item()

		pred = torch.argmax(pred, dim=1)
		correct_preds += pred.eq(label.view_as(pred)).sum().item()

	test_loss = running_loss / len(dataloader)
	test_acc = correct_preds / len(dataloader)

	return test_loss, test_acc
