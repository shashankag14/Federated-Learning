import os
import torch.nn
from tqdm import tqdm
import time
import numpy

import model
import dataloader
import utils
import plot
import test

print("RUNNING FEDEARTED LEARNING")

device = utils.device
print("Device being used : {}".format(device))

server_model = model.Model().to(device)
client_models = [model.Model().to(device) for client in range(utils.args.num_clients)]
for model in client_models:
    model.load_state_dict(server_model.state_dict())

client_opt = [torch.optim.Adam(model.parameters(), lr = utils.args.init_lr) for model in client_models]
criterion = torch.nn.CrossEntropyLoss().to(device)
client_dataloader = dataloader.prepare_client_data(dataloader.train_data)

def train_locally(client_model, optimizer, client_train_loader, local_epoch, selected_client):
    client_model.train()
    client_loss = 0

    for local_epoch_num in range(1, local_epoch+1):
        running_loss = 0

        with tqdm(client_train_loader, unit="Batch") as tepoch:
            for batch_idx, (data, target) in enumerate(client_train_loader):
                tepoch.set_description(f"\tClient-ID{selected_client} Local Epoch {local_epoch_num}")

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                tepoch.update()
                tepoch.set_postfix(loss=loss.item())

            per_epoch_loss = running_loss / len(client_train_loader)
            client_loss += per_epoch_loss
                
    print(f'\tClient-ID{selected_client} Average Train Loss {client_loss / local_epoch:.3f}\n')
    return client_loss / local_epoch

def update_server(server_model, client_models):
    server_dict = server_model.state_dict()
    for k in server_dict.keys():
        server_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    server_model.load_state_dict(server_dict)
    for model in client_models:
        model.load_state_dict(server_model.state_dict())

def train(global_epoch, local_epoch, server_model, client_models, client_opt, client_dataloader):
    selected_client_list = numpy.random.permutation(utils.args.num_clients)[:utils.args.num_select_clients]
    
    for epoch in range(1, global_epoch+1):
        print("-"*50)
        print(f'Global Epoch: {epoch}\n')
        start_time = time.time()
        total_client_loss = 0
 
        # Train each client locally and then aggregate their parameters in server
        for selected_client in range(utils.args.num_select_clients):
            total_client_loss += train_locally(client_models[selected_client],
                                               client_opt[selected_client],
                                               client_dataloader[selected_client_list[selected_client]],
                                               local_epoch,
                                               selected_client_list[selected_client])

        current_epoch_loss = total_client_loss / utils.args.num_select_clients

        # Update server with avg of client parameters
        update_server(server_model, client_models)

        end_time = time.time()
        epoch_mins, epoch_secs = utils.compute_time(start_time, end_time)

        f = open(os.path.join(utils.fed_results, "fed_train_loss.txt"), 'w')
        f.write(str(current_epoch_loss))
        f.close()

        print(f'Global Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tGlobal Average Train Loss: {current_epoch_loss:.3f}\n')


if __name__ == '__main__':
    start_time = time.time()
    train(utils.args.global_epoch, utils.args.local_epoch, server_model,
                         client_models, client_opt, client_dataloader)
    end_time = time.time()
    total_min, total_secs = utils.compute_time(start_time, end_time)
    print("Total training time : {total_min}m {total_secs}s'")

    test_loss, test_acc = test.test(server_model, dataloader.test_dataloader)
 
    print('-'*50)
    print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:.2f}')

