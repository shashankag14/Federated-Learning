import torch
import os
import argparse

# For monitoring epoch time
def compute_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--data_dir', type=str, default='data/',
                    help='Directory to store the data')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Batch size')
parser.add_argument('--epoch', type=int, default=15,
                    help='Number of epochs')

parser.add_argument('--global_epoch', type=int, default=5,
                    help='[ONLY FOR FED_LEARNING] Number of global epochs (updates to server)')
parser.add_argument('--local_epoch', type=int, default=5,
                    help='[ONLY FOR FED_LEARNING] Number of epochs for clients to train per global epoch')

parser.add_argument('--init_lr', type=float, default=5e-5,
                    help='Initial learning rate')
parser.add_argument('--enable_fed_learning', type=bool, default=1,
                    help='Enable federated learning technique for value 1')

parser.add_argument('--num_clients', type=int, default=8,
                    help='Total number of clients')
parser.add_argument('--num_select_clients', type=int, default=4,
                    help='Number of randomly selected clients for local training')

args = parser.parse_args()

data_dir = args.data_dir
if not os.path.exists(data_dir):
	os.mkdir(data_dir)

results = 'results/'
if not os.path.exists(results):
	os.mkdir(results)

baseline_results = 'results/baseline_results'
if not os.path.exists(baseline_results):
	os.mkdir(baseline_results)

fed_results = 'results/fed_results'
if not os.path.exists(fed_results):
	os.mkdir(fed_results)