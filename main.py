import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse, pdb, os, copy

from util.load_data import load_mnist_data
from model.build_model import Network
from model.REINFORCE import Policy
from model.layers import LAYERS_TYPE, NUM_LAYERS_TYPE

def parse():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', default=128, type=int,
						help='Mini-batch size')
	parser.add_argument('--max_layers', default=5, type=int,
						help='Max number of layers')
	parser.add_argument('--alpha', default=0.6, type=float, help='Alpha')
	parser.add_argument('--gamma', default=0.99, type=float, help='Discounting factor (gamma)')
	parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
						help='Learning rate')

	args = parser.parse_args()
	return args

def train_mnist_network(layer_lists):

	for layer_list in layer_lists:
		net = Network(layer_list=layer_list, w_in=28, h_in=28, c_in=1, out_dim=10)
		optimizer = optim.Adam(params=REINFORCE_policy_net.parameters(), lr=args.learning_rate)

		del net
		del optimizer


if __name__ == '__main__':

	args = parse()

	train_loader, valid_loader, test_loader = load_mnist_data(batch_size=args.batch_size,
															  test_batch_size=args.batch_size,
															  alpha=args.alpha)

	#layer_list = ['conv_32_3_1', 'pool_2_2', 'conv_32_3_1', 'pool_2_2', 'out']
	#net = Network(layer_list=layer_list, w_in=28, h_in=28, c_in=1, out_dim=10)

	'''
	net.eval()

	for batch_idx, (data, target) in enumerate(train_loader):

		data, target = Variable(data, requires_grad=False), \
					   Variable(target, requires_grad=False)

		pred = net(data)

		break
	'''

	REINFORCE_policy_net = Policy(NUM_LAYERS_TYPE, 64, args.gamma)
	optimizer = optim.Adam(params=REINFORCE_policy_net.parameters(), lr=args.learning_rate)

	start_state = Variable(torch.zeros(1, 16), requires_grad=False)
	layer_list = []
	layer_lists = [[]]
	for layer_i in range(args.max_layers-1):
		new_action = REINFORCE_policy_net.select_action(start_state)
		print(new_action)
		if new_action == 15:
			break
		layer_list.append(LAYERS_TYPE[new_action])
		layer_lists.append(copy.deepcopy(layer_list))

