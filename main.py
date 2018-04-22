import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse, pdb, os, copy

from util.load_data import load_mnist_data
from util.util import displayModelSetting
from model.build_model import Network
from model.REINFORCE import Policy
from model.layers import LAYERS_TYPE, NUM_LAYERS_TYPE

use_cuda = torch.cuda.is_available()

def parse():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', default=128, type=int,
						help='Mini-batch size')
	parser.add_argument('--max-layers', default=4, type=int,
						help='Max number of layers')
	parser.add_argument('--alpha', default=0.6, type=float, help='Alpha')
	parser.add_argument('--gamma', default=0.99, type=float, help='Discounting factor (gamma)')
	parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
						help='Learning rate')
	parser.add_argument('--num-episodes', default=100, type=int, help='Number of episodes')

	args = parser.parse_args()
	return args

def mnist_env(layer_list, train_loader, valid_loader, learning_rate, 
			  test_loader=None, save_model=False):

	net = Network(layer_list=layer_list, w_in=28, h_in=28, c_in=1, out_dim=10)
	if use_cuda:
		net.cuda()

	optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

	def train(net, optimizer, train_loader):
		net.train()

		for epoch_i in range(1):
			print('|\t\t\tTrain epoch {}:'.format(epoch_i+1))
			total_loss = 0
			total_batch = 0
			for batch_idx, (data, target) in enumerate(train_loader):
				if use_cuda:
					data, target = data.cuda(), target.cuda()


				data, target = Variable(data, requires_grad=False), \
						   Variable(target, requires_grad=False)
				optimizer.zero_grad()
				pred = net(data)
				loss = net.loss(pred, target)
				total_loss += loss
				loss.backward()
				optimizer.step()
				total_batch += 1

			print('|\t\t\t\tAverage loss={:.4f}'.format(total_loss/float(total_batch)))


	def eval(net, optimizer, data_loader):
		net.eval()

		total_loss, correct = 0, 0
		total_data, total_batch = 0, 0

		for batch_idx, (data, target) in enumerate(data_loader):
			if use_cuda:
				data, target = data.cuda(), target.cuda()

			data, target = Variable(data, requires_grad=False), \
					   Variable(target, requires_grad=False)

			with torch.no_grad(): # prevent memory leak
				pred = net(data)
				_, predicted = torch.max(pred.data, 1)
				correct += (predicted == target.data).sum()
				total_data += len(data)

		accuracy = correct.cpu().numpy() / float(total_data)

		return accuracy

	train(net, optimizer, train_loader)
	valid_accuracy = eval(net, optimizer, valid_loader)

	test_accuracy = None
	if test_loader:
		test_accuracy = eval(net, optimizer, test_loader)

	if save_model:
		return valid_accuracy, test_accuracy, net, optimizer
	else:
		del net
		del optimizer
		return valid_accuracy, test_accuracy


if __name__ == '__main__':

	args = parse()

	train_loader, valid_loader, test_loader = load_mnist_data(batch_size=args.batch_size,
															  test_batch_size=args.batch_size,
															  alpha=args.alpha)

	REINFORCE_policy_net = Policy(NUM_LAYERS_TYPE, 64, args.gamma)
	if use_cuda:
		REINFORCE_policy_net.cuda()
	optimizer = optim.Adam(params=REINFORCE_policy_net.parameters(), lr=args.learning_rate)

	start_state = Variable(torch.zeros(1, NUM_LAYERS_TYPE), requires_grad=False)
	start_state = start_state.cuda()

	for epi_i in range(1, args.num_episodes+1):
		print('|\tEpisode #{}:'.format(epi_i))
		layer_list = []
		for layer_i in range(args.max_layers-1):
			print('|\t\tArchitecture #{}:'.format(layer_i+1))
			new_action = REINFORCE_policy_net.select_action(start_state)
			if new_action == NUM_LAYERS_TYPE-1:
				if layer_i == 0:
					displayModelSetting([])
					reward, _ = mnist_env([], train_loader, valid_loader, args.learning_rate)
					print('|\t\t\tValidation accuracy={:.4f}'.format(reward))
					REINFORCE_policy_net.rewards.append(reward)
				break

			layer_list.append(LAYERS_TYPE[new_action])
			displayModelSetting(layer_list)
			reward, _ = mnist_env(layer_list, train_loader, valid_loader, args.learning_rate)
			print('|\t\t\tValidation accuracy={:.4f}'.format(reward))
			REINFORCE_policy_net.rewards.append(reward)

		optimizer.zero_grad()
		model_loss = REINFORCE_policy_net.finish_episode(0)
		model_loss.backward()
		optimizer.step()


