import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse, pdb, os, copy

from src.util.load_data import load_cifar10_data
from src.util.util import displayModelSetting, save_checkpoint
from src.model.build_model import Network
from src.model.REINFORCE import Policy
from src.model.layers import LAYERS_TYPE, NUM_LAYERS_TYPE

use_cuda = torch.cuda.is_available()
print('USE CUDA: {}'.format(use_cuda))

global best_accuracy
best_accuracy = 0

def parse():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', default=128, type=int,
						help='Mini-batch size')
	parser.add_argument('--max-layers', default=15, type=int,
						help='Max number of layers')
	parser.add_argument('--alpha', default=0.7, type=float, help='Alpha')
	parser.add_argument('--gamma', default=0.8, type=float, help='Discounting factor (gamma)')
	parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
						help='Learning rate')
	parser.add_argument('--num-episodes', default=200, type=int, help='Number of episodes')
	parser.add_argument('--model-dir', default='./saved_model', type=str, help='Directory for saved models')
	parser.add_argument('--model-name', default='cifar10_', type=str, help='Model name')
	parser.add_argument('--check-memory', action='store_true', help='Flag for check memory leak')
	parser.add_argument('--save-freq', default=10, type=int, help='Save frequency')

	args = parser.parse_args()
	return args

def cifar_env(layer_list, train_loader, valid_loader, learning_rate, 
			  test_loader=None):

	net, optimizer = None, None

	try:
		net = Network(layer_list=layer_list, w_in=32, h_in=32, c_in=3, out_dim=10)
		if use_cuda:
			net.cuda()

		optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

		def train(net, optimizer, train_loader):
			net.train()

			for epoch_i in range(10):
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


		def eval(net, data_loader):
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
		valid_accuracy = eval(net, valid_loader)

		test_accuracy = None
		if test_loader:
			test_accuracy = eval(net, optimizer, test_loader)

		global best_accuracy
		if valid_accuracy > best_accuracy:
			best_accuracy = valid_accuracy
			save_checkpoint({'layer_list': layer_list, 'state_dict': net.state_dict(),
							  'optimizer': optimizer.state_dict()},
							  os.path.join(model_dir, model_name+'reinforce_best.pt'))
			del net
			del optimizer
			return valid_accuracy, test_accuracy, None
		else:
			del net
			del optimizer
			return valid_accuracy, test_accuracy, None

	except Exception as e:
		del net
		del optimizer
		print(e)
		return 0, 0, True


if __name__ == '__main__':

	args = parse()

	if os.path.isdir(args.model_dir):
		print("{} exists!".format(args.model_dir))
	else:
		os.makedirs(args.model_dir)

	train_loader, valid_loader, test_loader = load_cifar10_data(batch_size=args.batch_size,
															  test_batch_size=args.batch_size,
															  alpha=args.alpha)

	REINFORCE_policy_net = Policy(NUM_LAYERS_TYPE, 32, args.gamma)
	if use_cuda:
		REINFORCE_policy_net.cuda()
	optimizer = optim.Adam(params=REINFORCE_policy_net.parameters(), lr=args.learning_rate)

	start_state = Variable(torch.zeros(1, NUM_LAYERS_TYPE), requires_grad=False)
	if use_cuda:
		start_state = start_state.cuda()

	all_accuracy = []
	global model_dir, model_name
	model_dir = args.model_dir
	model_name = args.model_name

	for epi_i in range(1, args.num_episodes+1):
		print('|\tEpisode #{}:'.format(epi_i))

		if args.check_memory:
			pid = os.getpid()
			prev_mem = 0

		layer_list = []
		for layer_i in range(args.max_layers-1):
			print('|\t\tArchitecture #{}:'.format(layer_i+1))
			new_action = REINFORCE_policy_net.select_action(start_state)
			if new_action == NUM_LAYERS_TYPE-1:
				if layer_i == 0:
					displayModelSetting([])
					reward, _, failed = cifar_env([], train_loader, valid_loader,
														args.learning_rate)
					print('|\t\t\tValidation accuracy={:.4f}'.format(reward))
					REINFORCE_policy_net.rewards.append(reward)
				break

			layer_list.append(LAYERS_TYPE[new_action])
			displayModelSetting(layer_list)
			reward, _, failed =  cifar_env(layer_list, train_loader, valid_loader,
												 args.learning_rate)
			print('|\t\t\tValidation accuracy={:.4f}'.format(reward))
			REINFORCE_policy_net.rewards.append(reward)

			if args.check_memory:
				cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
				add_mem = cur_mem - prev_mem
				prev_mem = cur_mem
				print("|\t\tadded mem: %sM"%(add_mem))

			if failed:
				break

		optimizer.zero_grad()
		model_loss = REINFORCE_policy_net.finish_episode(0)
		model_loss.backward()
		optimizer.step()

		if (epi_i+1) % args.save_freq == 0:
			save_checkpoint({'args': args,
							 'state_dict': REINFORCE_policy_net.state_dict(),
							 'optimizer': optimizer.state_dict()},
							  os.path.join(model_dir, model_name+'reinforce_{}.pt'.format(epi_i+1)))


