import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse, pdb, os, copy

from src.util.load_data import load_mnist_data
from src.util.util import displayModelSetting, save_checkpoint, load_checkpoint
from src.model.build_model import Network
from src.model.REINFORCE import Policy
from src.model.layers import LAYERS_TYPE, NUM_LAYERS_TYPE

use_cuda = torch.cuda.is_available()
print('USE CUDA: {}'.format(use_cuda))

global best_accuracy
best_accuracy = 0
model_dir, model_name = None, None

def parse():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', default=128, type=int,
						help='Mini-batch size')
	parser.add_argument('--max-layers', default=15, type=int,
						help='Max number of layers')
	parser.add_argument('--alpha', default=0.7, type=float, help='Alpha')
	parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
						help='Learning rate')
	parser.add_argument('--num-epochs', default=10, type=int, help='Number of epochs')
	parser.add_argument('--model-dir', default='./saved_model', type=str, help='Directory for saved models')
	parser.add_argument('--model-name', default='mnist_final', type=str, help='Model name')
	parser.add_argument('--load-path', default='./saved_model/mnist_reinforce_best.pt', type=str)

	args = parser.parse_args()
	return args

def train_mnist(net, optimizer, train_loader, valid_loader, test_loader=None, num_epochs=200):

	def train(net, optimizer, train_loader, num_epochs):
		net.train()

		for epoch_i in range(num_epochs):
			print('|\t\t\tTrain epoch {}:'.format(epoch_i+1))
			total_loss = 0
			total_batch = 0
			for batch_idx, (data, target) in enumerate(train_loader):
				if use_cuda:
					data, target = data.cuda(async=True), target.cuda(async=True)

				data, target = Variable(data, requires_grad=False), \
						   Variable(target, requires_grad=False)
				optimizer.zero_grad()
				pred = net(data)
				loss = net.loss(pred, target)
				total_loss += loss.item()
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

			pred = net(data)
			_, predicted = torch.max(pred.data, 1)
			correct += (predicted == target.data).sum().item()
			total_data += len(data)

		accuracy = correct / float(total_data)

		return accuracy

	train(net, optimizer, train_loader, num_epochs)
	valid_accuracy = eval(net, valid_loader)

	test_accuracy = None
	if test_loader:
		test_accuracy = eval(net, test_loader)

	global best_accuracy
	if valid_accuracy > best_accuracy:
		best_accuracy = valid_accuracy
		save_checkpoint({'state_dict': net.state_dict(),
						 'optimizer': optimizer.state_dict()},
						  os.path.join(model_dir, model_name+'.pt'))
		return valid_accuracy, test_accuracy
	else:
		del net
		del optimizer
		return valid_accuracy, test_accuracy


if __name__ == '__main__':
	args = parse()
	layer_list, net, optimizer = load_checkpoint(args.load_path, w_in=28, h_in=28, c_in=1, out_dim=10)


	train_loader, valid_loader, test_loader = load_mnist_data(batch_size=args.batch_size,
															  test_batch_size=args.batch_size,
															  alpha=args.alpha)

	model_dir = args.model_dir
	model_name = args.model_name

	displayModelSetting(layer_list)
	valid_accuracy, test_accuracy = train_mnist(net, optimizer, train_loader, valid_loader, test_loader, num_epochs=args.num_epochs)
	print('Validation accuracy={:.4f}\tTest accuracy={:.4f}'.format(valid_accuracy, test_accuracy))
