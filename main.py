import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse, pdb, os

from util.load_data import load_mnist_data
from model.build_model import Model


def parse():
	praser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', default=128, type=int,
						help='Mini-batch size')
	parser.add_argument('--max_layers', default=5, typr=int,
						help='Max number of layers')
	parser.add_argument('--alpha', default=0.6, type=float, help='Alpha')

	args = parser.parser_args()
	return args

if __name__ == '__main__':

	args = parse()

	train_loader, valid_loader, test_loader = load_mnist_data(batch_size=args.batch_size,
															  test_batch_size=args.batch_size,
															  alpha=args.alpha)

	layer_list = ['conv_32_3_1', 'pool_2_2', 'conv_32_3_1', 'pool_2_2', 'out']
	model = Model(layer_list=layer_list, w_in=28, h_in=28, c_in=1, out_dim=10)