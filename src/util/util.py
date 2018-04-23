import torch
import torch.optim as optim

import os
from ..model.build_model import Network

def displayModelSetting(layer_list):
	print('|\t\t\tLayers: {')
	for layer in layer_list:
		print('|\t\t\t\t{}'.format(layer))
	print('|\t\t\t\tlinear+softmax')
	print('|\t\t\t}')

def save_checkpoint(state, model_name):
	torch.save(state, model_name)
	print('Finished saving model: {}'.format(model_name))

def load_checkpoint(model_name, w_in=32, h_in=32, c_in=3, out_dim=10):
	if model_name and os.path.isfile(model_name):
		checkpoint = torch.load(model_name)
		layer_list = checkpoint['layer_list']

		net = Network(layer_list=layer_list,
					  w_in=w_in, h_in=h_in, c_in=c_in, out_dim=out_dim)

		if torch.cuda.is_available():
			net = net.cuda()
		optimizer = optim.Adam(params=net.parameters(), lr=1e-4)

		net.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

		print('Finished loading model and optimizer from {}'.format(model_name))

	else:
		print('File {} not found.'.format(model_name))
		raise FileNotFoundError
	return layer_list, net, optimizer
