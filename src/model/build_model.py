import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
import torch.nn.functional as F
from .layers import *

class Network(nn.Module):

	def __init__(self, layer_list, w_in, h_in, c_in, out_dim):
		super(Network, self).__init__()
		self.layer_list = layer_list
		self.out_dim = out_dim
		self.build_model(w_in, h_in, c_in)

	def build_model(self, w_in, h_in, c_in):

		layers = []
		for layer_i in self.layer_list:
			if 'conv' in layer_i:
				layers.append(ConvLayer(input_channel=c_in,
											 output_channel=LAYERS[layer_i]['output_channel'],
											 kernel_size=LAYERS[layer_i]['kernel_size'],
											 stride=LAYERS[layer_i]['stride']))
				w_in, h_in = layers[-1].get_output_size(w_in, h_in)
				c_in = LAYERS[layer_i]['output_channel']

			elif 'pool' in layer_i:
				layers.append(PoolLayer(kernel_size=LAYERS[layer_i]['kernel_size'],
											 stride=LAYERS[layer_i]['stride']))
				w_in, h_in = layers[-1].get_output_size(w_in, h_in)

			elif 'drop' in layer_i:
				layers.append(DropoutLayer(drop_ratio=LAYERS[layer_i]['drop_propt']))

			elif 'res' in layer_i:
				layers.append(ResidualLayer(input_channel=c_in, output_channel=LAYERS[layer_i]['output_channel'],
											stride=LAYERS[layer_i]['stride']))
				c_in = LAYERS[layer_i]['output_channel']
				w_in, h_in = layers[-1].get_output_size(w_in, h_in)

		self.net = nn.Sequential(*layers)
		in_dim = c_in*w_in*h_in
		self.out_in_dim = in_dim
		self.out_layer = OutputLayer(in_dim, self.out_dim)

	def forward(self, x):
		x = self.net(x)
		x = x.view(-1, self.out_in_dim)
		return self.out_layer(x)

	def loss(self, out, target):
		return F.nll_loss(out, target)