import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
import torch.nn.functional as F
from layers import *

class Model(nn.Module):

	def __init__(self, layer_list, w_in, h_in, c_in, out_dim):
		self.layer_list = layer_list
		self.out_dim = out_dim
		self.build_model(w_in, h_in)

	def build_model(self, w_in, h_in, c_in):
		self.layers = []
		for layer_i in layer_list:
			if 'conv' in layer_i:
				self.layers.append(ConvLayer(input_channel=c_in,
											 output_channel=LAYERS[layer_i]['output_channel'],
											 kernel_size=LAYERS[layer_i]['kernel_size'],
											 stride=LAYERS[layer_i]['stride']))
				w_in, h_in = self.layers[-1].get_output_size(w_in, h_in)
				c_in = LAYERS[layer_i]['output_channel']

			elif 'pool' in layer_i:
				self.layers.append(PoolLayer(kernel_size=LAYERS[layer_i]['kernel_size'],
											 stride=LAYERS[layer_i]['stride']))
				w_in, h_in = self.layers[-1].get_output_size(w_in, h_in)

			else:
				in_dim = c_in*w_in*h_in
				self.layers.append(OutputLayer(in_dim, self.out_dim))
				break

	def forward(self, x):
		for layer_i in range(len(self.layers)-1):
			x = self.layers[layer_i](x)

		return self.layers[-1](x)