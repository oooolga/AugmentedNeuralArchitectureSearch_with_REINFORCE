import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
import torch.nn.functional as F

LAYERS = {
		  'conv_32_3_1': {'input_channel':None, 'output_channel':32, 'kernel_size':3, 'stride':1},
		  'conv_64_3_1': {'input_channel':None, 'output_channel':64, 'kernel_size':3, 'stride':1},
		  'conv_128_3_1': {'input_channel':None, 'output_channel':128, 'kernel_size':3, 'stride':1},
		  'conv_32_4_1': {'input_channel':None, 'output_channel':32, 'kernel_size':4, 'stride':1},
		  'conv_64_4_1': {'input_channel':None, 'output_channel':64, 'kernel_size':4, 'stride':1},
		  'conv_128_4_1': {'input_channel':None, 'output_channel':128, 'kernel_size':4, 'stride':1},
		  'conv_48_5_1': {'input_channel':None, 'output_channel':48, 'kernel_size':5, 'stride':1},
		  'conv_64_5_1': {'input_channel':None, 'output_channel':64, 'kernel_size':5, 'stride':1},
		  'conv_128_5_1': {'input_channel':None, 'output_channel':128, 'kernel_size':5, 'stride':1},
		  'pool_2_1': {'kernel_size':2, 'stride':1},
		  'pool_2_2': {'kernel_size':2, 'stride':2},
		  'pool_3_1': {'kernel_size':3, 'stride':1},
		  'pool_3_2': {'kernel_size':3, 'stride':2},
		  'pool_5_3': {'kernel_size':5, 'stride':3},
		  'dropout_0.25': {'drop_propt': 0.25},
		  'dropout_0.5': {'drop_propt': 0.5},
		  'dropout_0.75': {'drop_propt': 0.75},
		  'res_32_1': {'output_channel': 32, 'stride': 1},
		  'res_64_1': {'output_channel': 64, 'stride': 1},
		  'res_128_1': {'output_channel': 128, 'stride': 1},
		  'res_32_2': {'output_channel': 32, 'stride': 2},
		  'res_64_2': {'output_channel': 64, 'stride': 2},
		  'res_128_2': {'output_channel': 128, 'stride': 2},
		  'z_out': {}
		  }
LAYERS_TYPE = list(LAYERS.keys())
LAYERS_TYPE.sort()
NUM_LAYERS_TYPE = len(LAYERS_TYPE)

class ConvLayer(nn.Module):

	def __init__(self, input_channel, output_channel, kernel_size, stride):
		super(ConvLayer, self).__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(input_channel, output_channel, kernel_size, stride),
			nn.BatchNorm2d(output_channel),
			nn.ReLU(inplace=True))

		self.in_chan = input_channel
		self.out_chan = output_channel
		self.kernel_size = kernel_size
		self.stride = stride

	def forward(self, x):
		return self.layer(x)

	def get_output_size(self, h_in, w_in):
		h_out = int((h_in-self.kernel_size)/self.stride+1)
		w_out = int((w_in-self.kernel_size)/self.stride+1)
		return h_out, w_out


class DropoutLayer(nn.Module):

	def __init__(self, drop_ratio=0.75):
		super(DropoutLayer, self).__init__()
		self.layer = nn.Sequential(
			nn.Dropout(drop_ratio)
			)
		self.drop_ratio = drop_ratio

	def forward(self, x):
		return self.layer(x)

	def get_output_size(self, h_in, w_in):
		return h_in, w_in

class PoolLayer(nn.Module):

	def __init__(self, kernel_size, stride):
		super(PoolLayer, self).__init__()
		self.layer = nn.Sequential(
			nn.MaxPool2d(kernel_size, stride)
			)
		self.stride = stride
		self.kernel_size = kernel_size

	def forward(self, x):
		return self.layer(x)

	def get_output_size(self, h_in, w_in):
		h_out = int((h_in-self.kernel_size)/self.stride+1)
		w_out = int((w_in-self.kernel_size)/self.stride+1)
		return h_out, w_out

class OutputLayer(nn.Module):

	def __init__(self, input_dim, out_dim=10):
		super(OutputLayer, self).__init__()
		self.layer = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, out_dim)
			)

	def forward(self, x):
		return F.log_softmax(self.layer(x), dim=1)

# Residual Block
# ref: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
					 stride=stride, padding=1, bias=False)

class ResidualLayer(nn.Module):
	def __init__(self, input_channel, output_channel, stride=1):
		super(ResidualLayer, self).__init__()
		self.conv1 = conv3x3(input_channel, output_channel, stride)
		self.bn1 = nn.BatchNorm2d(output_channel)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(output_channel, output_channel)
		self.bn2 = nn.BatchNorm2d(output_channel)
		self.downsample = None

		if (stride != 1) or (input_channel != output_channel):
			self.downsample = nn.Sequential(
				conv3x3(input_channel, output_channel, stride=stride),
				nn.BatchNorm2d(output_channel))
		self.stride = stride
		
	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		return out

	def get_output_size(self, h_in, w_in):
		h_out = int((h_in-1)/self.stride+1)
		w_out = int((w_in-1)/self.stride+1)
		return h_out, w_out