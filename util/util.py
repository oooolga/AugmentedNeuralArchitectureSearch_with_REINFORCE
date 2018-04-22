import torch

def displayModelSetting(layer_list):
	print('|\t\t\tLayers: {')
	for layer in layer_list:
		print('|\t\t\t\t{}'.format(layer))
	print('|\t\t\t\tlinear+softmax')
	print('|\t\t\t}')

def save_checkpoint(state, model_name):
	torch.save(state, model_name)
	print('Finished saving model: {}'.format(model_name))

def load_model():
	pass