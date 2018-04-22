def displayModelSetting(layer_list):
	print('|\t\t\tLayers: {')
	for layer in layer_list:
		print('|\t\t\t\t{}'.format(layer))
	print('|\t\t\t\tlinear+softmax')
	print('|\t\t\t}')