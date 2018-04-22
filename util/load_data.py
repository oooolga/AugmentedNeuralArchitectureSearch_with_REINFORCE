import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

def load_mnist_data(batch_size, test_batch_size, alpha=1):

	train_data = dset.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
	train_data.train_data = train_data.train_data[:50000]
	train_data.train_labels = train_data.train_labels[:50000]
	valid_data = dset.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
	valid_data.train_data = valid_data.train_data[50000:]
	valid_data.train_labels = valid_data.train_labels[50000:]

	train_data_len = len(train_data)
	train_data.train_data = train_data.train_data[:int(alpha*train_data_len)]
	train_data.train_labels = train_data.train_labels[:int(alpha*train_data_len)]

	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size, shuffle=True, drop_last=True)
	valid_loader = torch.utils.data.DataLoader(
		valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
		dset.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
		batch_size=test_batch_size, shuffle=True, drop_last=True)

	return train_loader, valid_loader, test_loader

def load_pmnist_data(batch_size, test_batch_size, alpha=1):

	train_data = dset.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
	train_data.train_data = train_data.train_data[:50000]
	train_data.train_labels = train_data.train_labels[:50000]
	valid_data = dset.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
	valid_data.train_data = valid_data.train_data[50000:]
	valid_data.train_labels = valid_data.train_labels[50000:]

	train_data_len = len(train_data)
	train_data.train_data = train_data.train_data[:int(alpha*train_data_len)]
	train_data.train_labels = train_data.train_labels[:int(alpha*train_data_len)]

	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size, shuffle=True, drop_last=True)
	valid_loader = torch.utils.data.DataLoader(
		valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
		dset.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
		batch_size=test_batch_size, shuffle=True, drop_last=True)

	permutation_list = Variable(torch.LongTensor(np.random.permutation(784)), volatile=True)

	return train_loader, valid_loader, test_loader, permutation_list