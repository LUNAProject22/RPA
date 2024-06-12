import torch


a = torch.load('epoch_0.pt')
b = torch.load('epoch_1.pt')

keys = a['state_dict'].keys()

for key in keys:
	c = a['state_dict'][key] - b['state_dict'][key]
	if torch.sum(c) != 0:
		print(key)
