# load and train model

import torch
import torch.nn as nn

#######This is the custom loss function defined for VAE
### You can even refere to: https://github.com/pytorch/examples/pull/226 

def VAE_loss(out, target, mean, logvar):
	category1 = nn.BCELoss()
	bce_loss = category1(out, target)
	
	#We will scale the following losses with this factor
	scaling_factor = out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]
	
	####Now we are gonna define the KL divergence loss
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	kl_loss = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
	kl_loss /= scaling_factor
	
	return bce_loss + kl_loss

######The function which we will call for training our model

def _train(trainloader, iters, model, device, optimizer, print_every):
	counter = 0
	for i in range(iters):
		model.train()
		model.to(device)
		for images in trainloader:
			images = images.to(device)
			optimizer.zero_grad()
			out, mean, logvar = model(images)
			loss = VAE_loss(out, images, mean, logvar)
			loss.backward()
			optimizer.step()
			
		if(counter % print_every == 0):
			yield loss.numpy().sum()

		counter += 1

import matplotlib.pyplot as plt
lossLst = []
######Setting all the hyperparameters
def train_model(model, trainloader,
				iters=26, num_latent=8,
				print_every=5):
    #print after every 5 iterations
	# model = VAE(num_latent, state_size)
	global lossLst

	device = ('cuda' if torch.cuda.is_available() else 'cpu')
	import torch.optim as optim
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	for lossShow in _train(trainloader, iters, model, device, optimizer, print_every):
		lossLst.append(lossShow)
		plt.plot(lossLst, 'b-')
		plt.ylabel('loss')
	plt.show()

