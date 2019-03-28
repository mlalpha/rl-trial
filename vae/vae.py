# choose model1 or model2
# train or use

from dataloader import dataloader
import model1
import torch
import torch.nn as nn

class vae_module(object):
	"""docstring for vae_module"""
	def __init__(self, num_latent, state_size, img_trans=None,
				dataset_folder="videos", dataset_format="mp4",
				filter_size=[3, 3, 3, 3], channels=[1, 4, 20, 128, 128],
				dconv_kernel_sizes=[3, 11, 23, 46], train=True):
		super(vae_module, self).__init__()
		# init trainloader
		if train:
			trainset = dataloader(dataset_folder,
										dataset_format, img_trans)
			self.trainloader = torch.utils.data.DataLoader(trainset,
									batch_size=100, shuffle=True)
		self.model = model1.VAE(num_latent,
								state_size, filter_size,
								channels, dconv_kernel_sizes)

	def train(self, iters=26, print_every=5, print_func=None):
	    #print after every 5 iterations

		device = ('cuda' if torch.cuda.is_available() else 'cpu')
		import torch.optim as optim
		optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

		self._train(iters, device, optimizer, print_every, print_func)

	######The function which we will call for training our model

	def _train(self, iters, device, optimizer, print_every, print_f=None):
		counter = 0
		for i in range(iters):
			self.model.train()
			self.model.to(device)
			for images in self.trainloader:
				images = images.to(device)
				optimizer.zero_grad()
				out, mean, logvar = self.model(images)
				loss = self.VAE_loss(out, images, mean, logvar)
				loss.backward()
				optimizer.step()

			if(counter % print_every == 0):
				self.model.eval()
				if print_f:
					print_f(loss.data.cpu().sum().numpy())
				else:
					print("loss.sum(): ", loss.data.cpu().sum().numpy())

			counter += 1

	def VAE_loss(self, out, target, mean, logvar):
		category1 = nn.BCELoss()
		bce_loss = category1(out, target)
		
		#We will scale the following losses with this factor
		scaling_factor = out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]
		
		####Now we are gonna define the KL divergence loss
		# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		kl_loss = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
		kl_loss /= scaling_factor

		return bce_loss + kl_loss

	def save(self, path="vae.pt"):
		torch.save(self.model.state_dict(), path)

	def load(self, path="vae.pt"):
		if torch.cuda.is_available():
			self.model.load_state_dict(torch.load(path))
		else:
			self.model.load_state_dict(torch.load(path, map_location='cpu'))
		self.model.eval()

	def encode(self, data):
		m, l = self.model.enc_func(data)
		return self.model.get_hidden(m, l)

	def decode(self, data):
		return self.model.dec_func(data)
