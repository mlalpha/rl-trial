# choose model1 or model2
# train or use

from dataloader import dataloader
import model1
import torch

class vae_module(object):
	"""docstring for vae_module"""
	def __init__(self, num_latent, state_size, img_trans=None,
				dataset_folder="videos", dataset_format="mp4",
				filter_size=[3, 3, 3], channels=[1, 4, 20, 20]):
		super(vae_module, self).__init__()
		# init trainloader
		self.trainloader = dataloader(dataset_folder,
									dataset_format, img_trans)
		self.model = model1.VAE(num_latent,
								state_size, filter_size,
								channels)
		
	def train(iters=26, num_latent=8, print_every=5):
	    #print after every 5 iterations
		# model = VAE(num_latent, state_size)

		device = ('cuda' if torch.cuda.is_available() else 'cpu')
		import torch.optim as optim
		optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

		self._train(iters, device, optimizer, print_every)
		# train.train_model(self.model,
		# 	self.trainloader, iters,
		# 	num_latent, print_every)

	######The function which we will call for training our model

	def _train(iters, device, optimizer, print_every):
		counter = 0
		for i in range(iters):
			self.model.train()
			self.model.to(device)
			for images, _ in self.trainloader:
				images = images.to(device)
				optimizer.zero_grad()
				out, mean, logvar = self.model(images)
				loss = self.VAE_loss(out, images, mean, logvar)
				loss.backward()
				optimizer.step()
				
			if(counter % print_every == 0):
				print(loss.numpy().sum())

			counter += 1

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

	def save(path="vae.pkl"):
		torch.save(self.model.state_dict(), path)

	def load(path="vae.pkl"):
		self.model.load_state_dict(torch.load(path))

	def encode(data):
		self.model.enc_func(data)
