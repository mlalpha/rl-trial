# choose model1 or model2
# train or use

from dataloader import dataloader
import model1
import train
import torch

class vae_module(object):
	"""docstring for vae_module"""
	def __init__(self, img_trans=None, dataset_folder="videos",
				dataset_format="mp4", num_latent, state_size,
				filter_size=[3, 3, 3], channels=[1, 4, 20, 20]):
		super(vae_module, self).__init__()
		# init trainloader
		self.trainloader = dataloader(dataset_folder,
									dataset_format, img_transform)
		self.model = model1.VAE(num_latent,
								state_size, filter_size,
								channels=)
		
	def train(iters=26, num_latent=8, print_every=5):
		train.train_model(self.model,
			self.trainloader, iters,
			num_latent, print_every)

	def save(path="vae.pkl"):
		torch.save(self.model.state_dict(), path)

	def load(path="vae.pkl"):
		self.model = model1.VAE(num_latent,
								state_size, filter_size,
								channels=)
		self.model.load_state_dict(torch.load(path))

	def encode(data):
		self.model.enc_func(data)
