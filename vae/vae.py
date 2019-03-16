# choose model1 or model2
# train or use

import dataloader
import model1
import model2
import train

class vae_module(object):
	"""docstring for vae_module"""
	def __init__(self, m="model1"):
		super(vae_module, self).__init__()
		# self.model = model1.VAE(...)
		
	def train(dataset_folder="videos", dataset_format="mp4"):
		pass

	def save(path="vae.pkl"):
		pass

	def load(path="vae.pkl"):
		pass

	def encode(data):
		pass
