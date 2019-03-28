import torch
import torch.nn as nn
import torch.nn.functional as F
import math

########Definition of the architecture of our encoder and decoder model with all the assisting functions

class VAE(nn.Module):
	def __init__(self, num_latent, state_size, filter_size=[3, 3, 3, 3], channels=[1, 4, 20, 128, 128], dconv_filter_size=[3, 8, 15, 42]):
		super().__init__()
		
		#So here we will first define layers for encoder network
		self.encoder = nn.Sequential(nn.Conv2d(channels[0], channels[1], filter_size[0], padding=1),
									nn.MaxPool2d(2, 2),
									nn.BatchNorm2d(channels[1]),
									nn.Conv2d(channels[1], channels[2], filter_size[1], padding=1),
									nn.MaxPool2d(2, 2),
									nn.BatchNorm2d(channels[2]),
									nn.Conv2d(channels[2], channels[3], filter_size[2], padding=1),
									nn.MaxPool2d(2, 2),
									nn.BatchNorm2d(channels[3]),
									nn.Conv2d(channels[3], channels[4], filter_size[3], padding=1))
		
		#These two layers are for getting logvar and mean
		encoder_out_size = state_size // 64
		encoder_out_width = int(math.sqrt(encoder_out_size))
		self.fc1_in_shape = [-1, channels[-1], encoder_out_width, encoder_out_width]
		self.fc1_in_size = channels[-1] * encoder_out_size
		fc2_in_size = self.fc1_in_size // 3
		fc2_out_size = fc2_in_size // 2
		self.fc1 = nn.Linear(self.fc1_in_size, fc2_in_size)
		self.fc2 = nn.Linear(fc2_in_size, fc2_out_size)
		self.mean = nn.Linear(fc2_out_size, num_latent)
		self.var = nn.Linear(fc2_out_size, num_latent)
		
		#######The decoder part
		#This is the first layer for the decoder part
		self.expand = nn.Linear(num_latent, fc2_out_size)
		self.fc3 = nn.Linear(fc2_out_size, fc2_in_size)
		self.fc4 = nn.Linear(fc2_in_size, self.fc1_in_size)
		self.decoder = nn.Sequential(nn.ConvTranspose2d(channels[4], channels[3], dconv_filter_size[0], padding=1),
									nn.BatchNorm2d(channels[3]),
									nn.ConvTranspose2d(channels[3], channels[2], dconv_filter_size[1]),
									nn.BatchNorm2d(channels[2]),
									nn.ConvTranspose2d(channels[2], channels[1], dconv_filter_size[2]),
									nn.BatchNorm2d(channels[1]),
									nn.ConvTranspose2d(channels[1], channels[0], dconv_filter_size[3]),
									nn.Sigmoid())
		
	def enc_func(self, x):
		#here we will be returning the logvar(log variance) and mean of our network
		x = self.encoder(x)
		x = x.view([-1, self.fc1_in_size])
		x = F.dropout2d(self.fc1(x), 0.5)
		x = self.fc2(x)
		
		mean = self.mean(x)
		logvar = self.var(x)
		return mean, logvar
	
	def dec_func(self, z):
		#here z is the latent variable state
		z = self.expand(z)
		z = F.dropout2d(self.fc3(z), 0.5)
		z = self.fc4(z)
		z = z.view(self.fc1_in_shape)
		
		out = self.decoder(z)
		out = torch.sigmoid(out)
		return out
	
	def get_hidden(self, mean, logvar):
		if self.training:
			std = torch.exp(0.5*logvar)   #So as to get std
			noise = torch.randn_like(mean)   #So as to get the noise of standard distribution
			return noise.mul(std).add_(mean)
		else:
			return mean
	
	def forward(self, x):
		mean, logvar = self.enc_func(x)
		z = self.get_hidden(mean, logvar)
		out = self.dec_func(z)
		return out, mean, logvar
