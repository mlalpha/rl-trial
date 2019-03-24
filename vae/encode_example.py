
from vae import vae_module
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def use_vae():
	IMAGE_SIZE = 84
	x = []

	def resize(img, size):
		return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

	def load_test_image(path='./test_img.png', size=(36, 36)):
		pic = cv2.imread(path)
		if pic is None:
			return None
		return resize(pic, size)

	def rgb2gray(img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	def img_transform(img):
		gray_image = rgb2gray(img)
		matrix = resize(gray_image, size=(IMAGE_SIZE, IMAGE_SIZE))
		matrix = matrix / 255.0
		return matrix.reshape(1, matrix.shape[0], matrix.shape[1]).astype(np.float32)

	test_image = load_test_image(size=(IMAGE_SIZE, IMAGE_SIZE))
	test_image = img_transform(test_image)
	test_image = torch.from_numpy(test_image)

	img_size = IMAGE_SIZE**2
	num_latent = 1024
	dconv_kernel_sizes = [7, 21, 40]

	vae = vae_module(num_latent, img_size, img_transform, dconv_kernel_sizes=dconv_kernel_sizes, train=False)
	vae.load()
	test_image.reshape(1, test_image.shape[0], test_image.shape[1], test_image.shape[2])
	l = vae.encode(test_image)
	print(l)
	# plt.imshow(vae.decode(l).data.cpu().numpy(), cmap='bone')


use_vae()
