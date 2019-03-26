
from vae import vae_module
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def use_vae():
	IMAGE_SIZE = 84
	x = []

	def resize(img, size):
		return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

	def load_test_image(path='./test_img.png', size=(36, 36)):
		pic = cv2.imread(path)
		if pic is None:
			raise 'Error, ' + path + ' not found'
		img = resize(pic, size)
		return img_transform(img)

	def rgb2gray(img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	def img_transform(img):
		gray_image = rgb2gray(img)
		matrix = resize(gray_image, size=(IMAGE_SIZE, IMAGE_SIZE))
		matrix = matrix / 255.0
		return matrix.reshape(1, matrix.shape[0], matrix.shape[1]).astype(np.float32)

	test_image = load_test_image(size=(IMAGE_SIZE, IMAGE_SIZE))
	test_image = np.expand_dims(test_image, axis=0)
	test_input = torch.from_numpy(test_image)
	plt.figure()
	plt.imshow(test_image.reshape(test_image.shape[2:]), cmap='bone')

	img_size = IMAGE_SIZE**2
	num_latent = 1024
	dconv_kernel_sizes = [3, 18, 47]

	vae = vae_module(num_latent, img_size, img_transform, dconv_kernel_sizes=dconv_kernel_sizes, train=False)
	vae.load()
	l = vae.encode(test_input)
	print(l)
	dec_img = vae.decode(l).data.cpu().numpy()
	dec_img = dec_img.reshape(dec_img.shape[2:])*255
	plt.figure()
	plt.imshow(dec_img.astype(np.int), cmap='bone')
	plt.show()


use_vae()
