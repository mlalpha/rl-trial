
from vae import vae_module
import cv2
import numpy as np
import torch

def use_vae():
	IMAGE_SIZE = 12

	def resize(img, size):
		return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

	def load_test_image(path='./test_img.png', size=(IMAGE_SIZE,IMAGE_SIZE)):
		pic = cv2.imread(path)
		return resize(pic, size)

	def rgb2gray(img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	def img_transform(img):
		gray_image = rgb2gray(img)
		small_image = resize(gray_image, size=(IMAGE_SIZE, IMAGE_SIZE))
		matrix = small_image / 255.0
		# return np.expand_dims(matrix, axis=0).astype(np.double)
		return np.expand_dims(matrix, axis=0)

	test_image = load_test_image()
	test_image = img_transform(test_image)
	test_image = torch.from_numpy(test_image)

	img_size = IMAGE_SIZE**2
	num_latent = 5

	vae = vae_module(num_latent, img_size, img_transform)
	vae.train(26, num_latent, 5)
	print(vae.encode(test_image))

	vae.save()

use_vae()
