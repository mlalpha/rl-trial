
from vae import vae_module
import cv2
from dataloader import dataloader

def use_vae():
	def resize(img, size):
		return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

	def load_test_image(path='./test_img.png', size=(80,80)):
		pic = cv2.imread(path)
		return resize(pic, size)

	def rgb2gray(img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	def img_transform(img):
		gray_image = rgb2gray(img)
		return resize(gray_image, size=(80, 80))

	test_image = load_test_image()
	test_image = img_transform(test_image)

	img_size = 81*81

	vae = vae_module(10, img_size, img_transform)
	vae.train(26, 10, 5)
	print(vae.encode(test_image))

	vae.save()

use_vae()
