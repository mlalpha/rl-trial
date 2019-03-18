# main.py

from vae import vae_module

def main():
	def load_test_image():
		pass

	def resize(img, size):
		pass # your code here

	def rgb2gray(rgb):
		return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

	def img_transform(img):
		gray_image = rgb2gray(image)
		return resize(gray_image, size=(81, 81))

	test_image = load_test_image()

	vae = vae_module(img_transform)
	vae.train(26, 10, 5)
	vae.encode(test_image)


if __name__ == '__main__':
	main()