# VAE
VAE model1 finished, dataloader finished

For model to train, save and load not finished

## dataloader
require opencv-python version 3.1 or above

## model1
require pytorch

A simple VAE in CNN + FC  
Based on https://github.com/aniket-agarwal1999/VAE-Pytorch

## model3
require pytorch

A VAE for review experience and encourage explore use RNN  
Take reference on https://github.com/aniket-agarwal1999/VAE-Pytorch

<!-- ## input data -->
<!-- 1 x 81 x 81 or RAW? -->

## How to use VAE module?

The default relative path the module will search for video is `videos/`  

The video format is expected to be `*.mp4`  

```

from vae import vae_module

def use_vae():
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

	# initialize the module and provide an image transform function
	vae = vae_module(img_transform)
	# or you can let it use raw image
	vae = vae_module()

	# to start training
	vae.train(26, 10, 5)

	# encode a photo
	output = vae.encode(test_image)

	# save the model
	vae.save("vae_model.pkl")

```

## How to use this package?
import
