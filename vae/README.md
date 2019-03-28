# VAE
VAE model1 finished, dataloader finished

For model to train, save and load not finished

## dataloader
require opencv-python version 3.1 or above

## model1
require pytorch

A simple VAE in CNN + FC  
Based on https://github.com/aniket-agarwal1999/VAE-Pytorch

### encoder_model_0.pt
Requires:
```
dconv_kernel_sizes = [7, 21, 40]
```

## model3
require pytorch

A VAE for review experience and encourage explore use RNN  
Take reference on https://github.com/aniket-agarwal1999/VAE-Pytorch

## test.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## input data
1 x 80 x 80?  
current using 1 x 88 x 88

## How to use VAE module?

The default relative path the module will search for video is `videos/`  

The video format is expected to be `*.mp4`  

```

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
		pic = resize(pic, size)
		pic = img_transform(pic)
		return torch.from_numpy(pic)

	def rgb2gray(img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	def img_transform(img):
		gray_image = rgb2gray(img)
		small_image = resize(gray_image, size=(IMAGE_SIZE, IMAGE_SIZE))
		matrix = small_image / 255.0
		return np.expand_dims(matrix, axis=0)

	test_image = load_test_image()

	img_size = IMAGE_SIZE**2
	num_latent = 5
	
	# initialize the module and provide an image transform function
	vae = vae_module(num_latent, img_size, img_transform)
	# or you can let it use raw image
	vae = vae_module(num_latent, img_size)

	# to start training
	vae.train(26, num_latent, 5)

	# encode a photo
	output = vae.encode(test_image)

	# save the model
	vae.save()

use_vae()

```

## How to use this package?
import
