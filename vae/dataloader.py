# load images from mp4 with openCV2
# refer to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class dataloader(Dataset):
	"""docstring for dataloader"""
	def __init__(self, folder_name="videos", filename_extension=".mp4", transform=None):
		super(dataloader, self).__init__()
		self.transform = transform
		self.fileLst = []
		for file in os.listdir(folder_name):
			if file.endswith(filename_extension):
				self.fileLst.append(os.path.join(folder_name, file))

		self.filenframe_idx = []
		self.len = 0
		fileIdx = 0
		for filename in self.fileLst:
			cap = cv2.VideoCapture(filename)
			frameIdx = 0
			while cap.isOpened():
				ret, frame = cap.read()
				if frame is None:
					break
				if ret and np.sum(frame) != 0:
					self.len += 1
					self.filenframe_idx.append([fileIdx, frameIdx])
					frameIdx += 1
			cap.release()
			fileIdx += 1

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		indexTuple = self.filenframe_idx[idx]
		fileIdx = indexTuple[0]
		frameIdx = indexTuple[1]
		filename = self.fileLst[fileIdx]
		cap = cv2.VideoCapture(filename)
		if cap.isOpened():
			cap.set(cv2.CAP_PROP_POS_FRAMES,frameIdx)
			ret, frame = cap.read()
			cap.release()
			if ret:
				if self.transform:
					frame = self.transform(frame)
				return frame # we ignored label here # return image, label


class buffer_dataloader(Dataset):
	"""docstring for buffer_dataloader"""
	def __init__(self, folder_name="videos", filename_extension=".mp4", transform=None,
				buffer_folder_name="buffer"):
		super(buffer_dataloader, self).__init__()
		self.transform = transform
		fileLst = []
		for file in os.listdir(folder_name):
			if file.endswith(filename_extension):
				fileLst.append(os.path.join(folder_name, file))

		if not os.path.exists(buffer_folder_name):
			os.makedirs(buffer_folder_name)
		self.fileLst = []
		filenum = 0
		for filename in fileLst:
			filenum += 1
			cap = cv2.VideoCapture(filename)
			count = 0
			while cap.isOpened():
				ret, frame = cap.read()
				if frame is None:
					break
				if ret:
					path = os.path.join(buffer_folder_name, str(filenum) + str(count) + ".png")
					if not os.path.exists(path):
						cv2.imwrite(path, frame)
					self.fileLst.append(path)
					count += 1
			cap.release()

	def __len__(self):
		return len(self.fileLst)

	def __getitem__(self, idx):
		img_name = self.fileLst[idx]
		image = cv2.imread(img_name)
		image = self.transform(image)

		return image # we ignored label here # return image, label
