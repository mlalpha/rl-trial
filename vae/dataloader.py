# load images from mp4 with openCV2
# refer to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from torch.utils.data import Dataset
import os
import cv2
import pickle

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
			if cap.isOpened():
				self.len += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
				for frameIdx in range(0, self.len - 1):
					self.filenframe_idx.append([fileIdx, frameIdx])
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
