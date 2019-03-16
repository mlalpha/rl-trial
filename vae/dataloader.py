# load images from mp4 with openCV2
from torch.utils.data import Dataset
import os
import cv2

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
				for frameIdx in xrange(0, self.len):
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
			cap.set(CV_CAP_PROP_POS_FRAMES,frameIdx)
			ret, frame = cap.read()
			cap.release()
			if ret:
				if self.transform:
					frame = self.transform(frame)
				return frame


class buffer_dataloader(Dataset):
	"""docstring for buffer_dataloader"""
	def __init__(self, folder_name="videos", filename_extension=".mp4",
				buffer_folder_name="buffer", transform=None):
		super(buffer_dataloader, self).__init__()
		self.transform = transform
		fileLst = []
		for file in os.listdir(folder_name):
			if file.endswith(filename_extension):
				fileLst.append(os.path.join(folder_name, file))

		self.fileLst = []
		for filename in fileLst:
			cap = cv2.VideoCapture(filename)
			count = 0
			while cap.isOpened():
				ret, frame = cap.read()
				if ret:
					path = os.path.join(buffer_folder_name, filename + count + ".png")
					cv2.imwrite(path, frame)
					self.fileLst.append(path)
					count += 1
			cap.release()

	def __len__(self):
		return len(self.fileLst)

	def __getitem__(self, idx):
		img_name = self.fileLst[idx]
		image = cv2.imread(img_name)

		if self.transform:
			image = self.transform(image)

		return image

		