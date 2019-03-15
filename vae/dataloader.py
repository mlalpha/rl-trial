# load images from mp4 with openCV2
import os
import cv2

class dataloader(object):
	"""docstring for dataloader"""
	def __init__(self, foldername="videos"):
		super(dataloader, self).__init__()
		self.fileLst = []
		for file in os.listdir(foldername):
			if file.endswith(".mp4"):
				self.fileLst.append(os.path.join(foldername, file))

	def nextFrame():
		for file in self.fileLst:
			cap = cv2.VideoCapture(file)
			while cap.isOpened():
				ret, frame = cap.read()
				if ret:
					yield frame
			cap.release()

