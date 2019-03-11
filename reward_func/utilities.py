import torch
import torch.nn.functional as F

storage = []

# The reward in explore stage
# a = torch.randn(100, 128)
# b = torch.randn(100, 128)
# output = F.cosine_similarity(a, b)
def cosine_similarity(a, b):
	return F.cosine_similarity(a, b)


def state_store(s):
	global storage
	a = state_similarity(s)
	storage.append(s)
	return a


def state_similarity(s, n=5):
	a = []
	global storage
	for x in storage:
		if x != s:
			a.append(cosine_similarity(x, s))
	a.sort()
	return a[n:]


def init():
	global storage
	storage = []
