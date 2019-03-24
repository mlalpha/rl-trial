import torch
import torch.nn.functional as F

storage = []

# The reward in explore stage
# a = torch.randn(100, 128)
# b = torch.randn(100, 128)
# output = F.cosine_similarity(a, b)
def cosine_similarity(a, b):
	return F.cosine_similarity(a, b)


def state_store(s, n=0):
	global storage
	# save state and get the smallest distance
	a = state_similarity(s, n)
	if s not in storage:
		storage.append(s)
	return a


def state_similarity(s, n=5):
	a = []
	global storage
	for x in storage:
		a.append(cosine_similarity(x, s))
	a.sort()
	if n:
		return a[:n]
	return a


def init():
	global storage
	storage = []

def torch_save_model(model):
	torch.save(model, 'model.pt')