import torch
import torch.nn.functional as F
import math

state_storage = []
reward_storage = []

# The reward in explore stage
# a = torch.randn(100, 128)
# b = torch.randn(100, 128)
# output = F.cosine_similarity(a, b)
def cosine_similarity(a, b):
	return F.cosine_similarity(a, b)


def new_episode():
	global state_storage, reward_storage
	state_storage.append([])
	reward_storage.append([])


def state_store(s, n=0):
	global state_storage
	# save state and get the smallest distance
	a = state_similarity(s, n)
	# s_not_in_state_storage = True
	# for episode in state_storage:
	# 	if s in episode:
	# 		s_not_in_state_storage = False
	# if s_not_in_state_storage:
	# 	state_storage[-1].append(s)
	state_storage[-1].append(s)
	return a


def reward_store(r):
	global reward_storage
	reward_storage.append(r)


def _search_none_zero_elements(l):
	r = []
	j = 0
	for i in l:
		if i:
			r.append(j, i)
		j += 1
	return r


def get_episode(e=-1):
	global state_storage, reward_storage
	# calc reward_storage[e] to slope/tangent
	r = _search_none_zero_elements(reward_storage[e])
	reward = []
	for i in r:
		c = math.pi / i[0]
		if i[1] < 0:
			f = math.sin
		else:
			f = math.cos
		for j in range(1, i[0]*2):
			reward[x] += f(c*j) * i[1]
	reward_slope = []
	for i in range(len(reward_storage[e])):
		reward_slope.append(reward[i + 1] - reward[i])
	return state_storage[e], reward_slope


def get_reward(ep=-1, index=-1):
	global reward_storage
	return reward_storage[ep][index]


def state_similarity(s, n=5):
	a = []
	global state_storage
	for episode in state_storage:
		for x in episode:
			a.append(cosine_similarity(x, s))
	a.sort()
	if n:
		return a[:n]
	return a


def init():
	global state_storage, reward_storage
	state_storage = [[]]
	reward_storage = [[]]


def torch_save_model(model):
	torch.save(model, 'model.pt')
