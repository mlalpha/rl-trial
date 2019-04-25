import math
from rnn_gru import RNN
import utilities
import numpy as np

MAX_REWARD = 1
# WIN = 
# LOSS = 
# FOUL = 
END_GAME = [WIN, LOSS, FOUL]
reward = 0
is_train = True
model = None
hidden = None
optimizer = None
largest_reward = 0
criterion = None

def reward_trans(raw_reward, state):
	# GRU predict reward train here
	if state in END_GAME:
		# if FOUL then end game, can also be FOUL then choose second best action
		if state is WIN:
			utilities.reward_store(MAX_REWARD)
		else:
			utilities.reward_store(-MAX_REWARD)
		for step, slope in zip(utilities.get_episode()):
		    # train(state, reward_record)
			train(step, slope)
		utilities.new_episode()
		return raw_reward

	global model, reward, is_train
	# store this state and get some cloesest distance (smallest)
	k = 4000
	curiosity_reward = np.asarray(utilities.state_store(state, k)).mean() * MAX_REWARD

	suprise_reward = 0
	# if model predict slope * MAX_REWARD < 0 and something

	utilities.reward_store(raw_reward)
	env_reward = 0 # GRU predict reward slope
	if is_train:
		reward = reward * 0.9999 + math.abs(reward - env_reward)
	else:
		reward = some how suprise_reward

	# merge new & old reward
	reward *= curiosity_reward

	if reward > largest_reward:
		largest_reward = reward
	reward /= largest_reward

	return reward


def reward_init():
	global model, hidden, optimizer, criterion
	# release GRU
	utilities.init()

	lr = 0.01
	model = RNN(features=features, cls_size=len(chars))
	if cuda:
	    model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()
    hidden = model.initHidden()


def train_rnn(input_data, reward_data):
	global model, hidden, optimizer, criterion
	model.zero_grad()
	output, hidden = model(input_data, var(hidden.data))
	loss = criterion(output, reward_data)
	loss.backward()
	optimizer.step()
	return output
