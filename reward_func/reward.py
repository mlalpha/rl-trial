import math
from rnn_gru import RNN
import utilities
import numpy as np

MAX_REWARD = 1
WIN = 0
LOSS = 1
FOUL = 2
END_GAME = [WIN, LOSS, FOUL]
reward = 0
is_train = True
model = None
hidden = None
optimizer = None
largest_reward = 0
criterion = None
LR = 0.01

def reward_trans(raw_reward, state):
	# GRU predict reward train here
	if state in END_GAME:
		# if FOUL then end game, can also be FOUL then choose second best action
		if state == WIN:
			utilities.reward_store(MAX_REWARD)
		else:
			utilities.reward_store(-MAX_REWARD)
		states, rewards = utilities.get_episode()
		for step, reward in zip(states, rewards):
			train_rnn(step, reward)
		utilities.new_episode()
		return raw_reward

	global model, reward
	# store this state and get some cloesest distance (smallest)
	k = 4000
	curiosity_reward = np.asarray(utilities.state_store(state, k)).mean() * MAX_REWARD
	store_reward = raw_reward

	env_reward = 0 # GRU predict reward slope
	reward = curiosity_reward#reward * 0.9999 + math.abs(reward - env_reward)
	if not is_train:
		# states = utilities.get_episode()[-100:]
		# if len(states) < seq_len:
		# 	states.unshift(0)
		predicted_reward = model(state, var(hidden.data))
		if suprised: # suprised
			reward += sth # suprise_reward
			store_reward = suprise_reward

	if reward > largest_reward:
		largest_reward = reward
	reward /= largest_reward
	if largest_reward > MAX_REWARD:
		largest_reward *= 0.999999

	utilities.reward_store(store_reward)
	return reward


def reward_replay():
	pass
	

def reward_init(state_size, win_state, loss_state, foul_state,
				rnn_hidden_size=32, rnn_num_layers=2):
	global model, hidden, optimizer, criterion, WIN, LOSS, FOUL

	WIN = win_state
	LOSS = loss_state
	FOUL = foul_state
	# release GRU
	utilities.init()

	model = RNN(state_size, rnn_hidden_size, rnn_num_layers)
	if cuda:
	    model.cuda()
	criterion = nn.CrossEntropyLoss() #nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9) #Adam(model.parameters(), lr=LR)
    hidden = model.initHidden()


def train_rnn(input_data, reward_data):
	global model, hidden, optimizer, criterion
	# model.zero_grad()
	output, hidden = model(input_data, var(hidden.data))
	loss = criterion(output, reward_data)
	loss.backward()
	optimizer.step()
	return output

def predict_rnn(input_data):
	output, _ = model(input_data, var(hidden.data))
	return output
