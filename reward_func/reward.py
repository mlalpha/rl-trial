import math
from rnn_gru import RNN
import utilities
import numpy as np

MAX_REWARD = 1
WIN = 0
LOSS = 1
FOUL = 2
END_GAME = [WIN, LOSS, FOUL]
rewards = []
model = None
hidden = None
optimizer = None
largest_reward = 0
criterion = None
LR = 0.01

def reward_trans(raw_reward, state):
	# GRU predict reward train here
	utilities.reward_store(raw_reward)
	if state in END_GAME:
		# if FOUL then end game, can also be FOUL then choose second best action
		if state == WIN:
			utilities.reward_store(MAX_REWARD)
		else:
			utilities.reward_store(-MAX_REWARD)
		states, raw_rewards = utilities.get_episode()
		rewards = _reward_replay(states, raw_rewards)
		for step, reward in zip(states, rewards):
			train_rnn(step, reward)
		utilities.new_episode()

		return rewards
	return None


def _reward_replay(states, rewards): # for first time internal replay
	baseReward = reward_to_wave(rewards)

	predicted_rewards = []
	for state in states:
		predicted_rewards.append(predict_rnn(state))
	supriseReward = baseReward - np.asarray(predicted_rewards)

	curiosity_rewards = []
	for state, reward, predicted_reward in zip(states, rewards, predicted_rewards[1:]):
		# store this state and get some cloesest distance (smallest)
		k = 4000
		curiosity = np.asarray(utilities.state_store(state, k)).mean() * MAX_REWARD
		curiosity_rewards.append(curiosity)
	curiousReward = np.asarray(curiosity_rewards)

	finalReward = curiousReward + supriseReward

	if np.max(finalReward) > largest_reward:
			largest_reward = np.max(finalReward)
	positive_mask = finalReward > 0
	finalReward[positive_mask] /= largest_reward
	if largest_reward > MAX_REWARD:
		largest_reward *= 0.999999
	finalReward = np.clip(finalReward, -MAX_REWARD, MAX_REWARD)

	final_reward = finalReward.tolist()

	utilities.reward_replace(final_reward)
	return final_reward


def reward_replay(ep=-2):
	_, r = utilities.get_episode(ep)
	return r


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
