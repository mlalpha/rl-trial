import math
from rnn_gru import RNN
import utilities
import numpy as np

MAX_REWARD = 100
rewards = []
model = None
hidden = None
optimizer = None

def reward_trans(old_reward, state):
	global rewards
	# store this state and get some cloesest distance (smallest)
	k = 4000
	new_reward = np.asarray(utilities.store(state, k)).mean() * MAX_REWARD

	pre_reward = rewards[-1]
	rewards.append(old_reward)
	old_reward = pre_reward * 0.9999 + math.abs(pre_reward - old_reward)

	# merge new & old reward
	new_reward *= old_reward
	# GRU predict bad reward? (window size 30)
	# for 30 frame, train
    # train(state, reward_record)

	return new_reward


def reward_init():
	global rewards, model, hidden, optimizer
	rewards = []
	# release GRU
	utilities.init()

	lr = 0.01
	model = RNN(features=features, cls_size=len(chars))
	if cuda:
	    model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    hidden = model.initHidden()


def train_rnn(input_data, reward_data):
	global model, hidden, optimizer
        model.zero_grad()
        output, hidden = model(input_data, var(hidden.data))
        loss = criterion(output, reward_data)
        loss.backward()
        optimizer.step()
        return output
