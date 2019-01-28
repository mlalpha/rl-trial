
# coding: utf-8

# In[1]:

import numpy as np
import random

EPOCHS_NUM = 4000000
BUFFER_SIZE = 10000000
GAMMA = 0.99
INITIAL_EPISLON = 1.0
FINAL_EPISLON = 0.1
FINAL_EXPLORATION_FRAME = 10000000
REPLAY_START_SIZE = 50000

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img)).reshape(105, 80, 1)

def transform_reward(reward):
    return np.sign(reward)


# In[2]:

# Define the model from online
import keras
from keras.layers import Input, Lambda, Convolution2D, Flatten, Dense, Multiply
from keras.models import Model
def atari_model(n_actions):
    
    # Channel last
    ATARI_SHAPE = (105, 80, 4)

    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    
    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.Convolution2D(
        16, 8, strides=(4, 4), activation='relu'
    )(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.Convolution2D(
        32, 4, strides=(2, 2), activation='relu'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.Multiply() ([output, actions_input])

    model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')
    
    model.summary()
    return model


# In[3]:

import random
import numpy as np

'''
ReplayBuffer Class:
the implementation of the replay buffer for
storing the previous episodes information
with 5 tuple (s, a, r, s', d)
based on tensorflow implementaion with modification
Planned extention:
1. Implementation of Prioritized Experience Buffer
'''
class ReplayBuffer(object):

    # initialize Buffer with the maximimum size
    def __init__(self, max_size):
        self.max_size = max_size
        self.cur_index = 0
        self.cur_size = 0
        self.buffer = {}

    # add the new sample into the buffer
    def add_sample(self, sample):
        self.buffer[self.cur_index] = sample
        self.cur_index = (self.cur_index + 1) % self.max_size
        self.cur_size = min(self.cur_size + 1, self.max_size)

    # get the batch of according to size
    def get_batch(self, batch_size):
        batch_size = min(self.cur_size, batch_size)
        idxs = random.sample(range(self.cur_size), batch_size)

        result = zip(*[self.buffer[idx] for idx in idxs])
        result = [np.array(list(r)) for r in result]
        return result


# In[4]:

def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
    """Do one deep Q learning iteration.
    
    Params:
    - model: The DQN
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal
    
    """
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    model.fit(
        [start_states, actions], actions * Q_values[:, None],
        nb_epoch=1, batch_size=len(start_states), verbose=0)


# In[5]:

def choose_best_action(model, state):
    
    all_actions_Q = model.predict([state.reshape(1, 105, 80, 4), np.ones((state.shape[0], 4))])
    
    result_action = np.argmax(all_actions_Q, axis =1)
    
    print(result_action[0])
    
    return result_action


# In[17]:

def q_iteration(env, model, state, iteration, memory):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)

    # Choose the action 
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, state)

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    next_frame, reward, is_done, _ = env.step(action)
    new_frame = preprocess(next_frame)
    action_arr = np.zeros(4)
    action_arr[action] = 1
    action = action_arr
    clipped_reward = transform_reward(reward)
    saving_frame = np.copy(state[:,:,1:])
#     print(saving_frame.shape, new_frame.shape)
    saving_frame = np.concatenate([saving_frame, new_frame], axis = 2)
    print(saving_frame.shape)
    
    memory.add_sample((state, action, clipped_reward, saving_frame, is_done))

    # Sample and fit
    start_states, actions, rewards, next_states, is_terminal = memory.get_batch(32)
    fit_batch(model, gamma= GAMMA, start_states = start_states, actions = actions,
              rewards = rewards, next_states = next_states, is_terminal = is_terminal)
    
    return reward, saving_frame, is_done


# In[7]:



def get_epsilon_for_iteration(iteration, replay_start_size = REPLAY_START_SIZE,
                              final_exploration_frame = FINAL_EXPLORATION_FRAME,
                              initial_epislon = INITIAL_EPISLON, final_epislon = FINAL_EPISLON):
    
    # the number of iteractions
    if iteration < replay_start_size:
        return 1.0
    else:
        return max(final_epislon, initial_epislon - float(iteration - replay_start_size) / (final_exploration_frame - replay_start_size) * (initial_epislon - final_epislon)) 
    


# In[ ]:




# In[8]:

# Trial of environment
# # Import the gym module
# import gym

# # Create a breakout environment
# env = gym.make('BreakoutDeterministic-v4')
# # Reset it, returns the starting frame
# frame = env.reset()
# # Render
# env.render()

# is_done = False
# while not is_done:
#   # Perform a random action, returns the new frame, reward and whether the game is over
#   frame, reward, is_done, _ = env.step(env.action_space.sample())
#   # Render
#   env.render()


# In[18]:


# Import the gym module
import gym

# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')
# Reset it, returns the starting frame


memory = ReplayBuffer(BUFFER_SIZE)
model = atari_model(env.action_space.n)
itr = 0
for i in range(EPOCHS_NUM):
    
    
    frame = env.reset()
    frame = preprocess(frame)
    frame = [np.copy(frame) for _ in range(4)]
    frame = np.concatenate(frame, axis = 2)
    is_done = False
    total_rewards = 0
    while not is_done:
        
#         print(frame.shape)
        reward, frame, is_done = q_iteration(env = env, model = model,
                    state = frame, iteration = itr, memory = memory)
        total_rewards += reward
        itr += 1
    
    print('ep %d : %lf' %(i, total_rewards))

        


# In[ ]:

import gym
env = gym.make('BreakoutDeterministic-v4')


# In[ ]:

print(env.action_space.n)

